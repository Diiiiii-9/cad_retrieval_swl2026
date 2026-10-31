"""
Unsupervised Training Loop for CAD Model Retrieval
==================================================
@Author: Di Liu
@Date: 2026-05-24
@Description:
Trains the GINE-based CAD Graph Encoder using Contrastive Learning.
Generates geometric embeddings by maximizing the similarity between 
augmented views of the same CAD graph while minimizing similarity 
with other graphs in the batch.
"""

import os
import argparse
import pathlib
import logging
import json
import random
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge

# Import custom modules
from src.dataset import CADGraphDataset
from src.networks import CADGraphEncoder, CADContrastiveModel
from src.features import FEATURE_MODES, get_transform, noise_dims_for
from src.run_config import save_run_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==========================================
# Data Augmentation Strategy
# ==========================================
def augment_graph_batch(batch, edge_drop_rate=0.15, feature_noise_std=0.02, aug_mode="both", noise_dims=None):
    """
    Creates an augmented view of a PyG Batch for contrastive learning.
    
    1. Edge Dropout: Randomly removes a percentage of edges to force the 
       network to not rely entirely on exact topology.
    2. Feature Noise: Adds slight Gaussian noise to node features to 
       simulate minor geometric scaling or translation differences.

    Args:
        aug_mode (str): Which augmentation(s) to apply. One of:
            "both"          -- edge dropout AND feature noise (original behavior)
            "edge_drop"     -- edge dropout only
            "feature_noise" -- feature noise only
        noise_dims (int or None): If set, Gaussian noise is only added to the first
            `noise_dims` node-feature columns (the continuous ones). Used with
            --features normalized so that one-hot category columns stay clean.
            None = add noise to every column (original behaviour).
    """
    if aug_mode not in ("both", "edge_drop", "feature_noise"):
        raise ValueError(f"Unknown aug_mode: {aug_mode!r}. Must be 'both', 'edge_drop', or 'feature_noise'.")

    # Clone batch to avoid modifying the original data reference
    aug_batch = batch.clone()
    
    # 1. Edge Dropout (Ensure force_undirected to keep the graph valid)
    if aug_mode in ("both", "edge_drop"):
        if aug_batch.edge_index is not None and aug_batch.edge_index.numel() > 0:
            edge_index, edge_mask = dropout_edge(
                aug_batch.edge_index, 
                p=edge_drop_rate, 
                force_undirected=True,
                training=True
            )
            aug_batch.edge_index = edge_index
            if aug_batch.edge_attr is not None:
                aug_batch.edge_attr = aug_batch.edge_attr[edge_mask]
            
    # 2. Node Feature Noise
    if aug_mode in ("both", "feature_noise"):
        if aug_batch.x is not None:
            x = aug_batch.x.clone()
            cols = slice(None) if noise_dims is None else slice(0, noise_dims)
            x[:, cols] = x[:, cols] + torch.randn_like(x[:, cols]) * feature_noise_std
            aug_batch.x = x
        
    return aug_batch

# ==========================================
# InfoNCE Loss Function
# ==========================================
def info_nce_loss(z1, z2, temperature=0.1):
    """
    Calculates the InfoNCE (Contrastive) Loss.
    Args:
        z1: Embeddings of view 1 [Batch_size, Proj_Dim]
        z2: Embeddings of view 2 [Batch_size, Proj_Dim]
        temperature: Scaling factor to control the sharpness of the distribution.
    """
    batch_size = z1.shape[0]
    
    # Compute cosine similarity matrix between view 1 and view 2
    # Since z1 and z2 are already L2 normalized in the network, matmul is equivalent to cosine similarity.
    logits = torch.matmul(z1, z2.T) / temperature
    
    # The true pairs (positive examples) are on the diagonal of the similarity matrix
    labels = torch.arange(batch_size, dtype=torch.long, device=z1.device)
    
    # Cross Entropy pushes the diagonal values to 1 (high similarity) 
    # and all off-diagonal values to 0 (low similarity)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    
    # Symmetric loss
    return (loss_i + loss_j) / 2

# ==========================================
# Main Training Loop
# ==========================================
def set_seed(seed):
    """Makes runs repeatable (up to non-deterministic CUDA scatter ops in PyG)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_arg_parser(default_data_dir="./data/processed_graphs/mfcad", default_save_dir="./checkpoints",
                     description="Train CAD Contrastive Model"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to .pt files")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GNN hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="Final CAD embedding dimension")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for InfoNCE loss")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "max_mean"], help="Graph readout strategy: mean, max, or max_mean (concatenated)")
    parser.add_argument("--aug_mode", type=str, default="both", choices=["both", "edge_drop", "feature_noise"], help="Which contrastive augmentation(s) to apply")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GINE message-passing layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout inside the encoder")
    parser.add_argument("--features", type=str, default="raw", choices=list(FEATURE_MODES),
                        help="'raw' = original 4/3-dim features; 'normalized' = scale-invariant sizes + one-hot types (see src/features.py)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (run 3 seeds to report mean +- std)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    return parser


def train(argv=None, dataset_cls=CADGraphDataset, parser=None):
    """Trains one config. `argv=None` reads sys.argv; `dataset_cls`/`parser` let the
    Fusion 360 entry point reuse this exact loop with different defaults."""
    parser = parser or build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)

    # 1. Hardware setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Initializing training on device: {device}")

    # 2. Setup Directories
    save_path = pathlib.Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 3. Load Dataset
    logging.info("Loading dataset...")
    try:
        dataset = dataset_cls(root_dir=args.data_dir, transform=get_transform(args.features))
        node_dim, edge_dim = dataset.get_feature_dimensions()
        logging.info(f"Dataset loaded: {len(dataset)} graphs found.")
        logging.info(f"Feature mode: {args.features} -> Node Features: {node_dim}, Edge Features: {edge_dim}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return None

    if len(dataset) < args.batch_size:
        logging.warning("Dataset is smaller than batch size. Reducing batch size.")
        args.batch_size = max(2, len(dataset) // 2)

    # drop_last=True is CRITICAL for contrastive learning to ensure stable BatchNorm and Loss calculations
    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            generator=generator, num_workers=args.num_workers)

    # 4. Initialize Models
    encoder = CADGraphEncoder(
        node_in_dim=node_dim, 
        edge_in_dim=edge_dim, 
        hidden_dim=args.hidden_dim, 
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling
    )
    
    # Wrap encoder in Contrastive Model (adds the projection head)
    model = CADContrastiveModel(encoder, hidden_dim=args.out_dim, proj_dim=args.out_dim // 2).to(device)

    # Record exactly how this checkpoint was produced (read back by retrieval.py / app.py / evaluation)
    run_config = vars(args).copy()
    run_config.update({"node_in_dim": node_dim, "edge_in_dim": edge_dim, "num_graphs": len(dataset),
                       "dataset_class": dataset_cls.__name__})
    save_run_config(save_path, run_config)
    noise_dims = noise_dims_for(args.features)
    
    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing smoothly decays the learning rate to zero, improving final convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. Training Loop
    best_loss = float('inf')
    history = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar for batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Step A: Generate two augmented views of the batch
            view1 = augment_graph_batch(batch, edge_drop_rate=0.1, feature_noise_std=0.01, aug_mode=args.aug_mode, noise_dims=noise_dims)
            view2 = augment_graph_batch(batch, edge_drop_rate=0.2, feature_noise_std=0.03, aug_mode=args.aug_mode, noise_dims=noise_dims)
            
            # Step B: Forward Pass
            optimizer.zero_grad()
            z1, z2 = model.forward_cl(view1, view2)
            
            # Step C: Calculate Contrastive Loss
            loss = info_nce_loss(z1, z2, temperature=args.temp)
            
            # Step D: Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            
        # Step E: Scheduler Update & Logging
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_epoch_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        history.append({"epoch": epoch, "loss": avg_epoch_loss, "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(time.time() - start_time, 1)})
        # Written every epoch so a killed job still leaves a usable loss curve for the report
        with open(save_path / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)
        
        # Step F: Save Best Checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(), # We only care about saving the Encoder!
                'model_state_dict': model.state_dict(),           # Save full model just in case we want to resume training
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path / "best_contrastive_model.pth")
            logging.info(f"[IMPORTANT] New best model saved at epoch {epoch}")

    # Save final model
    torch.save(model.encoder.state_dict(), save_path / "final_encoder_weights.pth")
    logging.info(f"[IMPORTANT] Training Complete! Final encoder saved to {save_path / 'final_encoder_weights.pth'}")
    return save_path

if __name__ == "__main__":
    train()