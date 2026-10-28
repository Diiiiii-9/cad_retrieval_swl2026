"""
Unsupervised Training Loop for Fusion 360
=========================================
@Author: Ayse Seray Seker
@Date: 2026-10-22
@Description:
Fusion 360 counterpart of src/train_unsupervised.py. Loads graphs through dataset_fusion360.py
and builds the model through networks_fusion360.py. The augmentation and the InfoNCE loss are
imported unchanged from src/train_unsupervised.py, so both datasets are trained identically.
Weights are saved to checkpoints/fusion360, so MFCAD checkpoints are never overwritten.
"""

import argparse
import pathlib
import logging
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

# Import custom modules
from src.train_unsupervised import augment_graph_batch, info_nce_loss
from src.fusion360.dataset_fusion360 import Fusion360GraphDataset
from src.fusion360.networks_fusion360 import build_fusion360_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BEST_MODEL_FILENAME = "best_contrastive_model.pth"
FINAL_WEIGHTS_FILENAME = "final_encoder_weights.pth"

# ==========================================
# Main Training Loop
# ==========================================
def train():
    parser = argparse.ArgumentParser(description="Train Fusion 360 CAD Contrastive Model")
    parser.add_argument("--data_dir", type=str, default="./data/processed_graphs/fusion360", help="Path to .pt files")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/fusion360", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GNN hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="Final CAD embedding dimension")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for InfoNCE loss")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "max_mean"], help="Graph readout strategy: mean, max, or max_mean (concatenated)")
    parser.add_argument("--aug_mode", type=str, default="both", choices=["both", "edge_drop", "feature_noise"], help="Which contrastive augmentation(s) to apply")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GINE message-passing layers")
    args = parser.parse_args()

    # 1. Hardware setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Initializing Fusion 360 training on device: {device}")

    # 2. Setup Directories
    save_path = pathlib.Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 3. Load Dataset
    logging.info("Loading dataset...")
    try:
        dataset = Fusion360GraphDataset(root_dir=args.data_dir)
        node_dim, edge_dim = dataset.get_feature_dimensions()
        logging.info(f"Dataset loaded: {len(dataset)} graphs found.")
        logging.info(f"Detected Node Features: {node_dim}, Edge Features: {edge_dim}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    if len(dataset) < args.batch_size:
        logging.warning("Dataset is smaller than batch size. Reducing batch size.")
        args.batch_size = max(1, len(dataset) // 2)

    # drop_last=True is CRITICAL for contrastive learning to ensure stable BatchNorm and Loss calculations
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 4. Initialize Models (encoder wrapped in the contrastive model with its projection head)
    _, model = build_fusion360_model(
        dataset,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        proj_dim=args.out_dim // 2,
        num_layers=args.num_layers,
        pooling=args.pooling
    )
    model = model.to(device)

    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing smoothly decays the learning rate to zero, improving final convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. Training Loop
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Progress bar for batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in progress_bar:
            batch = batch.to(device)

            # Step A: Generate two augmented views of the batch
            view1 = augment_graph_batch(batch, edge_drop_rate=0.1, feature_noise_std=0.01, aug_mode=args.aug_mode)
            view2 = augment_graph_batch(batch, edge_drop_rate=0.2, feature_noise_std=0.03, aug_mode=args.aug_mode)

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

        # Step F: Save Best Checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),  # Only the encoder is used for retrieval
                'model_state_dict': model.state_dict(),            # Full model in case training is resumed
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path / BEST_MODEL_FILENAME)
            logging.info(f"[IMPORTANT] New best model saved at epoch {epoch}")

    # Save final model
    final_path = save_path / FINAL_WEIGHTS_FILENAME
    torch.save(model.encoder.state_dict(), final_path)
    logging.info(f"[IMPORTANT] Training Complete! Final encoder saved to {final_path}")

if __name__ == "__main__":
    train()
