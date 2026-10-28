"""
Graph Neural Network Wrapper for Fusion 360
===========================================
@Author: Ayse Seray Seker
@Date: 2026-10-22
@Description:
Reuses CADGraphEncoder and CADContrastiveModel from src/networks.py, but reads 
input feature dimensions from the Fusion 360 dataset instead of hardcoding them. 
This ensures the encoder adapts to the actual graph features and catches mismatches 
early, matching the pattern used in src/retrieval.py and src/train_unsupervised.py.
"""

# Import custom modules
from src.networks import CADGraphEncoder, CADContrastiveModel
from src.fusion360.dataset_fusion360 import Fusion360GraphDataset

# ==========================================
# Fusion 360 Model Factory
# ==========================================
def build_fusion360_model(dataset, hidden_dim=128, out_dim=128, proj_dim=64,
                          num_layers=3, dropout=0.2, pooling="mean"):
    """
    Builds the contrastive model sized for a Fusion 360 dataset.

    Args:
        dataset (Fusion360GraphDataset): Dataset used to infer node/edge input dimensions.
        hidden_dim, out_dim, num_layers, dropout, pooling: Forwarded to CADGraphEncoder.
        proj_dim (int): Output size of the contrastive projection head.

    Returns:
        tuple: (encoder, model) where model wraps the encoder for contrastive training.
    """
    node_in_dim, edge_in_dim = dataset.get_feature_dimensions()
    if node_in_dim == 0:
        raise ValueError("Fusion 360 dataset is empty. Did you run build_fusion360_graphs.py?")

    encoder = CADGraphEncoder(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
    )
    # The projection head consumes the encoder output, so its input size is out_dim
    model = CADContrastiveModel(encoder=encoder, hidden_dim=out_dim, proj_dim=proj_dim)
    return encoder, model


# ==========================================
# Quick Test Block (Executes only if run directly)
# ==========================================
if __name__ == "__main__":
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./data/processed_graphs/fusion360", help="Path to .pt files")
    args = parser.parse_args()

    print("Testing GINE-based CAD Encoder on Fusion 360 graphs...")

    dataset = Fusion360GraphDataset(root_dir=args.dir)
    encoder, model = build_fusion360_model(dataset)

    node_dim, edge_dim = dataset.get_feature_dimensions()
    print(f"Fusion 360 dataset: {len(dataset)} models | Node dim: {node_dim} | Edge dim: {edge_dim}")

    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    # Test Inference
    model.eval()
    embeddings = model(batch)
    print(f"Graph Embedding Output Shape: {embeddings.shape}")  # Should be [4, out_dim]

    # Test Contrastive Training Pass
    model.train()
    z1, z2 = model.forward_cl(batch, batch)
    print(f"Projection Head Output Shape: {z1.shape}")  # Should be [4, proj_dim]
    print("Fusion 360 network tests passed successfully!")
