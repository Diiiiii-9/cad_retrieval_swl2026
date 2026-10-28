"""
CAD Vector Retrieval System for Fusion 360
==========================================
@Author: Ayse Seray Seker
@Date: 2026-10-22
@Description:
Fusion 360 counterpart of src/retrieval.py. Loads graphs through dataset_fusion360.py and
builds the encoder through networks_fusion360.py, then builds/queries a FAISS database.
The database is written to its own folder (checkpoints/fusion360) under its own file names
(faiss_index_fusion360.bin / metadata_fusion360.pkl), so it can never overwrite or be mixed
up with the MFCAD database (checkpoints/faiss_index.bin / metadata.pkl).
"""

import argparse
import pathlib
import logging
import pickle
import numpy as np

import torch
from torch_geometric.loader import DataLoader
import faiss

# Import custom modules
from src.fusion360.dataset_fusion360 import Fusion360GraphDataset
from src.fusion360.networks_fusion360 import build_fusion360_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

INDEX_FILENAME = "faiss_index_fusion360.bin"
METADATA_FILENAME = "metadata_fusion360.pkl"

# ==========================================
# Database Builder
# ==========================================
def build_database(args):
    """
    Passes all Fusion 360 CAD models through the trained encoder, extracts embeddings,
    and builds a FAISS Index.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Building Fusion 360 Database on: {device}")

    # 1. Load Dataset
    try:
        dataset = Fusion360GraphDataset(root_dir=args.data_dir)
        node_dim, edge_dim = dataset.get_feature_dimensions()
        logging.info(f"Loaded {len(dataset)} graphs for the database.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # CRITICAL: drop_last MUST be False here. We need every single CAD model in the database!
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. Load Trained Encoder
    encoder, _ = build_fusion360_model(
        dataset,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        pooling=args.pooling
    )
    encoder = encoder.to(device)

    weights_path = pathlib.Path(args.weights_dir) / "final_encoder_weights.pth"
    if not weights_path.exists():
        # Fallback to the best contrastive checkpoint if final is not found
        weights_path = pathlib.Path(args.weights_dir) / "best_contrastive_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No trained weights found in {args.weights_dir}. Run training first!")

        checkpoint = torch.load(weights_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        encoder.load_state_dict(torch.load(weights_path, map_location=device))

    encoder.eval()
    logging.info(f"Trained Encoder loaded successfully from {weights_path}.")

    # 3. Extract Embeddings
    all_embeddings = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Handle isolated graphs safely
            edge_index = batch.edge_index if batch.edge_index is not None else torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = batch.edge_attr if batch.edge_attr is not None else torch.empty((0, edge_dim), dtype=torch.float, device=device)

            # Forward pass: shape [Batch_size, Out_dim]
            embs = encoder(batch.x, edge_index, edge_attr, batch.batch)

            all_embeddings.append(embs.cpu().numpy())
            all_names.extend(batch.name)  # Extract filenames attached in dataset.py

    # Stack into a single numpy array of float32 (FAISS requirement)
    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    logging.info(f"Generated Embedding Matrix shape: {embedding_matrix.shape}")

    # 4. Build FAISS Index
    # IndexFlatIP computes the Inner Product.
    # Since our embeddings are L2 normalized, Inner Product == Cosine Similarity.
    index = faiss.IndexFlatIP(args.out_dim)
    index.add(embedding_matrix)

    # 5. Save Database and Metadata (separate folder + file names from the MFCAD database)
    db_path = pathlib.Path(args.save_dir)
    db_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(db_path / INDEX_FILENAME))

    with open(db_path / METADATA_FILENAME, "wb") as f:
        pickle.dump(all_names, f)

    logging.info(f"[Pass] Fusion 360 FAISS database and metadata saved to {db_path}!")

# ==========================================
# Database Query Engine
# ==========================================
def query_database(args):
    """
    Simulates a retrieval request. Given a Fusion 360 CAD model name,
    finds the Top-K most similar models.
    """
    db_path = pathlib.Path(args.save_dir)
    index_file = db_path / INDEX_FILENAME
    meta_file = db_path / METADATA_FILENAME

    if not index_file.exists() or not meta_file.exists():
        logging.error("Fusion 360 database files missing. Please run with --build first.")
        return

    # 1. Load FAISS and Metadata
    index = faiss.read_index(str(index_file))
    with open(meta_file, "rb") as f:
        cad_names = pickle.load(f)

    logging.info(f"Loaded Fusion 360 FAISS index with {index.ntotal} models.")

    # 2. Find the Query Vector
    try:
        query_idx = cad_names.index(args.query)
    except ValueError:
        logging.error(f"CAD model {args.query} not found in the database.")
        return

    # Reconstruct the vector directly from FAISS (shape: [1, out_dim])
    query_vector = np.expand_dims(index.reconstruct(query_idx), axis=0)

    # 3. Perform Search
    k = args.top_k + 1  # +1 because the top result will be the query itself
    distances, indices = index.search(query_vector, k)

    # 4. Display Results
    print("\n" + "="*50)
    print(f"Fusion 360 Retrieval Results for: {args.query}")
    print("="*50)

    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == query_idx:
            continue  # Skip the self-match

        sim_score = dist * 100  # Inner product converted to percentage
        print(f"Rank {rank}: {cad_names[idx]:<40} | Similarity: {sim_score:.2f}%")

    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion 360 CAD Model Vector Retrieval")
    parser.add_argument("--data_dir", type=str, default="./data/processed_graphs/fusion360", help="Path to .pt files")
    parser.add_argument("--weights_dir", type=str, default="./checkpoints", help="Directory containing the trained encoder weights (read only, MFCAD-trained encoder by default)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/fusion360", help="Directory where the Fusion 360 database is saved/loaded")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GNN hidden dimension")
    parser.add_argument("--out_dim", type=int, default=64, help="Final CAD embedding dimension")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "max_mean"], help="Graph readout strategy used when this checkpoint was trained")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GINE layers used when this checkpoint was trained")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")

    # Mode selection
    parser.add_argument("--build", action="store_true", help="Build the FAISS database from the Fusion 360 dataset")
    parser.add_argument("--query", type=str, default=None, help="Filename (without extension) to search for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar models to retrieve")

    args = parser.parse_args()

    if args.build:
        build_database(args)
    elif args.query:
        query_database(args)
    else:
        logging.warning("Please specify an action: --build OR --query <model_name>")
