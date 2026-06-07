"""
CAD Vector Retrieval System
===========================
@Author: Di Liu
@Date: 2026-05-24
@Description:
Builds and queries a FAISS vector database using the trained GNN Encoder.
Uses Cosine Similarity (Inner Product on L2-normalized embeddings) to 
find geometrically similar CAD models in milliseconds.
"""

import os
import argparse
import pathlib
import logging
import pickle
import numpy as np

import torch
from torch_geometric.loader import DataLoader
import faiss

# Import custom modules
from src.dataset import CADGraphDataset
from src.networks import CADGraphEncoder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==========================================
# Database Builder
# ==========================================
def build_database(args):
    """
    Passes all CAD models through the trained encoder, extracts embeddings,
    and builds a FAISS Index.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Building Database on: {device}")

    # 1. Load Dataset
    try:
        dataset = CADGraphDataset(root_dir=args.data_dir)
        node_dim, edge_dim = dataset.get_feature_dimensions()
        logging.info(f"Loaded {len(dataset)} graphs for the database.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # CRITICAL: drop_last MUST be False here. We need every single CAD model in the database!
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. Load Trained Encoder
    encoder = CADGraphEncoder(
        node_in_dim=node_dim, 
        edge_in_dim=edge_dim, 
        hidden_dim=args.hidden_dim, 
        out_dim=args.out_dim,
        num_layers=3
    ).to(device)

    weights_path = pathlib.Path(args.save_dir) / "final_encoder_weights.pth"
    if not weights_path.exists():
        # Fallback to the best contrastive checkpoint if final is not found
        weights_path = pathlib.Path(args.save_dir) / "best_contrastive_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No trained weights found in {args.save_dir}. Run training first!")
        
        checkpoint = torch.load(weights_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        encoder.load_state_dict(torch.load(weights_path, map_location=device))
        
    encoder.eval()
    logging.info("Trained Encoder loaded successfully.")

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
            all_names.extend(batch.name) # Extract filenames attached in dataset.py

    # Stack into a single numpy array of float32 (FAISS requirement)
    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    logging.info(f"Generated Embedding Matrix shape: {embedding_matrix.shape}")

    # 4. Build FAISS Index
    # IndexFlatIP computes the Inner Product. 
    # Since our embeddings are L2 normalized, Inner Product == Cosine Similarity.
    index = faiss.IndexFlatIP(args.out_dim)
    index.add(embedding_matrix)

    # 5. Save Database and Metadata
    db_path = pathlib.Path(args.save_dir)
    faiss.write_index(index, str(db_path / "faiss_index.bin"))
    
    with open(db_path / "metadata.pkl", "wb") as f:
        pickle.dump(all_names, f)
        
    logging.info("[Pass] FAISS database and metadata successfully saved!")

# ==========================================
# Database Query Engine
# ==========================================
def query_database(args):
    """
    Simulates a retrieval request. Given a CAD model name (or index), 
    finds the Top-K most similar models.
    """
    db_path = pathlib.Path(args.save_dir)
    index_file = db_path / "faiss_index.bin"
    meta_file = db_path / "metadata.pkl"
    
    if not index_file.exists() or not meta_file.exists():
        logging.error("Database files missing. Please run with --build first.")
        return

    # 1. Load FAISS and Metadata
    index = faiss.read_index(str(index_file))
    with open(meta_file, "rb") as f:
        cad_names = pickle.load(f)
        
    logging.info(f"Loaded FAISS index with {index.ntotal} models.")

    # 2. Find the Query Vector
    # In a real app (Streamlit), you would process the uploaded STEP file live.
    # For testing, we just pick an existing model from the dataset.
    try:
        query_idx = cad_names.index(args.query)
    except ValueError:
        logging.error(f"CAD model '{args.query}' not found in the database.")
        return

    # Reconstruct the vector directly from FAISS (shape: [1, out_dim])
    query_vector = np.expand_dims(index.reconstruct(query_idx), axis=0)

    # 3. Perform Search
    k = args.top_k + 1 # +1 because the top result will be the query itself
    distances, indices = index.search(query_vector, k)

    # 4. Display Results
    print("\n" + "="*50)
    print(f"Retrieval Results for: {args.query}")
    print("="*50)
    
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == query_idx:
            continue # Skip the self-match
        
        sim_score = dist * 100 # Inner product converted to percentage
        print(f"Rank {rank}: {cad_names[idx]:<25} | Similarity: {sim_score:.2f}%")
        
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAD Model Vector Retrieval")
    parser.add_argument("--data_dir", type=str, default="./data/processed_graphs/mfcad", help="Path to .pt files")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory containing weights & db")
    parser.add_argument("--hidden_dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--out_dim", type=int, default=128, help="Final CAD embedding dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    
    # Mode selection
    parser.add_argument("--build", action="store_true", help="Build the FAISS database from the dataset")
    parser.add_argument("--query", type=str, default=None, help="Filename (without extension) to search for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar models to retrieve")
    
    args = parser.parse_args()

    if args.build:
        build_database(args)
    elif args.query:
        query_database(args)
    else:
        logging.warning("Please specify an action: --build OR --query <model_name>")