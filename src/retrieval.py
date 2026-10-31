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
from src.features import get_transform
from src.run_config import resolve_arch, load_encoder_weights

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==========================================
# Database Builder
# ==========================================
def build_database(args, dataset_cls=CADGraphDataset):
    """
    Passes all CAD models through the trained encoder, extracts embeddings,
    and builds a FAISS Index.

    Architecture settings (hidden_dim/out_dim/num_layers/pooling/features) are read
    from <weights_dir>/config.json when it exists, so they can't be mismatched.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Building Database on: {device}")

    weights_dir = args.weights_dir or args.save_dir
    args, found = resolve_arch(args, weights_dir)
    if found:
        logging.info(f"Using architecture from {weights_dir}/config.json: "
                     f"out_dim={args.out_dim}, layers={args.num_layers}, pooling={args.pooling}, features={args.features}")
    else:
        logging.warning(f"No config.json in {weights_dir}; using CLI values "
                        f"(out_dim={args.out_dim}, layers={args.num_layers}, pooling={args.pooling}, features={args.features}).")

    # 1. Load Dataset
    try:
        dataset = dataset_cls(root_dir=args.data_dir, transform=get_transform(args.features))
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
        num_layers=args.num_layers,
        pooling=args.pooling
    ).to(device)

    loaded = load_encoder_weights(encoder, weights_dir, device)
    logging.info(f"Weights loaded from {loaded}")
        
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
    db_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(db_path / args.index_name))
    
    with open(db_path / args.meta_name, "wb") as f:
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
    index_file = db_path / args.index_name
    meta_file = db_path / args.meta_name
    
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
    
    rank = 0
    for dist, idx in zip(distances[0], indices[0]):
        if idx == query_idx or idx == -1:
            continue # Skip the self-match
        rank += 1
        if rank > args.top_k:
            break
        
        sim_score = dist * 100 # Inner product converted to percentage
        print(f"Rank {rank}: {cad_names[idx]:<25} | Similarity: {sim_score:.2f}%")
        
    print("="*50 + "\n")


def build_arg_parser(default_data_dir="./data/processed_graphs/mfcad", default_save_dir="./checkpoints",
                     index_name="faiss_index.bin", meta_name="metadata.pkl",
                     description="CAD Model Vector Retrieval"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to .pt files")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Directory where the FAISS index + metadata are written/read")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory with the trained encoder (+ config.json). Defaults to --save_dir")
    parser.add_argument("--index_name", type=str, default=index_name, help="FAISS index file name inside --save_dir")
    parser.add_argument("--meta_name", type=str, default=meta_name, help="Metadata file name inside --save_dir")
    # Architecture flags: only used for old checkpoints WITHOUT config.json
    parser.add_argument("--hidden_dim", type=int, default=128, help="GNN hidden dimension (legacy checkpoints only)")
    parser.add_argument("--out_dim", type=int, default=64, help="Final CAD embedding dimension (legacy checkpoints only)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "max_mean"], help="Readout used at training time (legacy checkpoints only)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GINE layers used at training time (legacy checkpoints only)")
    parser.add_argument("--features", type=str, default="raw", choices=["raw", "normalized"], help="Feature mode used at training time (legacy checkpoints only)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    
    # Mode selection
    parser.add_argument("--build", action="store_true", help="Build the FAISS database from the dataset")
    parser.add_argument("--query", type=str, default=None, help="Filename (without extension) to search for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar models to retrieve")
    return parser


def main(argv=None, parser=None, dataset_cls=CADGraphDataset):
    parser = parser or build_arg_parser()
    args = parser.parse_args(argv)

    if args.build:
        build_database(args, dataset_cls=dataset_cls)
    elif args.query:
        query_database(args)
    else:
        logging.warning("Please specify an action: --build OR --query <model_name>")


if __name__ == "__main__":
    main()
