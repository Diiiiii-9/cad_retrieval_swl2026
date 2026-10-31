"""
CAD Vector Retrieval System for Fusion 360
==========================================
@Author: Ayse Seray Seker
@Date: 2026-10-22
@Description:
Fusion 360 entry point for src/retrieval.py. Same build/query logic, different defaults:
  * graphs from data/processed_graphs/fusion360
  * weights AND database in checkpoints/fusion360
  * own file names (faiss_index_fusion360.bin / metadata_fusion360.pkl), so it can never
    overwrite or be mixed up with the MFCAD database.

To index Fusion 360 with the MFCAD-trained encoder instead (cross-dataset transfer), pass
    --weights_dir ./checkpoints --save_dir ./checkpoints/fusion360_from_mfcad

Usage:
    python -m src.fusion360.retrieval_fusion360 --build
    python -m src.fusion360.retrieval_fusion360 --query <model_name>
"""

from src.retrieval import build_arg_parser, main as _main
from src.fusion360.dataset_fusion360 import Fusion360GraphDataset

INDEX_FILENAME = "faiss_index_fusion360.bin"
METADATA_FILENAME = "metadata_fusion360.pkl"


def main(argv=None):
    parser = build_arg_parser(
        default_data_dir="./data/processed_graphs/fusion360",
        default_save_dir="./checkpoints/fusion360",
        index_name=INDEX_FILENAME,
        meta_name=METADATA_FILENAME,
        description="Fusion 360 CAD Model Vector Retrieval",
    )
    return _main(argv, parser=parser, dataset_cls=Fusion360GraphDataset)


if __name__ == "__main__":
    main()
