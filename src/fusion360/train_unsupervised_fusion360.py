"""
Unsupervised Training Loop for Fusion 360
=========================================
@Author: Ayse Seray Seker
@Date: 2026-10-22
@Description:
Fusion 360 entry point for training. It reuses the exact training loop in
src/train_unsupervised.py (same augmentations, InfoNCE loss, --features/--seed
flags, config.json + train_history.json output) and only changes the defaults:
data is read from data/processed_graphs/fusion360 and weights are saved to
checkpoints/fusion360, so MFCAD checkpoints are never overwritten.

Usage:
    python -m src.fusion360.train_unsupervised_fusion360 ...   (same flags as src.train_unsupervised)
"""

from src.train_unsupervised import build_arg_parser, train as _train
from src.fusion360.dataset_fusion360 import Fusion360GraphDataset


def train(argv=None):
    parser = build_arg_parser(
        default_data_dir="./data/processed_graphs/fusion360",
        default_save_dir="./checkpoints/fusion360",
        description="Train Fusion 360 CAD Contrastive Model",
    )
    return _train(argv, dataset_cls=Fusion360GraphDataset, parser=parser)


if __name__ == "__main__":
    train()
