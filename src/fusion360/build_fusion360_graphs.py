"""
Fusion 360 Graph Builder
========================
@Author: Ayse Seray Seker
@Date: 2026-10-22
@Description:
Converts the raw Fusion 360 STEP files (data/raw_step/fusion360) into PyG Attributed
Adjacency Graphs (.pt) and stores them in data/processed_graphs/fusion360.
Reuses the exact same B-Rep feature extraction (src.data_converter.worker) as the MFCAD
pipeline, so both datasets share identical node/edge feature definitions.
"""

import argparse
import pathlib
import logging
import multiprocessing
from collections import Counter
from functools import partial

from tqdm import tqdm

# Import custom modules
from src.data_converter import worker
from src.dataset import CADGraphDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

STEP_PATTERNS = ("*.step", "*.stp")
SKIPPED_MSG = "Already exists"  # Message returned by data_converter.worker for resumed jobs

# ==========================================
# Fusion 360 Graph Builder
# ==========================================
def collect_step_files(input_path, limit=None):
    """Returns a sorted list of STEP files (.step / .stp) found directly inside input_path."""
    step_files = []
    for pattern in STEP_PATTERNS:
        step_files.extend(input_path.glob(pattern))

    step_files = sorted(step_files)
    return step_files[:limit] if limit else step_files


def build_graphs(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.is_dir():
        logging.error(f"Input directory not found: {input_path}")
        return

    step_files = collect_step_files(input_path, args.limit)
    if not step_files:
        logging.error(f"No STEP files found in {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Found {len(step_files)} Fusion 360 models. Converting with {args.cores} cores...")

    # Same worker as the MFCAD pipeline; already-converted files are skipped (resumable)
    worker_func = partial(worker, output_dir=output_path)
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, step_files, chunksize=8),
                            total=len(step_files),
                            desc="Converting Fusion 360 to Graphs"))

    # Summary Statistics
    skipped = [name for success, name, msg in results if success and msg == SKIPPED_MSG]
    failed = [(name, msg) for success, name, msg in results if not success]
    converted = len(results) - len(skipped) - len(failed)

    print("\n" + "="*50)
    print("Fusion 360 Graph Conversion Complete")
    print(f"Converted: {converted} | Already existed: {len(skipped)} | Failed: {len(failed)}")
    print(f"Graphs saved to: {output_path}")

    if failed:
        # Keep the console clean: write every failure to a log file next to the graphs
        log_path = output_path / "conversion_failures.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for name, msg in sorted(failed):
                f.write(f"{name}\t{msg}\n")
        print(f"Top failure reasons: {dict(Counter(msg for _, msg in failed).most_common(3))}")
        print(f"Full list written to: {log_path}")

    # Sanity check: feature dimensions must match the ones the MFCAD encoder was trained with (4 / 3)
    if converted + len(skipped) > 0:
        node_dim, edge_dim = CADGraphDataset(root_dir=output_path).get_feature_dimensions()
        print(f"Feature dimensions -> Node: {node_dim} | Edge: {edge_dim}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion 360 STEP-to-Graph Builder")
    parser.add_argument("--input", type=str, default="./data/raw_step/fusion360", help="Input folder with Fusion 360 STEP files")
    parser.add_argument("--output", type=str, default="./data/processed_graphs/fusion360", help="Output folder for .pt files")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() // 2),
                        help="Number of CPU cores to use (default: half of available)")
    parser.add_argument("--limit", type=int, default=None, help="Only convert the first N files (quick smoke test)")
    args = parser.parse_args()

    build_graphs(args)
