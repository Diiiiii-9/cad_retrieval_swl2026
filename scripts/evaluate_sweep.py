"""
Ablation Sweep Evaluation: Embedding Dimension + Temperature
==============================================================
Evaluates all 6 checkpoints from the dim/temp sweep using the EXACT same
manufacturing-aware Recall@K / MRR logic validated in
notebooks/ablation_study_mfcad.ipynb, looped across configs.

Each config's faiss_index.bin + metadata.pkl (built by
scripts/build_sweep_databases.ps1) lives under checkpoints/sweep/<name>/.

Usage (works from any directory; paths are resolved relative to the repo
root via this file's own location):
    python scripts/evaluate_sweep.py
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

# ==========================================
# Config
# ==========================================
# This script lives in scripts/, one level below the repo root, so anchor
# PROJECT_ROOT off this file's own location rather than the current working
# directory -- that way it works correctly regardless of where it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_ROOT / "data" / "raw_step" / "mfcad_label"
SWEEP_DIR = PROJECT_ROOT / "checkpoints" / "sweep"

NUM_CLASSES = 16
MATCH_THRESHOLD = 0.95
K_MAX = 10

# Maps config folder name -> (out_dim, temp, pooling, aug_mode, num_layers) for the results table.
# Must match exactly what was passed to train_unsupervised.py for each run.
CONFIGS = [
    {"name": "baseline_dim64_temp01", "out_dim": 64,  "temp": 0.1,  "pooling": "mean", "aug_mode": "both", "num_layers": 3},
    {"name": "dim32",                 "out_dim": 32,  "temp": 0.1,  "pooling": "mean", "aug_mode": "both", "num_layers": 3},
    {"name": "dim128",                "out_dim": 128, "temp": 0.1,  "pooling": "mean", "aug_mode": "both", "num_layers": 3},
    {"name": "dim256",                "out_dim": 256, "temp": 0.1,  "pooling": "mean", "aug_mode": "both", "num_layers": 3},
    {"name": "temp005",               "out_dim": 64,  "temp": 0.05, "pooling": "mean", "aug_mode": "both", "num_layers": 3},
    {"name": "temp05",                "out_dim": 64,  "temp": 0.5,  "pooling": "mean", "aug_mode": "both", "num_layers": 3},
    {"name": "pooling_max",           "out_dim": 64,  "temp": 0.1,  "pooling": "max",      "aug_mode": "both", "num_layers": 3},
    {"name": "pooling_max_mean",      "out_dim": 64,  "temp": 0.1,  "pooling": "max_mean", "aug_mode": "both", "num_layers": 3},
    {"name": "aug_edge_drop",         "out_dim": 64,  "temp": 0.1,  "pooling": "mean", "aug_mode": "edge_drop",     "num_layers": 3},
    {"name": "aug_feature_noise",     "out_dim": 64,  "temp": 0.1,  "pooling": "mean", "aug_mode": "feature_noise", "num_layers": 3},
    {"name": "layers2",               "out_dim": 64,  "temp": 0.1,  "pooling": "mean", "aug_mode": "both", "num_layers": 2},
    {"name": "layers5",               "out_dim": 64,  "temp": 0.1,  "pooling": "mean", "aug_mode": "both", "num_layers": 5},
]


def cosine_sim(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def build_label_signatures(metadata):
    """Builds the 16-D feature histogram + validity mask for each model name.
    This only depends on labels, not on the model -- so it's computed ONCE
    and reused across all 6 configs, since metadata (the list of CAD model
    names) is identical across configs (same dataset, same order is NOT
    guaranteed, so we still build it per-config from that config's metadata
    to be safe, but the underlying label files are shared)."""
    signatures = []
    valid_mask = []

    for meta in metadata:
        base_name = meta.replace(".step", "")
        label_path = LABEL_DIR / f"{base_name}.face_truth"

        if label_path.exists():
            with open(label_path, "rb") as file:
                face_labels = pickle.load(file)

            labels = face_labels.values() if isinstance(face_labels, dict) else face_labels

            hist = np.zeros(NUM_CLASSES)
            for lbl in labels:
                if 0 <= lbl < NUM_CLASSES:
                    hist[lbl] += 1

            signatures.append(hist)
            valid_mask.append(True)
        else:
            signatures.append(np.zeros(NUM_CLASSES))
            valid_mask.append(False)

    return np.array(signatures), valid_mask


def evaluate_config(cfg):
    save_dir = SWEEP_DIR / cfg["name"]
    index_path = save_dir / "faiss_index.bin"
    metadata_path = save_dir / "metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        print(f"[SKIP] {cfg['name']}: missing faiss_index.bin or metadata.pkl")
        return None

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    index = faiss.read_index(str(index_path))
    num_vectors = index.ntotal
    vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])

    signatures, valid_mask = build_label_signatures(metadata)

    distances, indices = index.search(vectors, K_MAX + 1)

    recalls = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    valid_queries = 0

    for i in tqdm(range(num_vectors), desc=f"Evaluating {cfg['name']}", leave=False):
        if not valid_mask[i]:
            continue

        query_sig = signatures[i]
        if np.linalg.norm(query_sig) == 0:
            continue

        valid_queries += 1
        retrieved_idx = indices[i][1:]  # exclude the query itself

        is_relevant = []
        for idx in retrieved_idx:
            if not valid_mask[idx]:
                is_relevant.append(False)
            else:
                sim = cosine_sim(query_sig, signatures[idx])
                is_relevant.append(sim >= MATCH_THRESHOLD)

        for k in recalls.keys():
            if any(is_relevant[:k]):
                recalls[k] += 1

        for rank, relevant in enumerate(is_relevant):
            if relevant:
                mrr += 1.0 / (rank + 1)
                break

    if valid_queries == 0:
        print(f"[WARNING] {cfg['name']}: valid_queries == 0, cannot compute metrics")
        return None

    for k in recalls.keys():
        recalls[k] /= valid_queries
    mrr /= valid_queries

    return {
        "name": cfg["name"],
        "out_dim": cfg["out_dim"],
        "temp": cfg["temp"],
        "pooling": cfg["pooling"],
        "aug_mode": cfg["aug_mode"],
        "num_layers": cfg["num_layers"],
        "valid_queries": valid_queries,
        "recall@1": recalls[1] * 100,
        "recall@5": recalls[5] * 100,
        "recall@10": recalls[10] * 100,
        "mrr": mrr,
    }


def main():
    print(f"Label directory exists: {LABEL_DIR.exists()}")
    print(f"Sweep directory: {SWEEP_DIR.resolve()}")
    print()

    results = []
    for cfg in CONFIGS:
        result = evaluate_config(cfg)
        if result is not None:
            results.append(result)
            print(
                f"[OK] {result['name']:24s} "
                f"R@1={result['recall@1']:6.2f}%  "
                f"R@5={result['recall@5']:6.2f}%  "
                f"R@10={result['recall@10']:6.2f}%  "
                f"MRR={result['mrr']:.4f}  "
                f"(valid_queries={result['valid_queries']})"
            )

    if not results:
        print("\nNo results computed -- check the [SKIP]/[WARNING] messages above.")
        return

    # ==========================================
    # Final comparison table
    # ==========================================
    print("\n" + "=" * 118)
    print("ABLATION RESULTS: ALL CONFIGS (DIMENSION, TEMPERATURE, POOLING, AUGMENTATION, LAYERS)")
    print("=" * 118)
    header = f"{'Config':<22}{'out_dim':>9}{'temp':>7}{'pooling':>10}{'aug_mode':>15}{'L':>3}{'Recall@1':>11}{'Recall@5':>11}{'Recall@10':>12}{'MRR':>9}"
    print(header)
    print("-" * 118)
    for r in results:
        print(
            f"{r['name']:<22}{r['out_dim']:>9}{r['temp']:>7}{r['pooling']:>10}{r['aug_mode']:>15}{r['num_layers']:>3}"
            f"{r['recall@1']:>10.2f}%{r['recall@5']:>10.2f}%{r['recall@10']:>11.2f}%{r['mrr']:>9.4f}"
        )
    print("=" * 118)

    # Save results to disk too, for later use in slides/reports
    out_path = PROJECT_ROOT / "checkpoints" / "sweep" / "ablation_results_all.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults also saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
