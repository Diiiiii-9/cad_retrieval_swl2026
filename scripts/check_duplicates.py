"""
Duplicate / Collapse Check
==========================
In the committed Fusion 360 index, ~85% of parts have a near-identical twin
(cosine > 0.9999) and there are only ~8.5k distinct embeddings for 15.4k parts.
(MFCAD: 0.1%.) There are two possible explanations, and they matter for the report:

  (a) real duplicates -- the same body (bolt, washer...) appears in many assemblies.
      Fine, but retrieval scores on the full gallery are then inflated by exact copies.
  (b) embedding collapse -- geometrically DIFFERENT parts get the same vector
      (e.g. huge raw areas saturating the network). That is a model bug.

This script looks at every near-identical pair and checks whether the two input
graphs are actually identical (same face/edge counts and same features up to rounding).

Usage:
    python scripts/check_duplicates.py --run_dir checkpoints/fusion360 \
        --graphs_dir data/processed_graphs/fusion360
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_retrieval import load_embeddings, l2_normalize  # noqa: E402


def graph_signature(data, decimals=3):
    """Order-independent fingerprint of a raw graph."""
    x = np.round(data.x.numpy().astype("float64"), decimals)
    ea = np.round(data.edge_attr.numpy().astype("float64"), decimals) if data.edge_attr is not None else np.zeros((0, 3))
    return (len(x), len(ea), tuple(map(tuple, x[np.lexsort(x.T[::-1])])), tuple(map(tuple, ea[np.lexsort(ea.T[::-1])])) if len(ea) else ())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--graphs_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.9999)
    parser.add_argument("--max_pairs", type=int, default=2000, help="Pairs to inspect (random sample)")
    args = parser.parse_args()

    names, vectors = load_embeddings(args.run_dir)
    X = l2_normalize(vectors)
    import faiss
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    D, I = index.search(X, 2)

    # With exact ties the twin can be ranked above the query itself, so take whichever
    # of the top-2 results is not the query.
    has_twin, pairs = 0, set()
    for i in range(len(X)):
        col = 1 if I[i, 0] == i else 0
        j, sim = int(I[i, col]), D[i, col]
        if j != i and sim > args.threshold:
            has_twin += 1
            pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)
    print(f"{has_twin}/{len(X)} parts ({100 * has_twin / len(X):.1f}%) have a twin with cosine > {args.threshold}")
    if not pairs:
        return

    rng = np.random.default_rng(0)
    sample = [pairs[k] for k in rng.choice(len(pairs), min(args.max_pairs, len(pairs)), replace=False)]

    same, different, missing, examples = 0, 0, 0, []
    for i, j in sample:
        pi, pj = Path(args.graphs_dir) / f"{names[i]}.pt", Path(args.graphs_dir) / f"{names[j]}.pt"
        if not (pi.exists() and pj.exists()):
            missing += 1
            continue
        gi, gj = torch.load(pi, weights_only=False), torch.load(pj, weights_only=False)
        if graph_signature(gi) == graph_signature(gj):
            same += 1
        else:
            different += 1
            if len(examples) < 5:
                examples.append((names[i], names[j], gi.x.shape[0], gj.x.shape[0],
                                 float(gi.x[:, 0].sum()), float(gj.x[:, 0].sum())))

    checked = same + different
    print(f"Checked {checked} pairs ({missing} skipped, graph missing):")
    print(f"  identical input graphs (real duplicates): {same} ({100 * same / max(checked, 1):.1f}%)")
    print(f"  DIFFERENT input graphs (embedding collapse): {different} ({100 * different / max(checked, 1):.1f}%)")
    for a, b, na, nb, aa, ab in examples:
        print(f"    e.g. {a} ({na} faces, area {aa:.3g})  vs  {b} ({nb} faces, area {ab:.3g})")
    if different > same:
        print("\n-> Mostly collapse: compare against a run trained with --features normalized.")
    else:
        print("\n-> Mostly real duplicates: mention it in the report; consider deduplicating the gallery.")


if __name__ == "__main__":
    main()
