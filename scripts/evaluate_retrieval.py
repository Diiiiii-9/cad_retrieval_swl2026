"""
Retrieval Evaluation with Baselines
===================================
@Description:
Evaluates one or more trained runs (each a folder with a FAISS index + metadata) and
compares them against two baselines that every results table should contain:

  * random       -- random unit vectors (mean of 5 draws): the "chance" level
  * handcrafted  -- a non-learned descriptor computed straight from the graphs
                    (surface/curve-type histograms, area/length-weighted versions,
                    graph size, closed-face/edge ratios). If the GNN does not beat
                    this, the learned embedding adds nothing over simple statistics.
                    Only computed when --graphs_dir is given.

Ground truth ("is retrieved part j relevant for query i?"):
  MFCAD
    class_L1      -- same top-level class in the file name   (e.g. '3-...')
    class_L2      -- same first two file-name tokens          (e.g. '3-7-...')
    feature_hist  -- per-face machining-feature histograms have cosine >= 0.95
                     (the metric used in ablation_study_final.ipynb; needs --label_dir)
  Fusion 360
    category      -- same manual category in data/fusion360_labels.json. Only labeled
                     parts are used, as both queries and gallery.

Metrics (query itself excluded, top 10 neighbours):
  P@k     -- precision: fraction of the top-k results that are relevant
  Hit@k   -- fraction of queries with >= 1 relevant result in the top-k
             (this is what the earlier notebooks called "Recall@k")
  MRR     -- mean reciprocal rank of the first relevant result (0 if none in top 10)
  mAP@10  -- mean average precision over the top 10

Runs whose folder names differ only by a `_s<N>` suffix (e.g. final_norm_s0, _s1, _s2)
are treated as seeds of one config and reported as mean +- std.

Usage (from the repo root):
  python scripts/evaluate_retrieval.py --dataset mfcad \
      --runs checkpoints/sweep/* \
      --graphs_dir data/processed_graphs/mfcad --label_dir data/raw_step/mfcad_label \
      --out results/mfcad

  python scripts/evaluate_retrieval.py --dataset fusion360 \
      --runs checkpoints/fusion360 \
      --labels data/fusion360_labels.json --graphs_dir data/processed_graphs/fusion360 \
      --out results/fusion360
"""

import argparse
import glob
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # evaluation works without faiss (slower brute-force search)
    faiss = None

K_MAX = 10
KS = (1, 5, 10)
NUM_FEATURE_CLASSES = 16
HIST_MATCH_THRESHOLD = 0.95
NUM_SURF_TYPES, NUM_CURVE_TYPES = 10, 8
SEED_SUFFIX = re.compile(r"_s\d+$")


# ==========================================
# Loading
# ==========================================
def find_index_files(run_dir):
    run_dir = Path(run_dir)
    idx = sorted(run_dir.glob("faiss_index*.bin"))
    meta = sorted(run_dir.glob("metadata*.pkl"))
    if not idx or not meta:
        return None, None
    return idx[0], meta[0]


def load_embeddings(run_dir):
    index_path, meta_path = find_index_files(run_dir)
    if index_path is None:
        return None, None
    with open(meta_path, "rb") as f:
        names = [str(n) for n in pickle.load(f)]
    if faiss is None:
        raise ImportError("faiss is required to read FAISS index files (pip install faiss-cpu)")
    index = faiss.read_index(str(index_path))
    vectors = index.reconstruct_n(0, index.ntotal).astype("float32")
    return names, vectors


def l2_normalize(x):
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ==========================================
# Ground truth
# ==========================================
def mfcad_class(name, level):
    return "-".join(Path(name).stem.split("-")[:level])


def load_feature_histograms(names, label_dir):
    """16-D per-face machining-feature histogram per model; None where the label file is missing."""
    hists = []
    for name in names:
        path = Path(label_dir) / f"{Path(name).stem}.face_truth"
        if not path.exists():
            hists.append(None)
            continue
        with open(path, "rb") as f:
            labels = pickle.load(f)
        labels = labels.values() if isinstance(labels, dict) else labels
        h = np.zeros(NUM_FEATURE_CLASSES)
        for lbl in labels:
            if 0 <= int(lbl) < NUM_FEATURE_CLASSES:
                h[int(lbl)] += 1
        hists.append(h if h.sum() > 0 else None)
    return hists


def load_fusion_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Labels were stored with the STL file name as key; the index uses the bare model name.
    return {Path(k).stem: v for k, v in raw.items()}


def build_ground_truths(dataset, names, label_dir=None, labels_path=None):
    """Returns (gallery_names, {scheme: relevance_fn}) where relevance_fn(i, js) -> bool array."""
    if dataset == "mfcad":
        gallery = list(names)
        schemes = {}
        for level, key in ((1, "class_L1"), (2, "class_L2")):
            cls = np.array([mfcad_class(n, level) for n in gallery])
            schemes[key] = (lambda c: (lambda i, js: c[js] == c[i]))(cls)
        if label_dir and Path(label_dir).exists():
            hists = load_feature_histograms(gallery, label_dir)
            valid = np.array([h is not None for h in hists])
            if valid.any():
                H = np.stack([h if h is not None else np.zeros(NUM_FEATURE_CLASSES) for h in hists])
                H = l2_normalize(H)

                def hist_rel(i, js, H=H, valid=valid):
                    if not valid[i]:
                        return None  # query skipped
                    return valid[js] & ((H[js] @ H[i]) >= HIST_MATCH_THRESHOLD)
                schemes["feature_hist"] = hist_rel
        return gallery, schemes

    if dataset == "fusion360":
        labels = load_fusion_labels(labels_path)
        gallery = [n for n in names if Path(n).stem in labels]
        cats = np.array([labels[Path(n).stem] for n in gallery])
        return gallery, {"category": lambda i, js, c=cats: c[js] == c[i]}

    raise ValueError(dataset)


# ==========================================
# Retrieval + metrics
# ==========================================
def knn(vectors, k=K_MAX):
    """Top-k neighbours by cosine similarity, excluding the query itself."""
    x = l2_normalize(vectors)
    k_search = min(k + 1, len(x))
    if faiss is not None:
        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        _, idx = index.search(x, k_search)
    else:
        sims = x @ x.T
        idx = np.argsort(-sims, axis=1)[:, :k_search]
    out = np.empty((len(x), k_search - 1), dtype=np.int64)
    for i, row in enumerate(idx):
        row = row[row != i][: k_search - 1]
        if len(row) < k_search - 1:  # duplicate vectors can push the query out of position 0
            row = np.concatenate([row, idx[i][: k_search - 1 - len(row)]])
        out[i] = row
    return out


def score(neighbours, relevance_fn):
    p = {k: [] for k in KS}
    hit = {k: [] for k in KS}
    rr, ap = [], []
    for i, js in enumerate(neighbours):
        rel = relevance_fn(i, js)
        if rel is None:
            continue
        rel = np.asarray(rel, dtype=bool)
        for k in KS:
            p[k].append(rel[:k].mean())
            hit[k].append(rel[:k].any())
        first = np.flatnonzero(rel)
        rr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
        if rel.any():
            prec_at_hits = np.cumsum(rel)[rel] / (np.flatnonzero(rel) + 1)
            ap.append(prec_at_hits.mean())
        else:
            ap.append(0.0)
    n = len(rr)
    if n == 0:
        return None
    res = {f"P@{k}": 100 * float(np.mean(p[k])) for k in KS}
    res.update({f"Hit@{k}": 100 * float(np.mean(hit[k])) for k in KS})
    res["MRR"] = float(np.mean(rr))
    res["mAP@10"] = float(np.mean(ap))
    res["queries"] = n
    return res


# ==========================================
# Baselines
# ==========================================
def handcrafted_descriptor(data):
    """Non-learned descriptor from a raw graph (x=[area, surf_idx, closed_u, closed_v],
    edge_attr=[length, curve_idx, closed])."""
    x = data.x.numpy().astype("float64")
    area, surf = np.clip(x[:, 0], 0, None), np.clip(np.round(x[:, 1]).astype(int), 0, NUM_SURF_TYPES - 1)
    n_faces = len(x)

    surf_count = np.bincount(surf, minlength=NUM_SURF_TYPES) / max(n_faces, 1)
    surf_area = np.bincount(surf, weights=area, minlength=NUM_SURF_TYPES) / max(area.sum(), 1e-12)
    closed_face = ((x[:, 2] + x[:, 3]) > 0).mean() if n_faces else 0.0

    ea = data.edge_attr.numpy().astype("float64") if data.edge_attr is not None else np.zeros((0, 3))
    ea = ea[::2]  # every undirected edge is stored twice
    n_edges = len(ea)
    if n_edges:
        length = np.clip(ea[:, 0], 0, None)
        curve = np.clip(np.round(ea[:, 1]).astype(int), 0, NUM_CURVE_TYPES - 1)
        curve_count = np.bincount(curve, minlength=NUM_CURVE_TYPES) / n_edges
        curve_len = np.bincount(curve, weights=length, minlength=NUM_CURVE_TYPES) / max(length.sum(), 1e-12)
        closed_edge = ea[:, 2].mean()
    else:
        curve_count = curve_len = np.zeros(NUM_CURVE_TYPES)
        closed_edge = 0.0

    size = [np.log1p(n_faces), np.log1p(n_edges), 2.0 * n_edges / max(n_faces, 1), closed_face, closed_edge]
    return np.concatenate([surf_count, surf_area, curve_count, curve_len, size])


def handcrafted_embeddings(names, graphs_dir):
    import torch  # only needed for this baseline
    vecs, missing = [], 0
    for name in names:
        path = Path(graphs_dir) / f"{Path(name).stem}.pt"
        if not path.exists():
            missing += 1
            vecs.append(None)
            continue
        data = torch.load(path, weights_only=False)
        vecs.append(handcrafted_descriptor(data))
    if missing == len(names):
        return None, missing
    dim = next(v for v in vecs if v is not None).shape[0]
    X = np.stack([v if v is not None else np.zeros(dim) for v in vecs])
    # z-score each dimension so no single statistic dominates the cosine similarity
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype("float32"), missing


def evaluate_vectors(vectors, schemes):
    nb = knn(vectors)
    return {scheme: score(nb, fn) for scheme, fn in schemes.items()}


# ==========================================
# Reporting
# ==========================================
def config_name(run_name):
    return SEED_SUFFIX.sub("", run_name)


def aggregate(rows):
    """rows: list of dicts with 'run','config','scheme', metrics -> mean/std per (config, scheme)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["scheme"])].append(r)
    out = []
    metric_keys = [f"P@{k}" for k in KS] + [f"Hit@{k}" for k in KS] + ["MRR", "mAP@10"]
    for (cfg, scheme), rs in groups.items():
        agg = {"config": cfg, "scheme": scheme, "n_runs": len(rs), "queries": rs[0]["queries"]}
        for m in metric_keys:
            vals = np.array([r[m] for r in rs])
            agg[m] = float(vals.mean())
            agg[m + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        out.append(agg)
    return out


def fmt(value, std, pct):
    if pct:
        s = f"{value:.1f}"
        return s + (f" ± {std:.1f}" if std is not None else "")
    s = f"{value:.3f}"
    return s + (f" ± {std:.3f}" if std is not None else "")


def markdown_tables(agg, order):
    lines = []
    for scheme in sorted({a["scheme"] for a in agg}):
        rows = [a for a in agg if a["scheme"] == scheme]
        rows.sort(key=lambda a: order.index(a["config"]) if a["config"] in order else len(order))
        lines.append(f"### Ground truth: `{scheme}` ({rows[0]['queries']} queries)\n")
        lines.append("| Config | Runs | P@1 | P@5 | P@10 | Hit@10 | MRR | mAP@10 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for a in rows:
            lines.append("| " + " | ".join([
                a["config"], str(a["n_runs"]),
                fmt(a["P@1"], a["P@1_std"], True), fmt(a["P@5"], a["P@5_std"], True),
                fmt(a["P@10"], a["P@10_std"], True), fmt(a["Hit@10"], a["Hit@10_std"], True),
                fmt(a["MRR"], a["MRR_std"], False), fmt(a["mAP@10"], a["mAP@10_std"], False),
            ]) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs against baselines")
    parser.add_argument("--dataset", required=True, choices=["mfcad", "fusion360"])
    parser.add_argument("--runs", nargs="+", required=True, help="Run folders (globs allowed), each with faiss_index*.bin + metadata*.pkl")
    parser.add_argument("--graphs_dir", default=None, help="Processed .pt graphs, enables the handcrafted baseline")
    parser.add_argument("--label_dir", default=None, help="MFCAD .face_truth folder, enables the feature_hist metric")
    parser.add_argument("--labels", default="data/fusion360_labels.json", help="Fusion 360 manual labels (JSON)")
    parser.add_argument("--no_baselines", action="store_true", help="Skip random/handcrafted baselines")
    parser.add_argument("--out", default=None, help="Output prefix, e.g. results/mfcad -> results/mfcad.md/.csv/.json")
    args = parser.parse_args(argv)

    run_dirs = []
    for pattern in args.runs:
        matches = sorted(glob.glob(pattern)) or [pattern]
        run_dirs += [Path(m) for m in matches if Path(m).is_dir()]

    rows, order = [], []
    gallery = schemes = None

    def add(run, cfg, results):
        if cfg not in order:
            order.append(cfg)
        for scheme, res in results.items():
            if res is not None:
                rows.append({"run": run, "config": cfg, "scheme": scheme, **res})

    for run_dir in run_dirs:
        names, vectors = load_embeddings(run_dir)
        if names is None:
            print(f"[SKIP] {run_dir}: no faiss_index*.bin / metadata*.pkl")
            continue
        if gallery is None:
            gallery, schemes = build_ground_truths(args.dataset, names, args.label_dir, args.labels)
            print(f"Gallery: {len(gallery)} models | ground truths: {list(schemes)}")
        pos = {n: i for i, n in enumerate(names)}
        missing = [n for n in gallery if n not in pos]
        if missing:
            print(f"[SKIP] {run_dir}: {len(missing)} gallery models missing from this index")
            continue
        X = vectors[[pos[n] for n in gallery]]
        results = evaluate_vectors(X, schemes)
        add(run_dir.name, config_name(run_dir.name), results)
        main_scheme = next(iter(results))
        r = results[main_scheme]
        print(f"[OK] {run_dir.name:28s} {main_scheme}: P@1={r['P@1']:.1f}  P@10={r['P@10']:.1f}  MRR={r['MRR']:.3f}")

    if gallery is None:
        print("No runs could be evaluated.")
        return 1

    if not args.no_baselines:
        rng = np.random.default_rng(0)
        for s in range(5):
            res = evaluate_vectors(rng.normal(size=(len(gallery), 64)).astype("float32"), schemes)
            add(f"baseline_random_s{s}", "baseline_random", res)
        print("[OK] random baseline (5 draws)")
        if args.graphs_dir:
            X, missing = handcrafted_embeddings(gallery, args.graphs_dir)
            if X is None:
                print(f"[SKIP] handcrafted baseline: no graphs found in {args.graphs_dir}")
            else:
                if missing:
                    print(f"[WARN] handcrafted baseline: {missing} graphs missing (zero vectors used)")
                add("baseline_handcrafted", "baseline_handcrafted", evaluate_vectors(X, schemes))
                print("[OK] handcrafted baseline")

    agg = aggregate(rows)
    table = markdown_tables(agg, order)
    print("\n" + table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# Retrieval results: {args.dataset}\n\n"
                  f"Generated by `python {' '.join(['scripts/evaluate_retrieval.py'] + (argv if argv is not None else sys.argv[1:]))}`\n\n"
                  "P@k = precision in top k (%), Hit@k = queries with >= 1 relevant result in top k (%), "
                  "MRR / mAP over the top 10. `± x` = std over seeds.\n\n")
        out.with_suffix(".md").write_text(header + table, encoding="utf-8")
        with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"per_run": rows, "aggregated": agg}, f, indent=1)
        import csv
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            writer.writeheader()
            writer.writerows(agg)
        print(f"Saved {out.with_suffix('.md')}, .csv, .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
