"""
Cross-Cell Comparison of the Ablation Study
===========================================
@Author: Eduardo Dall Igna
@Date: 2026-11-12
@Description:
Cross-cell overview of the ablation study.

Reads the per-cell results written by scripts/lrz/evaluate_ablation.sh
(results/ablation_<cell>.json) and prints ONE table: every config as a row,
every (cell, ground truth) as a column, each entry = the difference to that
cell's baseline plus a verdict from the 2x-std rule (Methods, Section 5):

    effect      |delta| >  2 * s_max          marked  ++ / --
    borderline  s_max < |delta| <= 2 * s_max  marked  +  / -
    no effect   |delta| <= s_max              marked  .

s_max = the larger of the two standard deviations over seeds (config, baseline).
A difference smaller than --min_delta (default 1 point for P@k/Hit@k, 0.01 for MRR/mAP)
is always reported as "." -- with very small seed spreads, the 2x rule alone would call
differences "effects" that are far too small to matter in practice.

    python scripts/compare_cells.py                      # P@1, all cells found
    python scripts/compare_cells.py --metric mAP@10
    python scripts/compare_cells.py --out results/ablation_overview
"""

import argparse
import json
from pathlib import Path

# (cell, ground truth, column title) -- the columns of the overview, in order
COLUMNS = [
    ("mfcad_raw", "class_L1", "MFCAD raw · L1"),
    ("mfcad_raw", "feature_hist", "MFCAD raw · hist"),
    ("mfcad_normalized", "class_L1", "MFCAD norm · L1"),
    ("mfcad_normalized", "feature_hist", "MFCAD norm · hist"),
    ("fusion360_raw", "category", "F360 raw"),
    ("fusion360_normalized", "category", "F360 norm"),
]

ROW_ORDER = ["dim32", "dim128", "dim256", "temp005", "temp05", "pooling_max", "pooling_max_mean",
             "aug_edge_drop", "aug_feature_noise", "layers2", "layers5",
             "mask_no_size", "mask_no_type", "mask_no_closed", "mask_no_edge_attr", "mask_topology_only",
             "baseline_handcrafted", "baseline_random"]


def verdict(delta, s_max, min_delta=0.0):
    """Returns the marker for one difference (see module docstring)."""
    if abs(delta) < min_delta:
        return "."
    if s_max is None or s_max == 0:
        return "?"
    if abs(delta) > 2 * s_max:
        return "++" if delta > 0 else "--"
    if abs(delta) > s_max:
        return "+" if delta > 0 else "-"
    return "."


def load_cells(results_dir):
    """{(cell, scheme): {config: aggregated row}} for every results file that exists."""
    table = {}
    for cell in sorted({c for c, _, _ in COLUMNS}):
        path = Path(results_dir) / f"ablation_{cell}.json"
        if not path.exists():
            continue
        agg = json.loads(path.read_text(encoding="utf-8"))["aggregated"]
        for row in agg:
            table.setdefault((cell, row["scheme"]), {})[row["config"]] = row
    return table


def compare(table, metric, min_delta=0.0):
    """Returns (baseline row, list of (config, [cell entries])) for the chosen metric."""
    cols = [c for c in COLUMNS if (c[0], c[1]) in table]
    base_line, rows = [], []
    for cell, scheme, _ in cols:
        b = table[(cell, scheme)].get("baseline")
        base_line.append(None if b is None else (b[metric], b.get(metric + "_std"), b["n_runs"]))
    configs = [c for c in ROW_ORDER if any(c in table[(cell, s)] for cell, s, _ in cols)]
    for cfg in configs:
        entries = []
        for (cell, scheme, _), base in zip(cols, base_line):
            r = table[(cell, scheme)].get(cfg)
            if r is None or base is None:
                entries.append(None)
                continue
            delta = r[metric] - base[0]
            stds = [s for s in (r.get(metric + "_std"), base[1]) if s is not None]
            s_max = max(stds) if stds else None
            entries.append({"delta": delta, "s_max": s_max, "mark": verdict(delta, s_max, min_delta), "n_runs": r["n_runs"]})
        rows.append((cfg, entries))
    return cols, base_line, rows


def to_markdown(cols, base_line, rows, metric):
    pct = metric.startswith(("P@", "Hit@"))
    f = (lambda v: f"{v:.1f}") if pct else (lambda v: f"{v:.3f}")
    head = "| Config | " + " | ".join(t for _, _, t in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    out = [f"### {metric}: difference to each cell's baseline", "",
           "`++`/`--` effect (|Δ| > 2·s_max), `+`/`-` borderline, `.` no measurable or relevant effect, "
           "`(n)` = fewer than 3 seeds.", "", head, sep]
    cells = []
    for b in base_line:
        cells.append("—" if b is None else f"**{f(b[0])}**" + (f" ± {f(b[1])}" if b[1] is not None else ""))
    out.append("| **baseline (absolute)** | " + " | ".join(cells) + " |")
    for cfg, entries in rows:
        cells = []
        for e in entries:
            if e is None:
                cells.append("")
                continue
            sign = "+" if e["delta"] >= 0 else "−"
            note = f" ({e['n_runs']})" if e["n_runs"] < 3 and not cfg.startswith("baseline_") else ""
            cells.append(f"{sign}{f(abs(e['delta']))} `{e['mark']}`{note}")
        out.append(f"| {cfg} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--metric", default="P@1", help="P@1, P@5, P@10, Hit@10, MRR or mAP@10")
    p.add_argument("--min_delta", type=float, default=None,
                   help="smallest difference that counts (default: 1.0 for P@k/Hit@k, 0.01 for MRR/mAP)")
    p.add_argument("--out", default=None, help="prefix, e.g. results/ablation_overview -> .md and .json")
    args = p.parse_args(argv)

    table = load_cells(args.results_dir)
    if not table:
        print(f"No results/ablation_*.json found in {args.results_dir}")
        return 1
    min_delta = args.min_delta
    if min_delta is None:
        min_delta = 1.0 if args.metric.startswith(("P@", "Hit@")) else 0.01
    cols, base_line, rows = compare(table, args.metric, min_delta)
    md = to_markdown(cols, base_line, rows, args.metric)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(md, encoding="utf-8")
        data = {"metric": args.metric, "columns": [t for _, _, t in cols],
                "baseline": base_line, "rows": [{"config": c, "entries": e} for c, e in rows]}
        out.with_suffix(".json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"Saved {out.with_suffix('.md')}, .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
