"""
Plot Retrieval Results
======================
Turns the JSON written by scripts/evaluate_retrieval.py into report-ready PNGs:
one horizontal bar chart per ground-truth scheme, learned configs in blue,
baselines in gray, error bars = std over seeds (when a config has >1 run).

Usage:
    python scripts/plot_results.py results/mfcad_final.json --metric P@1
    python scripts/plot_results.py results/fusion360.json --metric P@10
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#898781"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BASELINES = {"baseline_random": "baseline: random", "baseline_handcrafted": "baseline: handcrafted"}


def plot_scheme(rows, scheme, metric, title, out_path):
    rows = sorted(rows, key=lambda r: r[metric])
    labels = [BASELINES.get(r["config"], r["config"]) for r in rows]
    values = [r[metric] for r in rows]
    errors = [r.get(metric + "_std") or 0.0 for r in rows]
    colors = [BASELINE_COLOR if r["config"] in BASELINES else MODEL_COLOR for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(labels, values, color=colors, height=0.6)
    for bar, e in zip(bars, errors):  # error bars only for configs that have >1 seed
        if e:
            ax.errorbar(bar.get_width(), bar.get_y() + bar.get_height() / 2, xerr=e,
                        color=INK_2, lw=1, capsize=2)
    pct = metric != "MRR" and not metric.startswith("mAP")
    for bar, v, e in zip(bars, values, errors):
        text = f"{v:.1f}" if pct else f"{v:.3f}"
        if e:
            text += f" ± {e:.1f}" if pct else f" ± {e:.3f}"
        ax.text(bar.get_width() + (e or 0) + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, text,
                va="center", fontsize=8, color=INK_2)
    ax.set_xlabel(f"{metric}{' (%)' if pct else ''}", color=INK_2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlim(0, max(v + e for v, e in zip(values, errors)) * 1.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c7c1")
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.grid(True, color="#e6e4df", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    parser.add_argument("--metric", default="P@1", help="P@1, P@5, P@10, Hit@10, MRR or mAP@10")
    parser.add_argument("--out_dir", default=None, help="Defaults to the JSON's folder")
    args = parser.parse_args()

    path = Path(args.results_json)
    agg = json.loads(path.read_text())["aggregated"]
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for scheme in sorted({r["scheme"] for r in agg}):
        rows = [r for r in agg if r["scheme"] == scheme]
        n_q = rows[0]["queries"]
        title = f"{path.stem}: {args.metric} by config (ground truth: {scheme}, {n_q} queries)"
        safe_metric = args.metric.replace("@", "at")
        out = out_dir / f"{path.stem}_{scheme}_{safe_metric}.png"
        plot_scheme(rows, scheme, args.metric, title, out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
