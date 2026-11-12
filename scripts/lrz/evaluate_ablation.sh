#!/bin/bash
# =====================================================================
# Ablation Evaluation per Cell (LRZ)
# @Author: Eduardo Dall Igna
# @Date: 2026-11-11
# @Description:
# Evaluates the ablation grid: ONE results table per cell (dataset x feature mode),
# so baselines of different cells are never averaged together.
#
#   bash scripts/lrz/evaluate_ablation.sh                 # all cells that have finished runs
#   bash scripts/lrz/evaluate_ablation.sh mfcad_raw       # only one cell
#   DRY_RUN=1 bash scripts/lrz/evaluate_ablation.sh       # only print the commands
#
# CPU is enough, but not on the login node:
#   salloc -p lrz-cpu --qos=cpu -N 1 --time=01:00:00, then srun --pty bash
# Output: results/ablation_<cell>.md|csv|json  (+ charts via scripts/plot_results.py)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results

MFCAD_GRAPHS=data/processed_graphs/mfcad
F360_GRAPHS=data/processed_graphs/fusion360
MFCAD_LABELS=data/raw_step/mfcad_label
F360_LABELS=data/fusion360_labels.json

CELLS="${*:-mfcad_raw mfcad_normalized fusion360_raw fusion360_normalized}"

for cell in $CELLS; do
  # 1. Collect the run folders of this cell
  runs=()
  for d in checkpoints/ablation/"$cell"/*/; do [ -d "$d" ] && runs+=("${d%/}"); done
  if [ "$cell" = "fusion360_normalized" ]; then
    # Hyper-parameter grid of this cell = Di's existing sweep (same code, see git log).
    # raw_features_* belongs to the fusion360_raw cell, so it is left out here.
    for d in checkpoints/sweep_fusion360/*/; do
      case "$d" in *raw_features_*) continue ;; esac
      [ -d "$d" ] && runs+=("${d%/}")
    done
  fi

  # 2. Only finished runs (index exists) -- a run that is still training is skipped
  done_runs=()
  for r in "${runs[@]}"; do
    compgen -G "$r/faiss_index*.bin" > /dev/null && done_runs+=("$r")
  done
  echo "=== $cell: ${#done_runs[@]} finished of ${#runs[@]} run folders"
  [ ${#done_runs[@]} -eq 0 ] && continue

  # 3. Dataset-specific ground truth
  if [[ "$cell" == mfcad_* ]]; then
    args=(--dataset mfcad --graphs_dir "$MFCAD_GRAPHS" --label_dir "$MFCAD_LABELS")
  else
    args=(--dataset fusion360 --graphs_dir "$F360_GRAPHS" --labels "$F360_LABELS")
  fi

  cmd=(python scripts/evaluate_retrieval.py "${args[@]}" --runs "${done_runs[@]}" --out "results/ablation_$cell")
  if [ -n "${DRY_RUN:-}" ]; then
    echo "${cmd[@]:0:8} --runs <${#done_runs[@]} folders> --out results/ablation_$cell"
  else
    "${cmd[@]}"
    python scripts/plot_results.py "results/ablation_$cell.json" --metric P@1 || true
  fi
done
