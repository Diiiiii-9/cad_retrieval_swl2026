#!/bin/bash
# =====================================================================
# Final MFCAD experiment: old vs. new features, best configs, 3 seeds each.
# Every config is trained with the same number of epochs so rows are comparable.
#
#   bash scripts/lrz/submit_mfcad_final.sh            # submit all jobs
#   EPOCHS=50 SEEDS="0 1" bash scripts/lrz/submit_mfcad_final.sh
#   DRY_RUN=1 bash scripts/lrz/submit_mfcad_final.sh  # only print what would be submitted
#
# Afterwards (CPU is enough):  bash scripts/lrz/evaluate_all.sh
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2}"
OUT=checkpoints/final_mfcad

# name | training flags
CONFIGS=(
  "raw_mean_dim64|--features raw --pooling mean --out_dim 64"              # = old baseline, now with seeds
  "norm_mean_dim64|--features normalized --pooling mean --out_dim 64"      # effect of the feature fix alone
  "norm_max_dim256|--features normalized --pooling max --out_dim 256"      # best pooling (class labels) + best dim
  "norm_maxmean_dim256|--features normalized --pooling max_mean --out_dim 256"  # best pooling (histogram metric) + best dim
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"; flags="${entry#*|}"
  for seed in $SEEDS; do
    run="$OUT/${name}_s${seed}"
    args="$flags --epochs $EPOCHS --seed $seed"
    echo "submit $run :: $args"
    if [ -z "${DRY_RUN:-}" ]; then
      DATASET=mfcad RUN_DIR="$run" TRAIN_ARGS="$args" \
        sbatch --job-name="mf_${name}_s${seed}" --export=ALL scripts/lrz/train_and_index.sbatch
    fi
  done
done
