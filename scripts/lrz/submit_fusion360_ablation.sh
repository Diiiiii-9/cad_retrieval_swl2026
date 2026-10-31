#!/bin/bash
# =====================================================================
# Fusion 360 ablation study -- same five ablations as the MFCAD study
# (embedding dim, temperature, pooling, augmentation, layers), one variable at a time.
# Uses the normalized features (scale invariance matters more for Fusion 360,
# whose parts come in arbitrary units and sizes).
#
# Prerequisite: graphs in data/processed_graphs/fusion360
#   (python -m src.fusion360.build_fusion360_graphs, run on a CPU node)
#
#   bash scripts/lrz/submit_fusion360_ablation.sh
#   SEEDS="0 1 2" bash scripts/lrz/submit_fusion360_ablation.sh   # if GPU time allows
#   DRY_RUN=1 bash scripts/lrz/submit_fusion360_ablation.sh
#
# Afterwards:  bash scripts/lrz/evaluate_all.sh
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0}"
OUT=checkpoints/sweep_fusion360
BASE="--features normalized --out_dim 64 --temp 0.1 --pooling mean --aug_mode both --num_layers 3"

# name | flags that override BASE (later flags win in argparse)
CONFIGS=(
  "baseline|"
  "dim32|--out_dim 32"   "dim128|--out_dim 128"   "dim256|--out_dim 256"
  "temp005|--temp 0.05"  "temp05|--temp 0.5"
  "pooling_max|--pooling max"   "pooling_max_mean|--pooling max_mean"
  "aug_edge_drop|--aug_mode edge_drop"   "aug_feature_noise|--aug_mode feature_noise"
  "layers2|--num_layers 2"   "layers5|--num_layers 5"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"; flags="${entry#*|}"
  for seed in $SEEDS; do
    run="$OUT/${name}_s${seed}"
    args="$BASE $flags --epochs $EPOCHS --seed $seed"
    echo "submit $run :: $args"
    if [ -z "${DRY_RUN:-}" ]; then
      DATASET=fusion360 RUN_DIR="$run" TRAIN_ARGS="$args" \
        sbatch --job-name="f360_${name}_s${seed}" --export=ALL scripts/lrz/train_and_index.sbatch
    fi
  done
done
