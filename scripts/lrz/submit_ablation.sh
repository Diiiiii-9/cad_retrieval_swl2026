#!/bin/bash
# =====================================================================
# Ablation Job Submission (LRZ)
# @Author: Eduardo Dall Igna
# @Date: 2026-11-10
# @Description:
# Full ablation on the updated code: one (dataset, feature mode) grid per call.
#   DATASET=mfcad     FEATURES=raw        bash scripts/lrz/submit_ablation.sh
#   DATASET=mfcad     FEATURES=normalized bash scripts/lrz/submit_ablation.sh
#   DATASET=fusion360 FEATURES=raw        bash scripts/lrz/submit_ablation.sh
#   SWEEP=masks DATASET=... FEATURES=... -> input ablation (feature groups hidden) instead of hyper-parameters
#   DRY_RUN=1 ...                      -> only print what would be submitted
#   OUT=checkpoints/ablation_smoke EPOCHS=1 SEEDS=0 ... -> 1-epoch smoke test of every config
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

DATASET="${DATASET:?set DATASET=mfcad or fusion360}"
FEATURES="${FEATURES:?set FEATURES=raw or normalized}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-checkpoints/ablation/${DATASET}_${FEATURES}}"   # override only for smoke tests

if [ "$DATASET" = "mfcad" ]; then INDEX=faiss_index.bin
else                              INDEX=faiss_index_fusion360.bin; fi

BASE="--features $FEATURES --out_dim 64 --temp 0.1 --pooling mean --aug_mode both --num_layers 3 --epochs $EPOCHS"

CONFIGS=(
  "baseline|"
  "dim32|--out_dim 32"   "dim128|--out_dim 128"   "dim256|--out_dim 256"
  "temp005|--temp 0.05"  "temp05|--temp 0.5"
  "pooling_max|--pooling max"   "pooling_max_mean|--pooling max_mean"
  "aug_edge_drop|--aug_mode edge_drop"   "aug_feature_noise|--aug_mode feature_noise"
  "layers2|--num_layers 2"   "layers5|--num_layers 5"
)
# SWEEP=masks -> input ablation instead: hide one feature group per config (needs --mask, src/features.py)
SWEEP="${SWEEP:-hparams}"
if [ "$SWEEP" = "masks" ]; then
  CONFIGS=(
    "mask_no_size|--mask no_size"
    "mask_no_type|--mask no_type"
    "mask_no_closed|--mask no_closed"
    "mask_no_edge_attr|--mask no_edge_attr"
    "mask_topology_only|--mask topology_only"
  )
elif [ "$SWEEP" != "hparams" ]; then
  echo "SWEEP must be 'hparams' (default) or 'masks', got '$SWEEP'"; exit 1
fi

# Jobs already waiting/running in SLURM (so a re-run never submits a duplicate)
QUEUED="$(squeue -u "${USER:-$(whoami)}" -h -o "%j" 2>/dev/null | awk '{print $1}' || true)"

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"; flags="${entry#*|}"
  for seed in $SEEDS; do
    run="$OUT/${name}_s${seed}"
    job="${DATASET:0:4}_${FEATURES:0:4}_${name}_s${seed}"
    if [ -f "$run/$INDEX" ]; then echo "skip   $run (done)"; continue; fi
    if grep -qx "$job" <<< "$QUEUED"; then echo "skip   $run (already in queue)"; continue; fi
    args="$BASE $flags --seed $seed"
    echo "submit $run :: $args"
    if [ -z "${DRY_RUN:-}" ]; then
      if ! DATASET="$DATASET" RUN_DIR="$run" TRAIN_ARGS="$args" \
           sbatch --time="${TIME:-01:00:00}" --job-name="$job" \
                  --export=ALL scripts/lrz/train_and_index.sbatch; then
        echo ">>> SLURM refused the job (probably the per-user job limit)."
        echo ">>> Nothing is lost: re-run this exact command later to submit the rest."
        exit 0
      fi
    fi
  done
done
