#!/bin/bash
# =====================================================================
# One training run + its FAISS index, as a single LRZ GPU job.
# Normally submitted by submit_mfcad_final.sh / submit_fusion360_ablation.sh,
# which set these environment variables:
#   DATASET    mfcad | fusion360
#   RUN_DIR    output folder, e.g. checkpoints/final/norm_mean_dim64_s0
#   TRAIN_ARGS extra flags for the training script, e.g. "--features normalized --seed 0"
# Manual use:
#   DATASET=mfcad RUN_DIR=checkpoints/test TRAIN_ARGS="--epochs 2" sbatch scripts/lrz/train_and_index.sbatch
# =====================================================================
#SBATCH --job-name=cad_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/cad_retrieval_swl2026}"
source ~/miniforge3/bin/activate
conda activate cad_retrieval

: "${DATASET:?set DATASET=mfcad or fusion360}"
: "${RUN_DIR:?set RUN_DIR}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

if [ "$DATASET" = "mfcad" ]; then
    TRAIN_MOD=src.train_unsupervised;               INDEX_MOD=src.retrieval
    DATA_DIR=./data/processed_graphs/mfcad
else
    TRAIN_MOD=src.fusion360.train_unsupervised_fusion360; INDEX_MOD=src.fusion360.retrieval_fusion360
    DATA_DIR=./data/processed_graphs/fusion360
fi

echo "[$(date)] Training $RUN_DIR on $DATASET with: $TRAIN_ARGS"
if [ -f "$RUN_DIR/final_encoder_weights.pth" ]; then
    echo "Weights already exist, skipping training (delete $RUN_DIR to retrain)."
else
    # shellcheck disable=SC2086
    python -m $TRAIN_MOD --data_dir "$DATA_DIR" --save_dir "$RUN_DIR" --num_workers 4 $TRAIN_ARGS
fi

echo "[$(date)] Building FAISS index (architecture read from $RUN_DIR/config.json)"
python -m $INDEX_MOD --build --data_dir "$DATA_DIR" --save_dir "$RUN_DIR"
echo "[$(date)] Done."
