#!/bin/bash
# =====================================================================
# Makes a trained run the model served by the Streamlit app.
#   bash scripts/lrz/deploy_model.sh mfcad     checkpoints/final_mfcad/raw_mean_dim64_s0
#   bash scripts/lrz/deploy_model.sh fusion360 checkpoints/sweep_fusion360/layers2_s2
# Copies weights, config.json and the FAISS index into checkpoints/ (MFCAD page)
# or checkpoints/fusion360/ (Fusion 360 page). Old files are moved to a backup folder.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET="${1:?usage: deploy_model.sh mfcad|fusion360 <run_dir>}"
RUN="${2:?usage: deploy_model.sh mfcad|fusion360 <run_dir>}"

if [ "$DATASET" = "mfcad" ]; then
  DEST=checkpoints; FILES="final_encoder_weights.pth config.json faiss_index.bin metadata.pkl"
else
  DEST=checkpoints/fusion360; FILES="final_encoder_weights.pth config.json faiss_index_fusion360.bin metadata_fusion360.pkl"
fi

for f in $FILES; do
  [ -f "$RUN/$f" ] || { echo "Missing $RUN/$f -- was the index built for this run?"; exit 1; }
done

BACKUP="$DEST/previous_model_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
for f in $FILES best_contrastive_model.pth; do
  [ -f "$DEST/$f" ] && mv "$DEST/$f" "$BACKUP/"
done
for f in $FILES; do cp "$RUN/$f" "$DEST/"; done
echo "$RUN" > "$DEST/DEPLOYED_FROM.txt"
echo "Deployed $RUN -> $DEST (previous files in $BACKUP)"
