#!/bin/bash
# =====================================================================
# Evaluates everything that has been trained so far and writes results/*.md|csv|json.
# CPU only -- run inside an interactive CPU job, not on the login node:
#   salloc -p lrz-cpu -N 1 --time=01:00:00
#   bash scripts/lrz/evaluate_all.sh
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/miniforge3/bin/activate
conda activate cad_retrieval
mkdir -p results

MFCAD_GRAPHS=data/processed_graphs/mfcad
F360_GRAPHS=data/processed_graphs/fusion360
MFCAD_LABELS=data/raw_step/mfcad_label

# 1. Original 30-epoch MFCAD ablation (single runs) + baselines
python scripts/evaluate_retrieval.py --dataset mfcad --runs checkpoints/sweep/* \
    --graphs_dir $MFCAD_GRAPHS --label_dir $MFCAD_LABELS --out results/mfcad_ablation_30ep

# 2. Final MFCAD runs (3 seeds -> mean +- std) + baselines
if compgen -G "checkpoints/final_mfcad/*" > /dev/null; then
  python scripts/evaluate_retrieval.py --dataset mfcad --runs checkpoints/final_mfcad/* \
      --graphs_dir $MFCAD_GRAPHS --label_dir $MFCAD_LABELS --out results/mfcad_final
fi

# 3. Cross-dataset transfer: index Fusion 360 with an MFCAD-trained encoder (no training needed)
XFER=checkpoints/fusion360_from_mfcad
if [ ! -f $XFER/faiss_index_fusion360.bin ]; then
  python -m src.fusion360.retrieval_fusion360 --build --data_dir $F360_GRAPHS \
      --weights_dir checkpoints --save_dir $XFER
fi

# 4. Fusion 360: trained-on-Fusion encoder vs transfer vs ablation sweep + baselines
python scripts/evaluate_retrieval.py --dataset fusion360 \
    --runs checkpoints/fusion360 $XFER checkpoints/sweep_fusion360/* \
    --labels data/fusion360_labels.json --graphs_dir $F360_GRAPHS --out results/fusion360

# 5. Are the many identical Fusion 360 embeddings real duplicate parts or a collapse?
python scripts/check_duplicates.py --run_dir checkpoints/fusion360 --graphs_dir $F360_GRAPHS \
    | tee results/fusion360_duplicates_raw_features.txt
for run in checkpoints/sweep_fusion360/baseline_s0; do
  [ -d "$run" ] && python scripts/check_duplicates.py --run_dir "$run" --graphs_dir $F360_GRAPHS \
    | tee results/fusion360_duplicates_normalized_features.txt
done

# 6. Charts
for f in results/*.json; do python scripts/plot_results.py "$f" --metric P@1; done

echo "Results written to results/"
