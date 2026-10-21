#!/bin/bash
#SBATCH --job-name=cad_build_faiss
#SBATCH --output=faiss_%j.out
#SBATCH --error=faiss_%j.err
#SBATCH --partition=lrz-v100x2 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd ~/cad_retrieval_swl2026
source ~/miniforge3/bin/activate 
conda activate cad_retrieval

python -m src.retrieval --build --data_dir ./data/processed_graphs/mfcad --save_dir ./checkpoints