#!/bin/bash
#SBATCH --job-name=cad_gine_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=lrz-v100x2 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd ~/cad_retrieval_swl2026
source ~/miniforge3/bin/activate
conda activate cad_retrieval
python -m src.train_unsupervised --data_dir ./data/processed_graphs/mfcad --epochs 100 --batch_size 32