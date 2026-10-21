#!/bin/bash
#SBATCH -J step2stl
#SBATCH -D ./
#SBATCH -o ./%x.%j.%N.out
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=04:00:00

module load slurm_setup

cd ~/cad_retrieval_swl2026
source ~/miniforge3/bin/activate 
conda activate cad_retrieval

echo "Starting batch conversion..."
python tools/batch_convert.py
echo "Job finished!"