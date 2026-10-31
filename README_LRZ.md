# LRZ AI Systems Setup & Execution Manual: CAD Retrieval Project

This guide provides the complete, step-by-step workflow for deploying and running the `cad_retrieval_swl2026` pipeline on the LRZ AI Systems. 

> **CRITICAL WARNING:** The LRZ AI Systems and the standard CoolMUC Linux Cluster are isolated hardware environments. **Always use the AI Systems login node (`login.ai.lrz.de`)** for this project. Never run compute-heavy tasks or parallel preprocessing directly on the login node, or your account may be banned.


## Phase 1: Network & Access

1. **Connect to the Network:** You must be connected to the eduroam WiFi or use the AnyConnect VPN to access the Munich Scientific Network (MWN).
2. **Configure VS Code:**
   Install the "Remote - SSH" extension in VS Code. Add the following to your SSH configuration file (`F1` -> `Remote-SSH: Open SSH Configuration File`):
   ```text
   Host lrz-ai-login
       HostName login.ai.lrz.de
       User <your_lrz_username>
    ```

3. **Login:** Connect to `lrz-ai-login` via VS Code and complete the 2FA authentication.



## Phase 2: Independent Environment Setup

Do not rely on the pre-installed LRZ modules or Jupyter NGC containers, as they conflict with the complex C++ backend of `pythonocc-core`. We will install a private Conda engine.

### 1. Install Private Miniforge

Run the following in the login node terminal:

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
conda init bash

```

*Close your current terminal and open a new one in VS Code. You should see `(base)` at the start of your prompt.*

### 2. Create the Virtual Environment

Create a strict Python 3.12 environment:

```bash
conda create -n cad_retrieval python=3.12 -y
conda activate cad_retrieval

```

### 3. Install Dependencies (Strict Order Required)

To prevent channel conflicts between Conda and Pip (which causes missing `libmkl` or `GLIBC` errors), follow this exact installation order:

```bash
# Step A: Install OpenCASCADE backend via conda-forge
conda install -c conda-forge pythonocc-core -y

# Step B: Install PyTorch with CUDA 12.4 via pip (Bundles all .so libraries)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step C: Install ML, Graph, and UI dependencies
pip install torch_geometric tqdm faiss-cpu streamlit plotly trimesh pandas scikit-learn matplotlib seaborn

```



## Phase 3: Repository & Data Preparation

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/Diiiiii-9/cad_retrieval_swl2026.git
cd cad_retrieval_swl2026
mkdir -p data/raw_step/mfcad

```

### 2. Transfer the Dataset

Open a **local terminal on your own computer** (not the VS Code cluster terminal) and use `rsync` to upload the raw CAD data. It supports resume if the VPN drops:

```bash
rsync -avzP /path/to/local/mfcad_dataset/ <your_lrz_username>@login.ai.lrz.de:/dss/dsshome1/07/<your_lrz_username>/cad_retrieval_swl2026/data/raw_step/mfcad/

```



## Phase 4: Execution Pipeline

### Step 1: Data Preprocessing (Interactive Job)

This step uses parallel CPU processing to convert `.step` files to `.pt` graphs. Request a short interactive node to avoid login node penalties.

```bash
# Request an interactive node for 30 minutes
salloc -p lrz-cpu -N 1 --time=00:30:00

# Once granted, activate environment and run processing
cd ~/cad_retrieval_swl2026
source ~/miniforge3/bin/activate
conda activate cad_retrieval
python -m src.data_converter --input ./data/raw_step/mfcad --output ./data/processed_graphs/mfcad

# Exit the compute node to free resources
exit

```

### Step 2: Unsupervised Model Training (Batch Job)

Training requires GPU resources. We will submit a background job to the `lrz-v100x2` partition.

Create a file named `run_training.sh` in the project root:

```bash
#!/bin/bash
#SBATCH --job-name=cad_gine_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd ~/cad_retrieval_swl2026
source ~/miniforge3/bin/activate
conda activate cad_retrieval

python -m src.train_unsupervised --data_dir ./data/processed_graphs/mfcad --epochs 100 --batch_size 32

```

Submit and monitor the job:

```bash
# Submit to the queue
sbatch run_training.sh

# Check queue status (Do not run this more than once every 10 minutes)
squeue -u <your_lrz_username>

# View real-time training logs
tail -f train_*.out

```


## Phase 5: Final Experiments (seed runs, Fusion 360 ablation, evaluation)

Ready-made SLURM scripts live in `scripts/lrz/` and are documented in the main `README.md`
(section "Final Experiments on the LRZ AI Systems"). In short:

```bash
cd ~/cad_retrieval_swl2026
DRY_RUN=1 bash scripts/lrz/submit_mfcad_final.sh   # check what will be submitted
bash scripts/lrz/submit_mfcad_final.sh             # 12 GPU jobs (4 configs x 3 seeds)
bash scripts/lrz/submit_fusion360_ablation.sh      # 12 GPU jobs (Fusion 360 ablation)

# after all jobs finished, on a CPU node (not the login node):
salloc -p lrz-cpu -N 1 --time=01:00:00
bash scripts/lrz/evaluate_all.sh
```

Each job trains one config and builds its FAISS index; re-submitting skips runs whose
`final_encoder_weights.pth` already exists, so a failed job can simply be resubmitted.
