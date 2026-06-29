# CAD Model Retrieval via Unsupervised Graph Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-red.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![pythonocc-core](https://img.shields.io/badge/pythonocc--core-7.7-green.svg)](https://github.com/tpaviot/pythonocc-core)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)

## 📌 Project Overview
This project presents an end-to-end prototype for **3D CAD Model Retrieval** using unsupervised graph machine learning. Developed as part of the **Software Lab 2026** at the Technical University of Munich (TUM), this pipeline processes boundary representation (B-Rep) CAD data from `.step` files, converts them into Attributed Adjacency Graphs (AAG), and extracts high-dimensional geometric embeddings to find and cluster geometrically similar models.

### ✨ Key Features:
1. **Robust B-Rep Parsing:** Converts raw STEP files to PyTorch Geometric (PyG) graphs using `pythonocc-core`, mapping complex topological entities (Faces/Edges) to graph Nodes and Edges with explicit geometric attributes.
2. **Unsupervised Contrastive Learning:** Trains a GINE (Graph Isomorphism Network) Encoder using InfoNCE loss and graph augmentations (edge dropout, node feature noise) without requiring manually labeled data (SimCLR framework). Architecture choices (pooling strategy, layer depth) and training behavior (augmentation strategy, temperature, embedding size) are all configurable via CLI flags — see [Ablation Study Pipeline](#-ablation-study-pipeline) below.
3. **Fast Similarity Search:** Utilizes the FAISS vector database for millisecond-level cosine similarity retrieval using L2-normalized graph embeddings.
4. **Interactive 3D Web UI:** Features a Streamlit-based dashboard with dynamic STEP-to-Mesh conversion and real-time interactive 3D rendering using `Plotly` and `Trimesh`.
5. **Comprehensive Evaluation & Ablation Studies:** A manufacturing-aware retrieval evaluation (Recall@K, MRR) is used to systematically ablate five independent design choices: embedding dimension, contrastive temperature, pooling strategy, augmentation mode, and GNN layer depth. Each ablation isolates exactly one variable while holding all others fixed, with results consolidated in `notebooks/ablation_study_final.ipynb`.


## 📂 Repository Structure
```text
cad_retrieval_project/
│
├── data/                      # Data storage (Ignored by Git)
│   ├── raw_step/mfcad/        # Place original .step/.stp files here
│   ├── raw_step/mfcad_label/  # Ground-truth per-face labels (.face_truth), used only for evaluation
│   └── processed_graphs/mfcad/# Generated PyG .pt files
│
├── checkpoints/               # Trained model weights and FAISS indices
│   ├── best_contrastive_model.pth
│   ├── final_encoder_weights.pth
│   ├── faiss_index.bin        # FAISS vector database
│   ├── metadata.pkl           # Filename mapping for the vector index
│   └── sweep/                 # One subfolder per ablation config (see below), each
│                               # containing its own weights + FAISS index + metadata
│
├── src/                       # Core backend logic
│   ├── __init__.py
│   ├── data_converter.py      # Multiprocessing STEP-to-Graph converter
│   ├── dataset.py             # PyG Dataset loader with ID tracking
│   ├── networks.py            # GINE Encoder and SimCLR Wrapper architectures
│   │                          #   supports configurable pooling (mean / max / max_mean)
│   │                          #   and configurable layer depth
│   ├── train_unsupervised.py  # Contrastive training loop and InfoNCE loss
│   │                          #   supports configurable embedding size, temperature,
│   │                          #   pooling, augmentation mode, and layer depth
│   └── retrieval.py           # FAISS vector database builder and query logic
│                               #   must be called with the SAME --pooling/--num_layers
│                               #   a checkpoint was trained with
│
├── notebooks/                 # Exploratory data analysis & Ablation studies
│   ├── ablation_study_final.ipynb   # Primary deliverable: all 5 ablations, self-contained
│   ├── ablation_study_mfcad.ipynb   # Single-config retrieval evaluation (Recall@K, MRR)
│   └── ablation_visualization.ipynb # Legacy exploratory notebook (superseded by ablation_study_final.ipynb)
│
├── scripts/                    # Sweep/automation scripts (moved out of repo root)
│   ├── evaluate_sweep.py       # Evaluates every checkpoint under checkpoints/sweep/
│   │                           #   against the shared manufacturing-aware metric
│   │                           #   (same logic as ablation_study_final.ipynb, script form)
│   ├── run_*_sweep.ps1         # Trains a batch of ablation configs sequentially (Windows)
│   └── build_*_databases.ps1   # Builds the FAISS index for a batch of ablation configs
│                                #   all scripts in this folder resolve paths relative to
│                                #   the repo root, so they can be run from anywhere
│
├── app.py                     # Web Interface (Streamlit)
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation
```


## ⚙️ Installation & Requirements

A Conda environment is **strictly required** to handle the complex C++ backend dependencies of `pythonocc-core`.

```bash
# 1. Create and activate the Conda environment
conda create -n cad_retrieval python=3.12 -y
conda activate cad_retrieval

# 2. Install OpenCASCADE Python bindings (Must be installed via conda-forge)
conda install -c conda-forge pythonocc-core

# 3. Install PyTorch (Adjust the CUDA version based on your GPU, e.g., cu126/cu128)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install remaining ML, Graph, and UI dependencies
pip install torch_geometric tqdm faiss-cpu streamlit plotly trimesh pandas scikit-learn matplotlib seaborn
```

> **Note (CPU-only / ARM machines):** `pythonocc-core` is only required for **Step 1** (converting raw `.step` files into graphs). If you already have processed `.pt` graphs and label files, Steps 2–5 below run on plain PyTorch/PyG and work fine CPU-only, including on ARM-based Windows laptops (no CUDA, no `pythonocc-core` needed).


## 🚀 Pipeline & Usage

**Important Execution Rule:** To avoid Python import path errors, always execute the scripts as modules from the project's **root directory** using the `-m` flag.

### Step 1: Data Preprocessing

Place your raw CAD models inside `data/raw_step/mfcad/`. The data converter uses parallel processing to parse topological and geometric features into PyG `.pt` files.

```bash
python -m src.data_converter --input ./data/raw_step/mfcad --output ./data/processed_graphs/mfcad
```

### Step 2: Unsupervised Model Training

Train the GINE encoder using graph augmentations and contrastive learning. The script saves the best checkpoint automatically.

```bash
python -m src.train_unsupervised --data_dir ./data/processed_graphs/mfcad --epochs 100 --batch_size 32
```

*(Tip: Training can be safely interrupted at any time using `Ctrl+C`, and the best checkpoint will be preserved).*

**Available training flags:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--hidden_dim` | `128` | GNN hidden dimension |
| `--out_dim` | `64` | Final CAD embedding dimension |
| `--temp` | `0.1` | Temperature for the InfoNCE contrastive loss |
| `--pooling` | `mean` | Graph readout strategy: `mean`, `max`, or `max_mean` (concatenated) |
| `--aug_mode` | `both` | Which contrastive augmentation(s) to apply: `both`, `edge_drop`, or `feature_noise` |
| `--num_layers` | `3` | Number of GINE message-passing layers |
| `--save_dir` | `./checkpoints` | Where to save weights for this run |

### Step 3: Build Vector Database

Pass all processed CAD graphs through the trained encoder to generate embeddings, and build the FAISS similarity index.

```bash
python -m src.retrieval --build --data_dir ./data/processed_graphs/mfcad --save_dir ./checkpoints
```

**⚠️ Important:** if you trained with non-default `--pooling`, `--out_dim`, or `--num_layers`, you must pass the **same values** here, e.g.:

```bash
python -m src.retrieval --build --save_dir ./checkpoints/sweep/pooling_max_mean --pooling max_mean --out_dim 64
```

Mismatched values will either crash on weight loading or silently build an index from a wrong-shaped network.

### Step 4: Launch 3D Web Application

Start the interactive 3D retrieval dashboard. Upload a query `.step` file to instantly find and render the Top-K most similar geometric models from your database.

```bash
streamlit run app.py
```

### Step 5: Post-Training Evaluation & Ablation Study

Run the primary ablation notebook to re-evaluate every trained config and reproduce the full results table, written interpretation, and charts:

```bash
jupyter notebook notebooks/ablation_study_final.ipynb
```

This notebook assumes the following are already built and present in `checkpoints/sweep/<config_name>/`: `final_encoder_weights.pth`, `faiss_index.bin`, and `metadata.pkl`, for each config it evaluates. See [Ablation Study Pipeline](#-ablation-study-pipeline) below for how to (re)generate these.


## 🔬 Ablation Study Pipeline

Five independent design choices are ablated, each isolating exactly one variable while holding everything else (epochs, dataset, batch size, and all other flags above) fixed:

| Ablation | Values tested | Key finding |
|---|---|---|
| Embedding dimension (`--out_dim`) | 32, 64, 128, 256 | Monotonic improvement, no ceiling found at 256 |
| Temperature (`--temp`) | 0.05, 0.1, 0.5 | Inverted-U; 0.1 is optimal |
| Pooling (`--pooling`) | mean, max, max_mean | max_mean slightly outperforms mean; max alone underperforms |
| Augmentation (`--aug_mode`) | both, edge_drop, feature_noise | Edge dropout alone nearly matches the combined baseline; feature noise alone is the weakest config tested |
| Layer depth (`--num_layers`) | 2, 3, 5 | Diminishing returns past 3 layers |

Full methodology, results table, and written interpretation (including honest single-run caveats) live in `notebooks/ablation_study_final.ipynb`.

**To reproduce or extend an ablation:**

1. **Train** the new config(s) — either call `src.train_unsupervised` directly with the relevant flag(s), or adapt one of the `scripts/run_*_sweep.ps1` scripts (they train a batch of configs sequentially, skip already-completed ones if re-run, and log each run to `logs/`).
2. **Build** the FAISS index for each new config — either call `src.retrieval --build` directly, or adapt one of the `scripts/build_*_databases.ps1` scripts. **Always pass the same `--pooling`/`--out_dim`/`--num_layers` the config was trained with.**
3. **Evaluate** — add the new config's settings to the `CONFIGS` list in `scripts/evaluate_sweep.py` (or directly in `ablation_study_final.ipynb`) and re-run with `python scripts/evaluate_sweep.py`. The evaluation uses a manufacturing-aware proxy: two models are considered a retrieval "match" if their per-face machining-feature-class histograms have cosine similarity ≥ 0.95.

**Note on `data/raw_step/mfcad_label/`:** this folder contains the ground-truth `.face_truth` label files used *only* for evaluation (not for training, since the model is unsupervised). If missing, it can be regenerated by cloning the original [`hducg/MFCAD`](https://github.com/hducg/MFCAD) dataset repository and copying its `.face_truth` files (found under `dataset/step/`) into `data/raw_step/mfcad_label/`.


## 👥 Team

**Software Lab 2026 - Group Project**

* Di Liu
* Ayse Seray Seker
* Eduardo Dall'Igna

## 🎓 Acknowledgements

This project is developed under the guidance of Konstantinos Gkrispanis and Dr. Stavros Nousias at the Technical University of Munich (TUM).
