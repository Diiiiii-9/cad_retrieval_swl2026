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
2. **Unsupervised Contrastive Learning:** Trains a GINE (Graph Isomorphism Network) Encoder using InfoNCE loss and graph augmentations (edge dropout, node feature noise) without requiring manually labeled data (SimCLR framework).
3. **Fast Similarity Search:** Utilizes the FAISS vector database for millisecond-level cosine similarity retrieval using L2-normalized graph embeddings.
4. **Interactive 3D Web UI:** Features a Streamlit-based dashboard with dynamic STEP-to-Mesh conversion and real-time interactive 3D rendering using `Plotly` and `Trimesh`.
5. **Comprehensive Evaluation:** Includes t-SNE latent space visualizations and ablation studies on feature importance evaluated via Silhouette scores.


## 📂 Repository Structure
```text
cad_retrieval_project/
│
├── data/                      # Data storage (Ignored by Git)
│   ├── raw_step/mfcad/        # Place original .step/.stp files here
│   └── processed_graphs/mfcad/# Generated PyG .pt files
│
├── checkpoints/               # Trained model weights and FAISS indices
│   ├── best_contrastive_model.pth 
│   ├── final_encoder_weights.pth
│   ├── faiss_index.bin        # FAISS vector database
│   └── metadata.pkl           # Filename mapping for the vector index
│
├── src/                       # Core backend logic
│   ├── __init__.py
│   ├── data_converter.py      # Multiprocessing STEP-to-Graph converter
│   ├── dataset.py             # PyG Dataset loader with ID tracking
│   ├── networks.py            # GINE Encoder and SimCLR Wrapper architectures
│   ├── train_unsupervised.py  # Contrastive training loop and InfoNCE loss
│   └── retrieval.py           # FAISS vector database builder and query logic
│
├── notebooks/                 # Exploratory data analysis & Ablation studies
│   └── ablation_study.ipynb   # t-SNE clustering & feature ablation evaluation
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

### Step 3: Build Vector Database

Pass all processed CAD graphs through the trained encoder to generate 64-dimensional embeddings, and build the FAISS similarity index.

```bash
python -m src.retrieval --build --data_dir ./data/processed_graphs/mfcad --save_dir ./checkpoints
```

### Step 4: Launch 3D Web Application

Start the interactive 3D retrieval dashboard. Upload a query `.step` file to instantly find and render the Top-K most similar geometric models from your database.

```bash
streamlit run app.py
```

### Step 5: Post-Training Evaluation

Run the Jupyter Notebook to generate t-SNE latent space visualizations and perform a feature ablation study measuring Silhouette clustering scores:

```bash
jupyter notebook notebooks/ablation_study.ipynb
```



## 👥 Team

**Software Lab 2026 - Group Project**

* Di Liu
* Ayse Seray Seker
* Eduardo Dall'Igna

## 🎓 Acknowledgements

This project is developed under the guidance of Konstantinos Gkrispanis and Dr. Stavros Nousias at the Technical University of Munich (TUM).

