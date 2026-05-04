# CAD Model Retrieval via Unsupervised Graph Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-red.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![pythonocc-core](https://img.shields.io/badge/pythonocc--core-7.7-green.svg)](https://github.com/tpaviot/pythonocc-core)

## 📌 Project Overview
This project presents an end-to-end prototype for **3D CAD Model Retrieval** using unsupervised graph machine learning. Developed as part of the **Software Lab 2026** at the Technical University of Munich (TUM), this pipeline processes boundary representation (B-Rep) CAD data from `.step` files, converts them into Attributed Adjacency Graphs (AAG), and extracts high-dimensional geometric embeddings to find and cluster geometrically similar models.

### Key Objectives:
1. **Robust B-Rep Parsing:** Convert raw STEP files to PyTorch Geometric (PyG) graphs using `pythonocc-core`.
2. **Unsupervised Learning:** Train a Graph Neural Network (GNN) without labeled data to generate geometric embeddings.
3. **Fast Similarity Search:** Implement a vector database for millisecond-level retrieval.
4. **Interactive UI:** Provide a lightweight web interface for end-users.

---

## 📂 Repository Structure
```text
cad_retrieval_project/
│
├── data/                      # Data storage (Ignored by Git)
│   ├── raw_step/              # Place original .step/.stp files here
│   └── processed_graphs/      # Generated PyG .pt files
│
├── checkpoints/               # Trained model weights and vector indices
│   ├── [TODO] gae_weights.pth 
│   └── [TODO] faiss_index.bin 
│
├── src/                       # Core backend logic
│   ├── __init__.py
│   ├── data_converter.py      # Multiprocessing STEP-to-Graph converter
│   ├── dataset.py             # PyG Dataset loader with ID tracking
│   ├── [TODO] networks.py           # GNN / Encoder architecture
│   ├── [TODO] train_unsupervised.py # Training loop and loss functions
│   └── [TODO] retrieval.py          # FAISS vector database builder
│
├── notebooks/                 # Exploratory data analysis & Ablation studies
│   └── [TODO] ablation_study.ipynb  
│
├── [TODO] app.py              # Web Interface (Streamlit/Gradio)
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Requirements

Ensure you have a working Python environment (Conda is highly recommended for `pythonocc-core` and `PyTorch` compatibility).
```bash
# Example Conda Environment Setup
conda create -n cad_retrieval python=3.12 -y
conda activate cad_retrieval

# Install core engineering and ML libraries
conda install -c conda-forge pythonocc-core
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
pip install torch_geometric tqdm faiss-cpu streamlit
```
*(A complete `requirements.txt` will be provided upon project completion).*

---

## 🚀 Pipeline & Usage

### Step 1: Data Preprocessing (Completed)
Place your raw CAD models inside `data/raw_step/`. The data converter uses parallel processing to parse topological and geometric features (faces, edges, curves, areas) into PyG `.pt` files.
```bash
python src/data_converter.py --input data/raw_step --output data/processed_graphs
```

### Step 2: Unsupervised Model Training (WIP)
Load the processed graphs using `src/dataset.py` and train the graph embedding model.
```bash
# [TODO: Implementation pending]
# python src/train_unsupervised.py --epochs 100 --batch_size 32
```

### Step 3: Build Vector Database (WIP)
Pass all training CAD models through the trained encoder to generate embeddings, and build a local similarity index.
```bash
# [TODO: Implementation pending]
# python src/retrieval.py --build_index
```

### Step 4: Web Application (WIP)
Launch the interactive tool to upload new STEP files and retrieve similar existing models.
```bash
# [TODO: Implementation pending]
# streamlit run app.py
```

---

## 👥 Team
**Software Lab 2026 - Group Project**
* Di Liu 
* Ayse Seray Seker
* Eduardo Dall'Igna

## 🎓 Acknowledgements
This project is developed under the guidance of Dr. Stavros Nousias and the Technical University of Munich (TUM). 
