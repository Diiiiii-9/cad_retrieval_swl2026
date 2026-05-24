"""
CAD Model Retrieval - Interactive Web Interface
===============================================
A Streamlit frontend that allows users to upload a 3D CAD model (.step)
and instantly retrieves the top geometrically similar models from the database
using the trained GNN Encoder and FAISS vector search.
"""

import os
import sys
import tempfile
import pathlib
import pickle
import logging
import numpy as np
import pandas as pd
import streamlit as st

import torch
from torch_geometric.data import Batch
import faiss

import plotly.graph_objects as go
import trimesh
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer

# ==========================================
# Path Configuration (Allow importing from src/)
# ==========================================
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

ORIGINAL_STEP_DIR = ROOT_DIR / "data" / "raw_step" / "mfcad"

# Import custom backend modules
from src.data_converter import build_pyg_graph
from src.networks import CADGraphEncoder

# OCC imports for real-time file reading
from OCC.Core.STEPControl import STEPControl_Reader

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="CAD Retrieval AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Caching Heavy Components
# ==========================================
@st.cache_resource
def load_backend_system(node_dim=4, edge_dim=3, hidden_dim=128, out_dim=64):
    """
    Loads the trained PyTorch model, FAISS index, and metadata into memory.
    @st.cache_resource prevents reloading these heavy files on every UI interaction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = ROOT_DIR / "checkpoints"
    
    # 1. Load FAISS Index & Metadata
    index_file = weights_path / "faiss_index.bin"
    meta_file = weights_path / "metadata.pkl"
    
    if not index_file.exists() or not meta_file.exists():
        return None, None, None, f"Database missing! Please run 'python src/retrieval.py --build' first."
        
    index = faiss.read_index(str(index_file))
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)
        
    # 2. Load PyTorch Encoder
    model_file = weights_path / "final_encoder_weights.pth"
    if not model_file.exists():
        model_file = weights_path / "best_contrastive_model.pth"
        if not model_file.exists():
            return None, None, None, "Model weights missing! Please run training first."
            
    encoder = CADGraphEncoder(
        node_in_dim=node_dim, 
        edge_in_dim=edge_dim, 
        hidden_dim=hidden_dim, 
        out_dim=out_dim,
        num_layers=3
    ).to(device)
    
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    # Handle both raw state_dicts and nested checkpoint dictionaries
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
        
    encoder.eval()
    
    return encoder, index, metadata, None

def render_shape_to_plotly(shape):
    """
    Takes an OCC TopoDS_Shape, tessellates it into a mesh, 
    and returns a 3D Plotly Figure for Streamlit rendering.
    """
    mesh = BRepMesh_IncrementalMesh(shape, 0.1) 
    mesh.Perform()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        stl_filepath = tmp.name

    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, stl_filepath)

    mesh_data = trimesh.load(stl_filepath)
    os.unlink(stl_filepath)

    fig = go.Figure(data=[go.Mesh3d(
        x=mesh_data.vertices[:, 0],
        y=mesh_data.vertices[:, 1],
        z=mesh_data.vertices[:, 2],
        i=mesh_data.faces[:, 0],
        j=mesh_data.faces[:, 1],
        k=mesh_data.faces[:, 2],
        color='lightblue', 
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.5, specular=0.2)
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' 
        )
    )
    return fig

# ==========================================
# Real-time Processing Helper
# ==========================================
def process_uploaded_cad(uploaded_file):
    """Saves the uploaded file to a temporary location and converts it to a PyG Graph."""
    try:
        # Create a temporary file to allow OCC to read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name

        # Read using OCC
        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp_filepath)
        
        if status != 1:
            os.unlink(tmp_filepath)
            return None, "Failed to parse STEP file format."
            
        reader.TransferRoots()
        shape = reader.OneShape()
        
        # Convert to Graph
        graph_data = build_pyg_graph(shape)
        
        # Cleanup temp file
        os.unlink(tmp_filepath)
        
        if graph_data is None:
            return None, "Successfully read file, but no valid B-Rep faces were found."
            
        return graph_data, shape, None
        
    except Exception as e:
        return None, f"Critical error during processing: {str(e)}"

# ==========================================
# Main UI Layout
# ==========================================
def main():
    st.title("3D CAD Model Retrieval System")
    st.markdown("Upload a **STEP** file to find geometrically similar models from the database using Unsupervised Graph Neural Networks.")
    
    # -- Sidebar Settings --
    st.sidebar.header("Retrieval Settings")
    top_k = st.sidebar.slider("Number of results to show (Top-K):", min_value=1, max_value=20, value=5)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**System Info:**\n\n"
        "Engine: GINE Contrastive\n\n"
        "Similarity: Cosine (FAISS)\n\n"
        "Input Format: B-Rep (Edges/Faces)"
    )

    # -- Load Backend --
    with st.spinner("Loading AI Models and Vector Database..."):
        encoder, index, metadata, error_msg = load_backend_system()
        
    if error_msg:
        st.error(error_msg)
        st.stop()

    # -- Main Interaction Area --
    uploaded_file = st.file_uploader("Upload a .step or .stp file", type=['step', 'stp'])

    if uploaded_file is not None:
        st.markdown("---")
        
        device = next(encoder.parameters()).device

        with st.spinner("Converting CAD to Graph and extracting features..."):
            graph_data, raw_shape, err = process_uploaded_cad(uploaded_file)
            
            if err:
                st.error(err)
                st.stop()
                
            # 2. Prepare Batch for Model
            batch = Batch.from_data_list([graph_data]).to(device)
            edge_index = batch.edge_index if batch.edge_index is not None else torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = batch.edge_attr if batch.edge_attr is not None else torch.empty((0, 3), dtype=torch.float, device=device)
            
            # 3. Generate Embedding
            with torch.no_grad():
                embedding = encoder(batch.x, edge_index, edge_attr, batch.batch)
                query_vector = embedding.cpu().numpy().astype('float32')
                
            # 4. Perform FAISS Search
            distances, indices = index.search(query_vector, top_k)
            
        # -- Display Results --
        results_data = []
        valid_indices = [] 
        
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1: 
                continue

            sim_score = float(dist) * 100
            matched_filename = metadata[idx]
            valid_indices.append(idx)
            
            results_data.append({
                "Rank": rank + 1,
                "Retrieved CAD Model": f"{matched_filename}.step",
                "Confidence": f"{sim_score:.2f}%"
            })
            
        st.success(f"Successfully retrieved top {len(results_data)} matches from a database of {index.ntotal} models in milliseconds!")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Your Query Model")
            fig_query = render_shape_to_plotly(raw_shape)
            st.plotly_chart(fig_query, use_container_width=True, height=450)
            
        with col2:
            st.subheader("Top Matches List")
            df = pd.DataFrame(results_data)
            st.dataframe(
                df, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Confidence": st.column_config.TextColumn(width="medium")
                }
            )
        
        st.markdown("---")
        st.subheader("Inspect Matches in 3D")
        
        if len(valid_indices) > 0:

            tab_titles = [f"Rank {i+1}" for i in range(len(valid_indices))]
            tabs = st.tabs(tab_titles)
            
            for i, tab in enumerate(tabs):
                with tab:
                    matched_name = metadata[valid_indices[i]]

                    step_file_path = ORIGINAL_STEP_DIR / f"{matched_name}.step" 
                    
                    st.markdown(f"**Model Name:** `{matched_name}.step` | **Similarity:** {results_data[i]['Confidence']}")
                    
                    if step_file_path.exists():
                        with st.spinner(f"Rendering 3D mesh for Rank {i+1}..."):
                            try:

                                reader = STEPControl_Reader()
                                status = reader.ReadFile(str(step_file_path))
                                
                                if status == 1:
                                    reader.TransferRoots()
                                    matched_shape = reader.OneShape()
                                    
                                    fig_match = render_shape_to_plotly(matched_shape)
                                    st.plotly_chart(fig_match, use_container_width=True, height=500)
                                else:
                                    st.error(f"OCC failed to parse file: {step_file_path.name}")
                            except Exception as render_err:
                                st.error(f"Render error: {render_err}")
                    else:
                        st.warning(f"Could not visualize: Original .step file not found at `{step_file_path}`")

if __name__ == "__main__":
    main()