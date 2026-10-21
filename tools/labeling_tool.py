import streamlit as st
import trimesh
import plotly.graph_objects as go
import os
import json
import random

# ================= Configuration =================
STL_DIR = "./data/raw_step/fusion360_stl/"
LABEL_FILE = "./data/fusion360_labels.json"

# Default CAD categories to start with
DEFAULT_TAGS = [
    "Bracket", "L-Bracket", "Flange", "Plate_with_holes", 
    "Gear", "Shaft", "Pulley", "Bearing",
    "Housing", "Pipe_Fitting", "Fastener_Bolt", "Nut_or_Washer"
]

# ================= Initialize Session State =================
if 'stl_files' not in st.session_state:
    # Fetch all STL files and shuffle them for random sampling
    files = [f for f in os.listdir(STL_DIR) if f.endswith('.stl')]
    random.shuffle(files)
    st.session_state.stl_files = files
    st.session_state.current_index = 0

if 'labels' not in st.session_state:
    # Load existing labels to resume progress
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r') as f:
            st.session_state.labels = json.load(f)
    else:
        st.session_state.labels = {}

if 'available_tags' not in st.session_state:
    # Merge default tags with any existing tags from the loaded JSON
    existing_tags = set(st.session_state.labels.values())
    merged_tags = list(set(DEFAULT_TAGS) | existing_tags)
    st.session_state.available_tags = sorted(merged_tags)

# ================= Helper Functions =================
def save_labels():
    with open(LABEL_FILE, 'w') as f:
        json.dump(st.session_state.labels, f, indent=4)

def next_part():
    st.session_state.current_index += 1

def undo_part():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
    
def assign_label(filename, label):
    st.session_state.labels[filename] = label
    if label not in st.session_state.available_tags:
        st.session_state.available_tags.append(label)
        st.session_state.available_tags.sort()
    save_labels()
    next_part()

# ================= UI Layout =================
st.set_page_config(layout="wide", page_title="Fusion360 Labeling Tool")
st.title("🛠️ Fusion360 Manual Evaluation Tool")

if st.session_state.current_index >= len(st.session_state.stl_files):
    st.success("🎉 You have reviewed all files!")
    st.stop()

current_file = st.session_state.stl_files[st.session_state.current_index]
st.subheader(f"Current Part: {current_file} (Index: {st.session_state.current_index})")

# Progress tracking
labeled_count = len(st.session_state.labels)
st.progress(labeled_count / max(100, labeled_count)) 
st.write(f"Total labeled so far: **{labeled_count}**")

col1, col2 = st.columns([2, 1])

with col1:
    # 3D Model Rendering with Enhanced Lighting & Depth Perception
    file_path = os.path.join(STL_DIR, current_file)
    try:
        mesh = trimesh.load(file_path)
        
        # Color intensity based on Z-axis to highlight holes and slots
        z_heights = mesh.vertices[:, 2] 
        
        fig = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            intensity=z_heights,
            colorscale='Viridis', # Creates a nice gradient to show depth
            showscale=False,
            flatshading=True,     # Highlights sharp edges and flat faces
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5)
        )])
        fig.update_layout(
            scene=dict(
                aspectmode='data', 
                bgcolor='rgb(240, 240, 240)' # Light gray background for contrast
            ), 
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load mesh: {e}")
        if st.button("⏭️ Skip (Error)", key="skip_err"):
            next_part()
            st.rerun()

with col2:
    st.write("### 🏷️ Assign a Category")
    
    # 1. Existing Categories (Buttons)
    st.write("**Quick Select:**")
    
    # Create a grid layout for buttons so they don't take up too much vertical space
    btn_cols = st.columns(2)
    for idx, tag in enumerate(st.session_state.available_tags):
        with btn_cols[idx % 2]:
            if st.button(tag, key=f"btn_{tag}", use_container_width=True):
                assign_label(current_file, tag)
                st.rerun()
            
    st.markdown("---")
    
    # 2. Add New Category
    new_tag = st.text_input("Or create a new category (e.g., Spring, Rotor):")
    if st.button("➕ Add & Assign", type="primary", use_container_width=True):
        if new_tag:
            # Capitalize and format the new tag
            formatted_tag = new_tag.strip().replace(" ", "_").title()
            assign_label(current_file, formatted_tag)
            st.rerun()
        else:
            st.warning("Please enter a category name.")
            
    st.markdown("---")
    
    # 3. Utility Buttons (Skip & Undo)
    util_col1, util_col2 = st.columns(2)
    with util_col1:
        if st.button("⏭️ Skip (Unclear)", use_container_width=True):
            next_part()
            st.rerun()
    with util_col2:
        if st.button("↩️ Undo Last", use_container_width=True):
            undo_part()
            st.rerun()