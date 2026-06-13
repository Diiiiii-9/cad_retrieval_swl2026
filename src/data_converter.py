"""
CAD to Graph Converter for Unsupervised Learning
================================================
@Author: Di Liu
@Date: 2026-05-01
@Description: 
- Reads 3D CAD models (STEP format), extracts B-Rep topological and geometric features using pure PythonOCC, and converts them into Attributed Adjacency Graphs (AAG). 
- Outputs PyTorch Geometric (PyG) .pt files. Designed with multiprocessing for fast, large-scale dataset preprocessing.
"""

import os
import argparse
import pathlib
import logging
import traceback
import multiprocessing
from functools import partial

import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ==========================================
# pythonocc-core imports
# ==========================================
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape, 
    TopTools_ListIteratorOfListOfShape,
    TopTools_IndexedMapOfShape  
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, 
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola,
    GeomAbs_BezierCurve, GeomAbs_BSplineCurve
)

# Suppress OCC's internal C++ warnings to keep the progress bar clean
logging.getLogger().setLevel(logging.ERROR)

# ==========================================
# Categorical Mappings for Geometry Types
# ==========================================
SURFACE_TYPES = {
    GeomAbs_Plane: 0.0, GeomAbs_Cylinder: 1.0, GeomAbs_Cone: 2.0, GeomAbs_Sphere: 3.0, 
    GeomAbs_Torus: 4.0, GeomAbs_BezierSurface: 5.0, GeomAbs_BSplineSurface: 6.0, 
    GeomAbs_SurfaceOfRevolution: 7.0, GeomAbs_SurfaceOfExtrusion: 8.0
}
DEFAULT_SURF_TYPE = 9.0  # 'Other'

CURVE_TYPES = {
    GeomAbs_Line: 0.0, GeomAbs_Circle: 1.0, GeomAbs_Ellipse: 2.0, 
    GeomAbs_Hyperbola: 3.0, GeomAbs_Parabola: 4.0, GeomAbs_BezierCurve: 5.0, 
    GeomAbs_BSplineCurve: 6.0
}
DEFAULT_CURVE_TYPE = 7.0 # 'Other'

# ==========================================
# Feature Extraction Functions
# ==========================================
def extract_face_features(face):
    """Extract 1D scalar features for a B-Rep Face (Node)."""
    try:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
    except Exception:
        area = 0.0
        
    try:
        surf_adaptor = BRepAdaptor_Surface(face)
        geom_type = surf_adaptor.GetType()
        surf_idx = SURFACE_TYPES.get(geom_type, DEFAULT_SURF_TYPE)
        closed_u = 1.0 if surf_adaptor.IsUClosed() else 0.0
        closed_v = 1.0 if surf_adaptor.IsVClosed() else 0.0
    except Exception:
        surf_idx = DEFAULT_SURF_TYPE
        closed_u, closed_v = 0.0, 0.0

    return [area, surf_idx, closed_u, closed_v]

def extract_edge_features(edge):
    """Extract 1D scalar features for a B-Rep Edge (Edge)."""
    try:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = props.Mass()
    except Exception:
        length = 0.0
        
    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        geom_type = curve_adaptor.GetType()
        curve_idx = CURVE_TYPES.get(geom_type, DEFAULT_CURVE_TYPE)
        closed = 1.0 if curve_adaptor.IsClosed() else 0.0
    except Exception:
        curve_idx = DEFAULT_CURVE_TYPE
        closed = 0.0

    return [length, curve_idx, closed]

# ==========================================
# Graph Construction
# ==========================================
def build_pyg_graph(shape):
    """Builds a PyTorch Geometric Data object directly from an OCC TopoDS_Shape."""
    if shape is None or shape.IsNull():
        return None

    # 1. Map Faces (Nodes)
    face_map = TopTools_IndexedMapOfShape()
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    num_faces = 0
    while face_explorer.More():
        idx = face_map.Add(face_explorer.Current())
        if idx > num_faces:
            num_faces = idx
        face_explorer.Next()
        
    if num_faces == 0:
        return None

    # Node Features (x)
    node_feats = []
    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        node_feats.append(extract_face_features(face))
        
    x = torch.tensor(node_feats, dtype=torch.float)

    # 2. Map Edges (Topology)
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    edge_map = TopTools_IndexedMapOfShape()
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    num_edges = 0
    while edge_explorer.More():
        idx = edge_map.Add(edge_explorer.Current())
        if idx > num_edges:
            num_edges = idx
        edge_explorer.Next()

    src_nodes, dst_nodes, edge_feats = [], [], []

    # 3. Iterate edges to build adjacency
    for i in range(1, num_edges + 1):
        edge = topods.Edge(edge_map.FindKey(i))
        try:
            adjacent_faces_list = edge_face_map.FindFromKey(edge)
        except Exception:
            continue
        
        adj_faces = []
        iterator = TopTools_ListIteratorOfListOfShape(adjacent_faces_list)
        while iterator.More():
            adj_faces.append(iterator.Value())  
            iterator.Next()
            
        if len(adj_faces) == 2:
            idx_0 = face_map.FindIndex(adj_faces[0]) - 1
            idx_1 = face_map.FindIndex(adj_faces[1]) - 1
            
            if idx_0 >= 0 and idx_1 >= 0:
                feat = extract_edge_features(edge)
                # Bidirectional edges for undirected graph
                src_nodes.extend([idx_0, idx_1])
                dst_nodes.extend([idx_1, idx_0])
                edge_feats.extend([feat, feat])

    # 4. Construct Graph Tensors
    if len(src_nodes) > 0:
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    # No 'y' (labels) included, as this is for unsupervised learning
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ==========================================
# Multiprocessing Worker
# ==========================================
def worker(file_path, output_dir):
    """
    Worker function to process a single STEP file.
    Returns: (bool success, str filename, str error_msg)
    """
    save_path = output_dir / f"{file_path.stem}.pt"
    
    # Skip if already processed (allows resuming interrupted jobs)
    if save_path.exists():
        return True, file_path.name, "Already exists"

    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(file_path))
        
        if status != 1:
            return False, file_path.name, "Read failed"
            
        reader.TransferRoots()
        shape = reader.OneShape()
        
        pyg_data = build_pyg_graph(shape)
        if pyg_data is None:
            return False, file_path.name, "Empty graph"
            
        torch.save(pyg_data, save_path)
        return True, file_path.name, ""
        
    except Exception as e:
        return False, file_path.name, str(e).split('\n')[0]

# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Parallel STEP to PyG Graph Converter")
    parser.add_argument("--input", type=str, default="../data/raw_step", help="Input folder with STEP files")
    parser.add_argument("--output", type=str, default="../data/processed_graphs", help="Output folder for .pt files")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() // 2), 
                        help="Number of CPU cores to use (default: half of available)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return
        
    output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.step")) + list(input_path.glob("*.stp"))
    
    if not step_files:
        print(f"No STEP files found in {input_path}")
        return

    print(f"Found {len(step_files)} CAD models. Starting parallel conversion using {args.cores} cores...")

    # Set up the worker with a fixed output directory using partial
    worker_func = partial(worker, output_dir=output_path)
    
    success_count = 0
    fail_count = 0

    # Execute with Multiprocessing Pool and TQDM Progress Bar
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, step_files), 
                            total=len(step_files), 
                            desc="Converting CAD to Graphs"))

    # Summary Statistics
    for success, name, msg in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            # Optional: Print failure reasons to a log file instead of cluttering console
            # print(f"Failed {name}: {msg}")

    print("\n" + "="*40)
    print("🎉 Conversion Complete!")
    print(f"✅ Successfully Processed/Skipped: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print("="*40)

if __name__ == "__main__":
    # Note: Requires `pip install tqdm torch_geometric` to run
    main()