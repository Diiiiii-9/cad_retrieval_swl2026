"""
Graph Neural Network Architectures for CAD Retrieval
===================================================
@Author: Di Liu
@Date: 2026-05-24
@Description:
Defines the deep learning models for unsupervised graph embedding.
Uses GINEConv (Graph Isomorphism Network with Edge Attributes) to fully 
leverage the B-Rep topology, mapping both Face (node) and Edge features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, MLP
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

class CADGraphEncoder(nn.Module):
    """
    The core Graph Neural Network. 
    Compresses a variable-sized CAD graph into a fixed-size vector (embedding).
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=256, out_dim=128, num_layers=3, dropout=0.2):
        """
        Args:
            node_in_dim (int): Number of input features per node (e.g., 4).
            edge_in_dim (int): Number of input features per edge (e.g., 3).
            hidden_dim (int): Dimensionality of hidden GNN layers.
            out_dim (int): Final output dimensionality of the graph embedding.
            num_layers (int): Number of GINE message passing layers.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.dropout = dropout
        
        # 1. Initial Feature Embedders
        # Maps raw features (areas, types, lengths) into a high-dimensional continuous space
        self.node_emb = Linear(node_in_dim, hidden_dim)
        self.edge_emb = Linear(edge_in_dim, hidden_dim)
        
        # 2. Message Passing Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # GINE requires an MLP to process the aggregated neighborhood messages
            nn_update = MLP([hidden_dim, hidden_dim * 2, hidden_dim], norm="batch_norm", act="relu")
            
            # GINEConv explicitly adds edge_attr to node features during aggregation
            # We set edge_dim=hidden_dim because we pre-embed the edges
            self.convs.append(GINEConv(nn_update, edge_dim=hidden_dim))
            
        # 3. Final Projection
        self.post_mp = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [num_nodes_in_batch, node_in_dim]
            edge_index: Graph connectivity [2, num_edges_in_batch]
            edge_attr: Edge features [num_edges_in_batch, edge_in_dim]
            batch: Batch vector indicating which node belongs to which graph
        """
        # Embed initial raw features
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        # Message Passing (Information propagates through the CAD topology)
        for conv in self.convs:
            # GINEConv takes x, edge_index, and edge_attr
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Global Pooling (Readout)
        # Aggregates all node vectors in a graph into a single vector representing the entire CAD part.
        # global_mean_pool computes the average node representation.
        graph_embedding = global_mean_pool(x, batch)
        
        # Final linear projection
        out = self.post_mp(graph_embedding)
        
        # L2 Normalization (Crucial for cosine similarity during retrieval)
        out = F.normalize(out, p=2, dim=1)
        
        return out


class CADContrastiveModel(nn.Module):
    """
    Wrapper model for Unsupervised Contrastive Learning (e.g., SimCLR).
    It attaches a Non-Linear Projection Head to the Graph Encoder.
    """
    def __init__(self, encoder, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        
        # Projection Head: Maps the encoder's representation to the space where 
        # the contrastive loss (InfoNCE) is applied. This improves representation quality.
        self.proj_head = MLP([hidden_dim, hidden_dim, proj_dim], norm="batch_norm", act="relu")

    def forward(self, data):
        """
        During inference/retrieval, we ONLY use the encoder embedding.
        The projection head is discarded after training.
        """
        # Robustness: Handle graphs with no edges
        if data.edge_index is None or data.edge_index.numel() == 0:
            # Create dummy edge_index and edge_attr on the same device
            edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)
            edge_attr = torch.empty((0, self.encoder.edge_emb.in_features), dtype=torch.float, device=data.x.device)
        else:
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            
        embedding = self.encoder(data.x, edge_index, edge_attr, data.batch)
        return embedding

    def forward_cl(self, data1, data2):
        """
        Used ONLY during training.
        Takes two augmented views of the same CAD model (data1, data2),
        extracts embeddings, and passes them through the projection head.
        """
        emb1 = self.forward(data1)
        emb2 = self.forward(data2)
        
        z1 = self.proj_head(emb1)
        z2 = self.proj_head(emb2)
        
        # L2 Normalize the projections
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1, z2

# ==========================================
# Quick Test Block (Executes only if run directly)
# ==========================================
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    import torch

    print("Testing GINE-based CAD Encoder...")
    
    # 1. Create a dummy graph simulating a CAD model
    num_nodes = 10
    num_edges = 14
    node_in_dim = 4  # [area, surf_idx, closed_u, closed_v]
    edge_in_dim = 3  # [length, curve_idx, closed]
    
    x = torch.randn((num_nodes, node_in_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn((num_edges, edge_in_dim))
    
    # Simulate a batch of 2 graphs
    data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data1, data2])

    # 2. Initialize Model
    encoder = CADGraphEncoder(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, out_dim=64)
    model = CADContrastiveModel(encoder=encoder, hidden_dim=64, proj_dim=32)

    # 3. Test Inference
    embeddings = model(batch)
    print(f"Graph Embedding Output Shape: {embeddings.shape}") # Should be [2, 64]
    
    # 4. Test Contrastive Training Pass
    z1, z2 = model.forward_cl(batch, batch)
    print(f"Projection Head Output Shape: {z1.shape}") # Should be [2, 32]
    print("Network tests passed successfully!")