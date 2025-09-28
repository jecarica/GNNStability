# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, topo=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.lin(g)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.lin  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, topo=None):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.lin(g)

class StabilityGNN(nn.Module):
    """
    StabilityGNN that consumes a precomputed topo vector (data.topo).
    """
    def __init__(self, in_feats, hidden_dim, num_classes, topo_dim):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # final linear on [g_emb || topo_emb]
        self.lin   = nn.Linear(hidden_dim + topo_dim, num_classes)

    def forward(self, x, edge_index, batch, topo):
        # 1) Standard 2-layer GCN
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # 2) Graph readout
        g = global_mean_pool(x, batch)            # (batch_size, hidden_dim)

        # 3) Ensure topo has shape (batch_size, topo_dim)
        if topo.dim() == 1:
            # single vector → expand to one per graph in batch
            topo = topo.unsqueeze(0).expand(g.size(0), -1)
        elif topo.size(0) != g.size(0):
            # maybe flattened: reshape if total length matches
            topo = topo.view(g.size(0), -1)

        # 4) Fuse and classify
        gtopo = torch.cat([g, topo], dim=1)       # (batch_size, hidden_dim + topo_dim)
        return self.lin(gtopo)
