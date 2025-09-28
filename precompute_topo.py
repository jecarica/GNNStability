# precompute_topo.py

import os
import torch
import numpy as np
from process_networkrepo import NetworkRepoDataset
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch

from gudhi import RipsComplex
from gudhi.representations import PersistenceImage

# Hyperparameters for persistence image
r0, r1 = 0.5, 1.0
pers_img = PersistenceImage(bandwidth=0.1, resolution=[20, 20])

def compute_persistence_image(edge_index, num_nodes):
    # Build adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=float)
    ei = edge_index.cpu().numpy()
    adj[ei[0], ei[1]] = 1.0
    adj[ei[1], ei[0]] = 1.0

    # Floyd–Warshall all‑pairs shortest paths
    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0.0)
    nbrs = np.where(adj > 0)
    dist[nbrs] = 1.0
    for k in range(num_nodes):
        dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

    # Build Rips at r1 and compute H1 intervals
    rips1 = RipsComplex(distance_matrix=dist, max_edge_length=r1)
    st1 = rips1.create_simplex_tree(max_dimension=2)
    st1.persistence()
    intervals = st1.persistence_intervals_in_dimension(1)

    # Clip to [r0, r1]
    rel = []
    for b, d in intervals:
        if d <= r0 or b >= r1: continue
        rel.append([max(b, r0), min(d, r1)])
    if len(rel) == 0:
        vec = np.zeros(pers_img.resolution[0] * pers_img.resolution[1])
    else:
        vec = pers_img.fit_transform([np.array(rel)])[0]
    return torch.tensor(vec, dtype=torch.float)

if __name__ == "__main__":
    root = "data/IMDB-BINARY"
    ds = NetworkRepoDataset(root=root, name="IMDB-BINARY")
    # extract raw Data objects
    data_list = [ds.get(i) for i in range(len(ds))]

    # compute and attach
    for data in data_list:
        topo = compute_persistence_image(data.edge_index, data.num_nodes)
        data.topo = topo

    # collate & overwrite processed file
    data_all, slices = ds.collate(data_list)
    os.makedirs(ds.processed_dir, exist_ok=True)
    torch.save((data_all, slices), ds.processed_paths[0])
    print(f"✓ Wrote {len(data_list)} graphs with .topo → {ds.processed_paths[0]}")
