# process_networkrepo.py

import os
import re
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.data import DataEdgeAttr

# Allowlist PyG DataEdgeAttr so torch.load can unpickle safely under PyTorch ≥2.6
torch.serialization.add_safe_globals([DataEdgeAttr])

class NetworkRepoDataset(InMemoryDataset):
    def __init__(self, root, name="IMDB-BINARY", transform=None, pre_transform=None):
        """
        InMemoryDataset for Network-Repository format graphs.
        Expects:
          root/raw/IMDB-BINARY.edges
          root/raw/IMDB-BINARY.graph_idx
          root/raw/IMDB-BINARY.graph_labels
        Produces:
          root/processed/data.pt
        """
        self.name = name
        super(NetworkRepoDataset, self).__init__(root, transform, pre_transform)
        # Force full pickle loading for PyG Data objects
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [
            f"{self.name}.edges",
            f"{self.name}.graph_idx",
            f"{self.name}.graph_labels",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # We assume raw files are already placed in raw/
        pass

    def process(self):
        # 1) Read graph_idx (one line per node, graph ID)
        idx_path = os.path.join(self.raw_dir, f"{self.name}.graph_idx")
        with open(idx_path, 'r') as f:
            graph_idx = [int(line.strip()) - 1 for line in f]  # convert to 0-based

        # 2) Read graph_labels (one line per graph)
        lab_path = os.path.join(self.raw_dir, f"{self.name}.graph_labels")
        with open(lab_path, 'r') as f:
            graph_labels = [int(line.strip()) for line in f]
        # Zero-base labels if they start from 1
        min_lab = min(graph_labels)
        if min_lab != 0:
            graph_labels = [l - min_lab for l in graph_labels]
        num_graphs = len(graph_labels)

        # 3) Read edges (comma- or whitespace-separated)
        edge_path = os.path.join(self.raw_dir, f"{self.name}.edges")
        src, dst = [], []
        splitter = re.compile(r"[,\s]+")
        with open(edge_path, 'r') as f:
            for line in f:
                parts = splitter.split(line.strip())
                if len(parts) >= 2:
                    u, v = int(parts[0]) - 1, int(parts[1]) - 1
                    src.append(u)
                    dst.append(v)

        # 4) Group edges by graph
        from collections import defaultdict
        edges_per_graph = defaultdict(list)
        for u, v in zip(src, dst):
            g = graph_idx[u]
            if graph_idx[v] == g:
                edges_per_graph[g].append((u, v))

        # 5) Collect node indices per graph
        node_sets = defaultdict(list)
        for node_id, g_id in enumerate(graph_idx):
            node_sets[g_id].append(node_id)

        # 6) Build Data objects
        data_list = []
        for g in range(num_graphs):
            nodes = node_sets[g]
            mapping = {n: i for i, n in enumerate(nodes)}
            eg = edges_per_graph.get(g, [])
            if eg:
                edge_index = torch.tensor([
                    [mapping[u] for u, _ in eg],
                    [mapping[v] for _, v in eg],
                ], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # No node attributes in this dataset → use constant 1-dim features
            x = torch.ones((len(nodes), 1), dtype=torch.float)
            y = torch.tensor([graph_labels[g]], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        # 7) Collate and save
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✓ Processed {len(data_list)} graphs → {self.processed_paths[0]}")

if __name__ == "__main__":
    root = "data/IMDB-BINARY"
    # Ensure directories exist
    os.makedirs(os.path.join(root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(root, "processed"), exist_ok=True)
    # Run processing
    ds = NetworkRepoDataset(root=root, name="IMDB-BINARY")
