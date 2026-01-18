# hk_stability_tudataset_with_perslay.py
# FIXED VERSION 2

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gudhi as gd

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse, dropout_edge
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import TUDataset

from torch.nn.utils.parametrizations import spectral_norm

# Import necessary functions from Perslay code
from scipy.sparse import csgraph
from scipy.linalg import eigh


# -------------------------
# Perslay Helper Functions
# -------------------------
def hks_signature(eigenvectors, eigenvals, time):
    """Heat Kernel Signature computation"""
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def apply_graph_extended_persistence(A, filtration_val):
    """Apply extended persistence to a graph"""
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()

    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)

    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)

    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])

    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()

    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]

    # Process diagrams
    dgmOrd0 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]])
                         for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0, 2])
    dgmRel1 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]])
                         for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0, 2])
    dgmExt0 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]])
                         for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0, 2])
    dgmExt1 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]])
                         for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0, 2])

    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1


def get_parameters(dataset):
    """Get dataset-specific parameters from Perslay code"""
    if dataset == "MUTAG" or dataset == "PROTEINS":
        dataset_parameters = {"data_type": "graph", "filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "COX2" or dataset == "ENZYMES" or dataset == "DHFR" or dataset == "NCI1" or dataset == "NCI109" or dataset == "IMDB-BINARY" or dataset == "IMDB-MULTI" or dataset == "REDDIT-BINARY":
        dataset_parameters = {"data_type": "graph",
                              "filt_names": ["Ord0_0.1-hks", "Rel1_0.1-hks", "Ext0_0.1-hks", "Ext1_0.1-hks",
                                             "Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "ORBIT5K" or dataset == "ORBIT100K":
        dataset_parameters = {"data_type": "orbit", "filt_names": ["Alpha0", "Alpha1"]}
    else:
        # Default parameters for other TUDatasets
        dataset_parameters = {"data_type": "graph",
                              "filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    return dataset_parameters


# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(dataset, train_ratio=0.8, seed=17):
    set_seed(seed)
    labels = [int(data.y.item()) for data in dataset]

    from collections import Counter
    class_counts = Counter(labels)

    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_idx = []
    test_idx = []

    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_idx.extend(indices[:split_idx])
        test_idx.extend(indices[split_idx:])

    train_idx.sort()
    test_idx.sort()

    train_ds = dataset.__class__(dataset.root, name=dataset.name)
    test_ds = dataset.__class__(dataset.root, name=dataset.name)

    train_list = [dataset[i] for i in train_idx]
    test_list = [dataset[i] for i in test_idx]

    train_ds.data, train_ds.slices = dataset.collate(train_list)
    test_ds.data, test_ds.slices = dataset.collate(test_list)

    trc = Counter([int(d.y.item()) for d in train_list])
    tec = Counter([int(d.y.item()) for d in test_list])
    print(f"Stratified split: train per class = {dict(trc)} test per class = {dict(tec)}")
    return train_ds, test_ds


# ---------------------------------------------
# 1) Dataset wrapper with Perslay diagrams
# ---------------------------------------------
class PerslayGraphDataset(TUDataset):
    """
    Builds per-graph node features and Perslay-style
    extended persistence diagrams for multiple filtrations.
    """

    def __init__(self, root, name, max_deg=50, p_edge_pert=0.05, max_points_per_diagram=50):
        super().__init__(root, name)
        self.max_deg = int(max_deg)
        self.p_edge_pert = float(p_edge_pert)
        self.max_points = int(max_points_per_diagram)

        # Get dataset-specific parameters
        dataset_params = get_parameters(name)
        self.filt_names = dataset_params["filt_names"]

        # Extract HKS times from filter names
        self.hks_times = []
        for filt in self.filt_names:
            if "hks" in filt:
                # Extract time value from names like "Ord0_10.0-hks"
                time_str = filt.split("_")[1].split("-")[0]
                try:
                    time_val = float(time_str)
                    if time_val not in self.hks_times:
                        self.hks_times.append(time_val)
                except:
                    pass

        self.num_filtrations = len(self.filt_names)
        self._attach_all_features()

    def _perturb_edge_index(self, data, p=0.05):
        A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        M = (torch.rand_like(A) < p).triu(1)
        A2 = A.clone()
        A2[M] = 1 - A2[M]
        A2.fill_diagonal_(0)
        A2 = torch.minimum(A2, A2.t())
        return dense_to_sparse(A2)[0]

    def _compute_extended_persistence_diagrams(self, A_np, hks_time):
        """Compute extended persistence diagrams for a given HKS time"""
        num_vertices = A_np.shape[0]

        # Compute Laplacian and eigenvalues
        L = csgraph.laplacian(A_np, normed=True)
        egvals, egvectors = eigh(L)

        # Compute HKS filtration
        filtration_val = hks_signature(egvectors, egvals, time=hks_time)

        # Get extended persistence diagrams
        dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(
            A_np, filtration_val
        )

        # Normalize diagrams to [0, 1]
        diagrams = {
            f'Ord0_{hks_time}-hks': dgmOrd0,
            f'Ext0_{hks_time}-hks': dgmExt0,
            f'Rel1_{hks_time}-hks': dgmRel1,
            f'Ext1_{hks_time}-hks': dgmExt1
        }

        # Clip diagrams
        for key in diagrams:
            if len(diagrams[key]) > 0:
                diagrams[key] = np.clip(diagrams[key], 0.0, 1.0)

        return diagrams

    def _onehot_degree(self, data):
        deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg = torch.clamp(deg, max=self.max_deg)
        return F.one_hot(deg, num_classes=self.max_deg + 1).float()

    def _pad_diagram(self, diagram, max_points):
        """Pad or truncate diagram to have exactly max_points"""
        if len(diagram) == 0:
            return np.zeros((max_points, 2), dtype=np.float32)

        # Truncate if too many points
        if len(diagram) > max_points:
            diagram = diagram[:max_points]

        # Pad if too few points
        if len(diagram) < max_points:
            padding = np.zeros((max_points - len(diagram), 2), dtype=np.float32)
            diagram = np.vstack([diagram, padding])

        return diagram

    def _attach_all_features(self):
        print(f"Attaching node features + Perslay extended persistence diagrams for {self.name}…")
        start = time.time()
        data_list = []

        for i in range(len(self)):
            d = self.get(i)

            # Node features
            if d.x is None or d.x.size(1) == 0:
                d.x = self._onehot_degree(d)

            # Convert to numpy for persistence computation
            A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes).squeeze(0).numpy()

            # Store all diagrams for this graph as a single tensor
            all_diagrams = []

            # Compute diagrams for each HKS time and diagram type
            for filt_name in self.filt_names:
                if "hks" in filt_name:
                    parts = filt_name.split("_")
                    diag_type = parts[0]
                    hks_time = float(parts[1].split("-")[0])

                    # Compute diagrams
                    diagrams_dict = self._compute_extended_persistence_diagrams(A, hks_time)

                    # Get the specific diagram
                    diag_key = f"{diag_type}_{hks_time}-hks"
                    if diag_key in diagrams_dict:
                        diag = diagrams_dict[diag_key]
                    else:
                        diag = np.array([[0.0, 0.0]])

                    # Pad/truncate diagram
                    diag = self._pad_diagram(diag, self.max_points)
                    all_diagrams.append(diag)

            # Convert to tensor: shape (num_filtrations, max_points, 2)
            diagrams_tensor = torch.tensor(np.stack(all_diagrams), dtype=torch.float)
            d.diagrams = diagrams_tensor

            # Also compute for perturbed graph
            ei_pert = self._perturb_edge_index(d, p=self.p_edge_pert)
            A_pert = to_dense_adj(ei_pert, max_num_nodes=d.num_nodes).squeeze(0).numpy()

            all_diagrams_pert = []
            for filt_name in self.filt_names:
                if "hks" in filt_name:
                    parts = filt_name.split("_")
                    diag_type = parts[0]
                    hks_time = float(parts[1].split("-")[0])

                    diagrams_dict = self._compute_extended_persistence_diagrams(A_pert, hks_time)
                    diag_key = f"{diag_type}_{hks_time}-hks"

                    if diag_key in diagrams_dict:
                        diag = diagrams_dict[diag_key]
                    else:
                        diag = np.array([[0.0, 0.0]])

                    # Pad/truncate diagram
                    diag = self._pad_diagram(diag, self.max_points)
                    all_diagrams_pert.append(diag)

            diagrams_pert_tensor = torch.tensor(np.stack(all_diagrams_pert), dtype=torch.float)
            d.diagrams_pert = diagrams_pert_tensor

            data_list.append(d)
            if (i + 1) % 50 == 0 or (i + 1) == len(self):
                print(f"  • {i + 1}/{len(self)} done")

        # Replace in-memory storage
        self.data, self.slices = self.collate(data_list)
        print(
            f"✅ Attached in {time.time() - start:.1f}s | {self.num_filtrations} filtrations per graph, {self.max_points} max points per diagram")


# --------------------------------------------
# 2) PyTorch Perslay Model Components
# --------------------------------------------
class PerslayPermutationEquivariantLayer(nn.Module):
    """PyTorch implementation of Perslay's PermutationEquivariant layer for NCI1"""

    def __init__(self, layer_sizes=[(25, None), (25, "max")]):
        super().__init__()
        self.layer_sizes = layer_sizes

        # Create layers
        self.layers = nn.ModuleList()
        for i, (size, _) in enumerate(layer_sizes):
            if i == 0:
                self.layers.append(nn.Linear(2, size))
            else:
                self.layers.append(nn.Linear(layer_sizes[i - 1][0], size))

        self.output_dim = layer_sizes[-1][0]

    def forward(self, diagrams):
        """
        diagrams: Tensor of shape (batch_size, max_points, 2)
        Returns: Tensor of shape (batch_size, output_dim)
        """
        batch_size, num_points, _ = diagrams.shape

        # Process through layers
        x = diagrams.reshape(-1, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        # Reshape back and apply max pooling
        x = x.reshape(batch_size, num_points, -1)
        x = torch.max(x, dim=1)[0]  # Max pooling

        return x


class PerslayModel(nn.Module):
    """PyTorch implementation of Perslay  (using PermutationEquivariant layers)"""

    def __init__(self, dataset_name, num_filtrations, max_points, hidden_dim=64, num_classes=2):
        super().__init__()

        self.dataset_name = dataset_name
        self.num_filtrations = num_filtrations
        self.max_points = max_points


        self.perslay_layers = nn.ModuleList([
            PerslayPermutationEquivariantLayer(layer_sizes=[(25, None), (25, "max")])
            for _ in range(num_filtrations)
        ])

        # Each layer outputs 25 features
        perslay_output_dim = num_filtrations * 25

        # Rho network (final classifier)
        self.rho = nn.Sequential(
            nn.Linear(perslay_output_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, diagrams_batch):
        """
        diagrams_batch: Tensor of shape (batch_size, num_filtrations, max_points, 2)
        Returns: Tensor of shape (batch_size, num_classes)
        """
        batch_size = diagrams_batch.shape[0]
        all_features = []

        # Process each filtration through its Perslay layer
        for i in range(self.num_filtrations):
            # Extract diagram for this filtration: (batch_size, max_points, 2)
            diag = diagrams_batch[:, i, :, :]
            features = self.perslay_layers[i](diag)  # (batch_size, 25)
            all_features.append(features)

        # Concatenate all features
        if len(all_features) > 0:
            combined = torch.cat(all_features, dim=1)  # (batch_size, num_filtrations * 25)
        else:
            combined = torch.zeros(batch_size, self.num_filtrations * 25, device=diagrams_batch.device)

        # Pass through rho network
        output = self.rho(combined)
        return output


# --------------------------------------------
# 3) Combined Model: GIN + Perslay
# --------------------------------------------
class PerslayGIN_HK(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dataset_name, num_filtrations, max_points):
        super().__init__()
        self.num_filtrations = num_filtrations
        self.hidden = hidden_dim

        # Structural pathway (GIN)
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ))

        # Perslay pathway
        self.perslay = PerslayModel(
            dataset_name=dataset_name,
            num_filtrations=num_filtrations,
            max_points=max_points,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        # Classifier combining both pathways
        self.fc1 = spectral_norm(nn.Linear(hidden_dim + num_classes, 128))
        self.fc2 = spectral_norm(nn.Linear(128, num_classes))
        self.dropout = nn.Dropout(0.5)

        # Orthogonal init
        for m in [self.fc1, self.fc2]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x, edge_index, batch, diagrams_batch):
        # Structural pathway
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        g_struct = global_add_pool(h, batch)  # (B, hidden)

        # Perslay pathway
        g_perslay = self.perslay(diagrams_batch)  # (B, num_classes)

        # Combine pathways
        g = torch.cat([g_struct, g_perslay], dim=1)  # (B, hidden + num_classes)
        g = F.elu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g)


# -------------------------------------------------------
# 4) Stability loss
# -------------------------------------------------------
def hk_stability_loss(logits_o, logits_p, diagrams_o, diagrams_p, L_pi=1.0):
    """
    Hiraoka–Kusano inspired stability loss.
    """
    # Per-graph Δlogits (L2)
    dlog = torch.norm(logits_o - logits_p, dim=1)  # (B,)

    # Compute distance between diagram batches (mean over filtrations)
    d_perslay = torch.norm(diagrams_o - diagrams_p, p=2, dim=(1, 2, 3))
    d_perslay = d_perslay.clamp(min=1e-8)  # (B,)

    # Inequality penalty
    stab_ineq = torch.relu(dlog - L_pi * d_perslay).mean()

    return stab_ineq


# --------------------------
# 5) Train / evaluate loops
# --------------------------
def train_epoch(model, loader, opt, device, L_pi=1.0):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)

        # Get batch size
        batch_size = int(data.batch.max().item()) + 1

        # Prepare diagrams - they are already tensors from the dataset
        diagrams_o = data.diagrams  # Shape: (batch_size * num_filtrations, max_points, 2)
        diagrams_p = data.diagrams_pert  # Same shape

        # Reshape to (batch_size, num_filtrations, max_points, 2)
        diagrams_o = diagrams_o.reshape(batch_size, model.num_filtrations, -1, 2)
        diagrams_p = diagrams_p.reshape(batch_size, model.num_filtrations, -1, 2)

        # Forward passes
        out_o = model(data.x, data.edge_index, data.batch, diagrams_o)
        out_p = model(data.x, data.edge_index, data.batch, diagrams_p)

        # Compute loss
        ce = F.cross_entropy(out_o, data.y)
        stab = hk_stability_loss(out_o, out_p, diagrams_o, diagrams_p, L_pi=L_pi)
        loss = ce + 0.3 * stab

        # Backward pass
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / total_graphs if total_graphs > 0 else 0.0


@torch.no_grad()
def eval_acc(model, loader, device, drop_edges=False, p=0.1):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        ei = data.edge_index

        if drop_edges:
            ei, _ = dropout_edge(ei, p=p, force_undirected=True, training=True)

        # Get batch size
        batch_size = int(data.batch.max().item()) + 1

        # Prepare diagrams
        diagrams = data.diagrams.reshape(batch_size, model.num_filtrations, -1, 2)

        # Forward pass
        out = model(data.x, ei, data.batch, diagrams)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += batch_size

    return correct / total if total > 0 else 0.0


# ---------------
# 6) Main script
# ---------------
if __name__ == "__main__":
    SEED = 17
    L_PI = 1.0
    EPOCHS = 50
    DATASET_NAME = "REDDIT-BINARY"  # Change to any TUDataset name
    MAX_POINTS_PER_DIAGRAM = 50  # Maximum points per persistence diagram

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset with Perslay diagrams
    ds_full = PerslayGraphDataset(
        root=f"data/{DATASET_NAME}",
        name=DATASET_NAME,
        max_deg=50,
        p_edge_pert=0.05,
        max_points_per_diagram=MAX_POINTS_PER_DIAGRAM
    )

    # Stratified split
    train_ds, test_ds = stratified_split(ds_full, train_ratio=0.8, seed=SEED)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Get model parameters
    in_dim = train_ds.num_node_features
    n_cls = ds_full.num_classes
    num_filtrations = ds_full.num_filtrations

    print(f"\nDataset: {DATASET_NAME}")
    print(f"in_dim={in_dim}, num_filtrations={num_filtrations}, num_classes={n_cls}")
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    print(f"Max points per diagram: {MAX_POINTS_PER_DIAGRAM}")

    # Create model
    model = PerslayGIN_HK(
        in_dim=in_dim,
        hidden_dim=64,
        num_classes=n_cls,
        dataset_name=DATASET_NAME,
        num_filtrations=num_filtrations,
        max_points=MAX_POINTS_PER_DIAGRAM
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training loop
    best = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, opt, device, L_pi=L_PI)

        if epoch % 5 == 0:
            clean = eval_acc(model, test_loader, device, drop_edges=False)
            noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
            best = max(best, clean)
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Clean {clean:.3f} | Noisy {noisy:.3f} "
                  f"| Drop {clean - noisy:.3f}")

    print(f"\nDone in {(time.time() - t0):.1f}s")
    final_clean = eval_acc(model, test_loader, device, drop_edges=False)
    final_noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)

    print(f"\nFinal → Clean: {final_clean:.3f} | Noisy: {final_noisy:.3f} | Drop: {final_clean - final_noisy:.3f}")
    print(f"Best Clean: {best:.3f}")

    # Print model summary
    print(f"\nModel architecture:")
    print(f"- GIN layers: 2 layers with hidden_dim=64")
    print(f"- Perslay layers: {num_filtrations} PermutationEquivariant layers")
    print(f"- Classifier: 2-layer MLP with spectral normalization")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")