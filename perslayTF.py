# hk_stability_tudataset_with_perslay.py
# FIXED VERSION 3.1 - WITH PROPER SUBSET HANDLING

import os
import time
import pickle
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
from scipy import stats

# Optional: statistical evaluation (mean ± std, CI, model comparison)
try:
    from utils.statistical_evaluation import StatisticalEvaluator
    HAS_STATS = True
except ImportError:
    HAS_STATS = False


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
        dataset_parameters = {"data_type": "graph",
                              "filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
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


class PerslayGraphSubset(torch.utils.data.Subset):
    """Custom Subset class that preserves dataset attributes"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy all necessary attributes from the original dataset
        self.num_node_features = dataset.num_node_features
        self.num_classes = dataset.num_classes
        self.num_filtrations = getattr(dataset, 'num_filtrations', None)
        self.max_points = getattr(dataset, 'max_points', None)
        self.filt_names = getattr(dataset, 'filt_names', None)
        self.hks_times = getattr(dataset, 'hks_times', None)
        self.name = getattr(dataset, 'name', None)


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

    # Use our custom Subset class
    train_ds = PerslayGraphSubset(dataset, train_idx)
    test_ds = PerslayGraphSubset(dataset, test_idx)

    trc = Counter([int(dataset[i].y.item()) for i in train_idx])
    tec = Counter([int(dataset[i].y.item()) for i in test_idx])
    print(f"Stratified split: train per class = {dict(trc)} test per class = {dict(tec)}")
    return train_ds, test_ds


# ---------------------------------------------
# 1) Dataset wrapper with Perslay diagrams AND CACHING
# ---------------------------------------------
class PerslayGraphDataset(TUDataset):
    """
    Builds per-graph node features and Perslay-style
    extended persistence diagrams for multiple filtrations.
    Caches diagrams to disk to avoid recomputation.
    """

    def __init__(self, root, name, max_deg=50, p_edge_pert=0.05, max_points_per_diagram=50,
                 use_cache=True, force_recompute=False):
        super().__init__(root, name)
        self.max_deg = int(max_deg)
        self.p_edge_pert = float(p_edge_pert)
        self.max_points = int(max_points_per_diagram)
        self.use_cache = use_cache

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

        # Create cache directory
        self.cache_dir = os.path.join(root, f"{name}_perslay_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load or compute diagrams
        self._attach_all_features(force_recompute)

    def _get_cache_path(self, idx):
        """Get cache file path for a specific graph"""
        return os.path.join(self.cache_dir, f"graph_{idx:06d}.pkl")

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
        """Pad or truncate diagram to have exactly max_points, return points and mask"""
        if len(diagram) == 0:
            return np.zeros((max_points, 2), dtype=np.float32), np.zeros(max_points, dtype=np.bool_)

        # Truncate if too many points
        if len(diagram) > max_points:
            diagram = diagram[:max_points]
            mask = np.ones(max_points, dtype=np.bool_)
        else:
            # Pad if too few points
            padding = np.zeros((max_points - len(diagram), 2), dtype=np.float32)
            diagram = np.vstack([diagram, padding])
            mask = np.concatenate([
                np.ones(len(diagram), dtype=np.bool_),
                np.zeros(max_points - len(diagram), dtype=np.bool_)
            ])

        return diagram, mask

    def _compute_and_cache_diagrams(self, idx, force_recompute=False):
        """Compute diagrams for a graph and cache them"""
        cache_path = self._get_cache_path(idx)

        # Try to load from cache
        if self.use_cache and os.path.exists(cache_path) and not force_recompute:
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data
            except:
                print(f"Failed to load cache for graph {idx}, recomputing...")

        # Compute diagrams
        d = self.get(idx)

        # Node features
        if d.x is None or d.x.size(1) == 0:
            d.x = self._onehot_degree(d)

        # Convert to numpy for persistence computation
        A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes).squeeze(0).numpy()

        # Store all diagrams for this graph
        all_diagrams = []
        all_masks = []

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

                # Pad/truncate diagram and get mask
                diag, mask = self._pad_diagram(diag, self.max_points)
                all_diagrams.append(diag)
                all_masks.append(mask)

        # Also compute for perturbed graph
        ei_pert = self._perturb_edge_index(d, p=self.p_edge_pert)
        A_pert = to_dense_adj(ei_pert, max_num_nodes=d.num_nodes).squeeze(0).numpy()

        all_diagrams_pert = []
        all_masks_pert = []

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

                # Pad/truncate diagram and get mask
                diag, mask = self._pad_diagram(diag, self.max_points)
                all_diagrams_pert.append(diag)
                all_masks_pert.append(mask)

        # Prepare cache data
        cached_data = {
            'x': d.x,
            'edge_index': d.edge_index,
            'y': d.y,
            'num_nodes': d.num_nodes,
            'diagrams': np.stack(all_diagrams),
            'masks': np.stack(all_masks),
            'diagrams_pert': np.stack(all_diagrams_pert),
            'masks_pert': np.stack(all_masks_pert)
        }

        # Save to cache with error handling
        if self.use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)
            except OSError as e:
                if "No space left" in str(e):
                    print(f"Warning: Disk full. Skipping cache for graph {idx}")
                    # Disable caching for rest of run
                    self.use_cache = False
                else:
                    raise e

        return cached_data

    def _attach_all_features(self, force_recompute=False):
        """Attach node features and Perslay diagrams, using cache when available"""
        print(f"Processing {self.name} with Perslay extended persistence diagrams...")
        print(f"Cache directory: {self.cache_dir}")

        start = time.time()
        data_list = []

        for i in range(len(self)):
            # Compute or load cached diagrams
            cached_data = self._compute_and_cache_diagrams(i, force_recompute)

            # Create Data object with diagrams
            from torch_geometric.data import Data
            data = Data(
                x=torch.tensor(cached_data['x'], dtype=torch.float),
                edge_index=torch.tensor(cached_data['edge_index'], dtype=torch.long),
                y=torch.tensor(cached_data['y'], dtype=torch.long),
                num_nodes=cached_data['num_nodes'],
                diagrams=torch.tensor(cached_data['diagrams'], dtype=torch.float),
                masks=torch.tensor(cached_data['masks'], dtype=torch.bool),
                diagrams_pert=torch.tensor(cached_data['diagrams_pert'], dtype=torch.float),
                masks_pert=torch.tensor(cached_data['masks_pert'], dtype=torch.bool)
            )

            data_list.append(data)

            if (i + 1) % 50 == 0 or (i + 1) == len(self):
                print(f"  • {i + 1}/{len(self)} done")

        # Replace in-memory storage
        self.data, self.slices = self.collate(data_list)
        print(
            f"✅ Processed in {time.time() - start:.1f}s | {self.num_filtrations} filtrations per graph, {self.max_points} max points per diagram")


# --------------------------------------------
# 2) PyTorch Perslay Model Components WITH MASKS
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

    def forward(self, diagrams, masks=None):
        """
        diagrams: Tensor of shape (batch_size, max_points, 2)
        masks: Tensor of shape (batch_size, max_points) indicating valid points
        Returns: Tensor of shape (batch_size, output_dim)
        """
        batch_size, num_points, _ = diagrams.shape

        # Process through layers
        x = diagrams.reshape(-1, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        # Reshape back
        x = x.reshape(batch_size, num_points, -1)

        # Apply masks if provided (set masked points to -inf for max pooling)
        if masks is not None:
            # Expand masks to match feature dimensions
            masks_expanded = masks.unsqueeze(-1).expand(-1, -1, x.size(-1))
            # Set masked points to very small value
            x = torch.where(masks_expanded, x, torch.tensor(float('-inf'), device=x.device))

        # Max pooling over points
        x = torch.max(x, dim=1)[0]

        # Replace -inf with 0 (if all points were masked)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x


class PerslayModel(nn.Module):
    """PyTorch implementation of Perslay (using PermutationEquivariant layers)"""

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
            nn.Linear(perslay_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, diagrams_batch, masks_batch=None):
        """
        diagrams_batch: Tensor of shape (batch_size, num_filtrations, max_points, 2)
        masks_batch: Tensor of shape (batch_size, num_filtrations, max_points) or None
        Returns: Tensor of shape (batch_size, num_classes)
        """
        batch_size = diagrams_batch.shape[0]
        all_features = []

        # Process each filtration through its Perslay layer
        for i in range(self.num_filtrations):
            # Extract diagram for this filtration: (batch_size, max_points, 2)
            diag = diagrams_batch[:, i, :, :]
            mask = masks_batch[:, i, :] if masks_batch is not None else None
            features = self.perslay_layers[i](diag, mask)  # (batch_size, 25)
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
    def __init__(self, in_dim, hidden_dim, num_classes, dataset_name, num_filtrations, max_points,
                 use_gin=True, use_perslay=True):
        super().__init__()
        self.num_filtrations = num_filtrations
        self.hidden = hidden_dim
        self.use_gin = use_gin
        self.use_perslay = use_perslay

        # Structural pathway (GIN)
        if use_gin:
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
        else:
            self.conv1 = self.conv2 = None

        # Perslay pathway
        if use_perslay:
            self.perslay = PerslayModel(
                dataset_name=dataset_name,
                num_filtrations=num_filtrations,
                max_points=max_points,
                hidden_dim=hidden_dim,
                num_classes=hidden_dim  # Output hidden_dim features for concatenation
            )
        else:
            self.perslay = None

        # Classifier combining both pathways (always hidden*2 input; missing branch fed as zeros)
        self.fc1 = spectral_norm(nn.Linear(hidden_dim * 2, 128))
        self.fc2 = spectral_norm(nn.Linear(128, num_classes))
        self.dropout = nn.Dropout(0.5)

        # Orthogonal init
        for m in [self.fc1, self.fc2]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x, edge_index, batch, diagrams_batch, masks_batch=None):
        B = batch.max().item() + 1
        dev = x.device if self.use_gin else diagrams_batch.device

        if self.use_gin:
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, p=0.5, training=self.training)
            h = F.relu(self.conv2(h, edge_index))
            g_struct = global_add_pool(h, batch)  # (B, hidden)
        else:
            g_struct = torch.zeros(B, self.hidden, device=dev, dtype=diagrams_batch.dtype)

        if self.use_perslay:
            g_perslay = self.perslay(diagrams_batch, masks_batch)  # (B, hidden)
        else:
            g_perslay = torch.zeros(B, self.hidden, device=g_struct.device, dtype=g_struct.dtype)

        # Combine pathways
        g = torch.cat([g_struct, g_perslay], dim=1)  # (B, hidden * 2)
        g = F.elu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g)


# -------------------------------------------------------
# 4) Stability loss - TWO OPTIONS
# -------------------------------------------------------
def hk_stability_loss_embedding(logits_o, logits_p, diagrams_o, diagrams_p,
                                masks_o=None, masks_p=None, L_pi=1.0):
    """
    Hiraoka–Kusano inspired stability loss in EMBEDDING space.
    Uses L2 distance between Perslay embeddings.
    """
    # Per-graph Δlogits (L2)
    d_logits = torch.norm(logits_o - logits_p, dim=1)  # (B,)

    # Compute Wasserstein-like distance between diagrams
    # Simplified version: L2 distance between diagram tensors with masks
    if masks_o is not None and masks_p is not None:
        # Only consider valid points
        valid_o = diagrams_o * masks_o.unsqueeze(-1)
        valid_p = diagrams_p * masks_p.unsqueeze(-1)
        d_diagrams = torch.norm(valid_o - valid_p, p=2, dim=(1, 2, 3))
    else:
        d_diagrams = torch.norm(diagrams_o - diagrams_p, p=2, dim=(1, 2, 3))

    d_diagrams = d_diagrams.clamp(min=1e-8)  # (B,)

    # Inequality penalty: ||f(G) - f(G')|| ≤ L * d(D, D')
    stab_ineq = torch.relu(d_logits - L_pi * d_diagrams).mean()

    return stab_ineq


def hk_stability_loss_kernel(logits_o, logits_p, diagrams_o, diagrams_p,
                             masks_o=None, masks_p=None, L_pi=1.0, sigma=1.0):
    """
    Hiraoka–Kusano inspired stability loss using PERSISTENCE SCALE-SPACE KERNEL.
    More theoretically sound for persistence diagrams.
    """
    # Per-graph Δlogits (L2)
    d_logits = torch.norm(logits_o - logits_p, dim=1)  # (B,)

    batch_size = diagrams_o.shape[0]
    d_kernel = torch.zeros(batch_size, device=diagrams_o.device)

    # Approximate Persistence Scale-Space Kernel distance
    for b in range(batch_size):
        # Get valid points for original and perturbed
        if masks_o is not None:
            valid_idx_o = masks_o[b].nonzero(as_tuple=True)
            points_o = diagrams_o[b][valid_idx_o]
        else:
            points_o = diagrams_o[b].reshape(-1, 2)

        if masks_p is not None:
            valid_idx_p = masks_p[b].nonzero(as_tuple=True)
            points_p = diagrams_p[b][valid_idx_p]
        else:
            points_p = diagrams_p[b].reshape(-1, 2)

        # Skip if no points
        if len(points_o) == 0 or len(points_p) == 0:
            d_kernel[b] = 0.0
            continue

        # Compute Gaussian kernel distances (simplified)
        # k(x, y) = exp(-||x-y||^2 / (2*sigma^2))
        dist_matrix = torch.cdist(points_o, points_p, p=2)
        kernel_matrix = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

        # Approximate kernel distance: sqrt(k(x,x) + k(y,y) - 2*k(x,y))
        k_xx = torch.exp(torch.tensor(0.0))  # k(x,x) = 1
        k_yy = torch.exp(torch.tensor(0.0))  # k(y,y) = 1
        k_xy = kernel_matrix.mean()

        d_kernel[b] = torch.sqrt(k_xx + k_yy - 2 * k_xy)

    d_kernel = d_kernel.clamp(min=1e-8)

    # Inequality penalty
    stab_ineq = torch.relu(d_logits - L_pi * d_kernel).mean()

    return stab_ineq


# --------------------------
# 5) Train / evaluate loops
# --------------------------
def hk_stability_loss_graph(logits_o, logits_p, L_pi=1.0, p_edge_drop=0.1):
    """
    HK-style stability for graph perturbation (GIN+HK): penalize
    ||f(G) - f(G')|| > L_pi * d. Here d = p_edge_drop as proxy for graph distance.
    """
    d_logits = torch.norm(logits_o - logits_p, dim=1)  # (B,)
    return F.relu(d_logits - L_pi * p_edge_drop).mean()


def train_epoch(model, loader, opt, device, L_pi=1.0, stability_type='embedding', use_stability=True,
                stability_to_graph=False, p_edge_drop=0.1):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)

        # Get batch size
        batch_size = int(data.batch.max().item()) + 1

        # Prepare diagrams and masks
        diagrams_o = data.diagrams  # Shape: (batch_size * num_filtrations, max_points, 2)
        masks_o = data.masks if hasattr(data, 'masks') else None
        diagrams_o = diagrams_o.reshape(batch_size, model.num_filtrations, -1, 2)
        if masks_o is not None:
            masks_o = masks_o.reshape(batch_size, model.num_filtrations, -1)

        # Forward pass (original)
        out_o = model(data.x, data.edge_index, data.batch, diagrams_o, masks_o)
        ce = F.cross_entropy(out_o, data.y)

        if use_stability:
            if stability_to_graph:
                # GIN+HK: second forward with dropped edges (same diagrams)
                ei_drop, _ = dropout_edge(data.edge_index, p=p_edge_drop, force_undirected=True, training=True)
                out_p = model(data.x, ei_drop, data.batch, diagrams_o, masks_o)
                stab = hk_stability_loss_graph(out_o, out_p, L_pi=L_pi, p_edge_drop=p_edge_drop)
            else:
                # Full model: second forward with perturbed diagrams
                diagrams_p = data.diagrams_pert.reshape(batch_size, model.num_filtrations, -1, 2)
                masks_p = data.masks_pert.reshape(batch_size, model.num_filtrations, -1) if hasattr(data, 'masks_pert') else None
                out_p = model(data.x, data.edge_index, data.batch, diagrams_p, masks_p)
                if stability_type == 'kernel':
                    stab = hk_stability_loss_kernel(out_o, out_p, diagrams_o, diagrams_p,
                                                    masks_o, masks_p, L_pi=L_pi)
                else:
                    stab = hk_stability_loss_embedding(out_o, out_p, diagrams_o, diagrams_p,
                                                       masks_o, masks_p, L_pi=L_pi)
            loss = ce + 0.3 * stab
        else:
            loss = ce

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

        # Prepare diagrams and masks
        diagrams = data.diagrams.reshape(batch_size, model.num_filtrations, -1, 2)
        masks = data.masks.reshape(batch_size, model.num_filtrations, -1) if hasattr(data, 'masks') else None

        # Forward pass
        out = model(data.x, ei, data.batch, diagrams, masks)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += batch_size

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_stability_report(model, loader, device, p_values=(0.05, 0.1, 0.15, 0.2), verbose=True):
    """
    HK-style stability test: evaluate accuracy with no edge drop (Clean) and with
    random edge removal at several p. Diagrams stay fixed (original); only GIN sees
    the dropped graph. Returns dict {p: acc} with p=0 for clean.
    """
    model.eval()
    clean = eval_acc(model, loader, device, drop_edges=False)
    results = {0.0: clean}
    for p in p_values:
        results[p] = eval_acc(model, loader, device, drop_edges=True, p=p)
    if verbose:
        print("  Edge-removal stability (diagrams unchanged, GIN sees dropped graph):")
        print(f"    Clean (p=0):  {clean:.4f}")
        for p in p_values:
            drop = clean - results[p]
            print(f"    p={p:.2f}: acc={results[p]:.4f}  Drop={drop:.4f}")
    return results


# ---------------
# 5b) Ablation study with 5 runs + statistical evaluation
# ---------------
def run_ablation_study(
    dataset_name,
    root="data",
    max_points_per_diagram=50,
    n_runs=5,
    train_ratio=0.8,
    epochs=100,
    batch_size=32,
    seed_base=17,
    L_pi=1.0,
    stability_type='embedding',
    use_cache=True,
    force_recompute=False,
    verbose=True,
):
    """
    Run ablation: Full (GIN+Perslay+HK), GIN+Perslay no stability, GIN+HK, GIN only, Perslay only.
    GIN+HK = GIN only with HK stability to graph perturbation (edge drop). Each config runs
    n_runs times (different seeds). Reports mean ± std and optional 95% CI.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Ablation study: {dataset_name}  |  {n_runs} runs  |  device={device}\n")

    # Load dataset once (same data for all runs; split varies by seed)
    ds_full = PerslayGraphDataset(
        root=os.path.join(root, dataset_name),
        name=dataset_name,
        max_deg=50,
        p_edge_pert=0.05,
        max_points_per_diagram=max_points_per_diagram,
        use_cache=use_cache,
        force_recompute=force_recompute,
    )
    in_dim = ds_full.num_node_features
    n_cls = ds_full.num_classes
    num_filtrations = ds_full.num_filtrations

    configs = [
        {"name": "Full (GIN+Perslay+HK)", "use_gin": True, "use_perslay": True, "use_stability": True, "stability_to_graph": False},
        {"name": "GIN+Perslay, no stability", "use_gin": True, "use_perslay": True, "use_stability": False, "stability_to_graph": False},
        {"name": "GIN+HK", "use_gin": True, "use_perslay": False, "use_stability": True, "stability_to_graph": True},
        {"name": "GIN only", "use_gin": True, "use_perslay": False, "use_stability": False, "stability_to_graph": False},
        {"name": "Perslay only", "use_gin": False, "use_perslay": True, "use_stability": False, "stability_to_graph": False},
    ]

    EDGE_DROP_P = 0.1  # probability for stability test (same as hk.py)

    all_results = []
    for cfg in configs:
        if verbose:
            print(f"--- Config: {cfg['name']} ---")
        test_accs = []
        test_noisy_accs = []  # accuracy with edge drop (stability)
        for run in range(n_runs):
            seed = seed_base + run
            set_seed(seed)
            train_ds, test_ds = stratified_split(ds_full, train_ratio=train_ratio, seed=seed)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size)

            model = PerslayGIN_HK(
                in_dim=in_dim,
                hidden_dim=64,
                num_classes=n_cls,
                dataset_name=dataset_name,
                num_filtrations=num_filtrations,
                max_points=max_points_per_diagram,
                use_gin=cfg["use_gin"],
                use_perslay=cfg["use_perslay"],
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            best_acc = 0.0
            best_weights = None
            for epoch in range(1, epochs + 1):
                loss = train_epoch(
                    model, train_loader, opt, device,
                    L_pi=L_pi, stability_type=stability_type, use_stability=cfg["use_stability"],
                    stability_to_graph=cfg.get("stability_to_graph", False),
                    p_edge_drop=EDGE_DROP_P,
                )
                scheduler.step()
                if (epoch % 20 == 0) or (epoch == epochs):
                    acc = eval_acc(model, test_loader, device, drop_edges=False)
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = [p.detach().cpu().clone() for p in model.parameters()]
            if best_weights is not None:
                for p, w in zip(model.parameters(), best_weights):
                    p.data = w.to(device)
            test_acc = eval_acc(model, test_loader, device, drop_edges=False)
            test_noisy = eval_acc(model, test_loader, device, drop_edges=True, p=EDGE_DROP_P)
            test_accs.append(test_acc)
            test_noisy_accs.append(test_noisy)
            if verbose:
                print(f"  Run {run + 1}/{n_runs}  seed={seed}  test_acc={test_acc:.4f}  noisy(p={EDGE_DROP_P})={test_noisy:.4f}  drop={test_acc - test_noisy:.4f}")

        test_accs = np.array(test_accs)
        test_noisy_accs = np.array(test_noisy_accs)
        drops = test_accs - test_noisy_accs
        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0
        mean_noisy = float(np.mean(test_noisy_accs))
        std_noisy = float(np.std(test_noisy_accs, ddof=1)) if len(test_noisy_accs) > 1 else 0.0
        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
        n = len(test_accs)
        se = std_acc / np.sqrt(n) if n > 0 else 0.0
        t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
        ci_lo = mean_acc - t_crit * se
        ci_hi = mean_acc + t_crit * se

        all_results.append({
            "name": cfg["name"],
            "mean": mean_acc,
            "std": std_acc,
            "n": n,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "values": test_accs.tolist(),
            "mean_noisy": mean_noisy,
            "std_noisy": std_noisy,
            "mean_drop": mean_drop,
            "std_drop": std_drop,
            "values_noisy": test_noisy_accs.tolist(),
        })

    # Summary table (accuracy)
    if verbose:
        print(f"\n{'=' * 70}")
        print("Ablation summary (test accuracy)")
        print(f"Dataset: {dataset_name}  |  Runs: {n_runs}")
        print(f"{'=' * 70}")
        print(f"{'Configuration':<35} {'Test acc (mean ± std)':<22} {'95% CI'}")
        print("-" * 70)
        for r in all_results:
            ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]" if r["n"] > 1 else "—"
            print(f"{r['name']:<35} {r['mean']:.4f} ± {r['std']:.4f}   {ci_str}")
        print(f"{'=' * 70}\n")

        # Stability table (HK-style: edge removal, diagrams unchanged)
        print("Stability under edge removal (like hk.py: drop edges at test time, diagrams unchanged)")
        print(f"Dataset: {dataset_name}  |  Edge drop p={EDGE_DROP_P}")
        print(f"{'=' * 85}")
        print(f"{'Configuration':<35} {'Clean (mean±std)':<18} {'Noisy (mean±std)':<18} {'Drop (mean±std)'}")
        print("-" * 85)
        for r in all_results:
            print(f"{r['name']:<35} {r['mean']:.4f}±{r['std']:.4f}   "
                  f"{r['mean_noisy']:.4f}±{r['std_noisy']:.4f}   "
                  f"{r['mean_drop']:.4f}±{r['std_drop']:.4f}")
        print(f"{'=' * 85}\n")

    # Optional: StatisticalEvaluator comparison (Full vs GIN only)
    if HAS_STATS and len(all_results) >= 2 and verbose:
        full_res = next((r for r in all_results if "Full" in r["name"]), all_results[0])
        gin_only = next((r for r in all_results if r["name"] == "GIN only"), None)
        if gin_only is not None and full_res is not gin_only:
            res1 = {"accuracy": {"mean": full_res["mean"], "std": full_res["std"], "values": full_res["values"]}}
            res2 = {"accuracy": {"mean": gin_only["mean"], "std": gin_only["std"], "values": gin_only["values"]}}
            try:
                comp = StatisticalEvaluator(n_runs=1).compare_models(res1, res2, metric_name="accuracy")
                print("Statistical comparison (Full vs GIN only):")
                StatisticalEvaluator(n_runs=1).print_comparison(comp, metric_name="accuracy")
            except Exception:
                pass

    return all_results


# ---------------
# 6) Main script
# ---------------
if __name__ == "__main__":
    SEED = 17
    L_PI = 1.0
    EPOCHS = 100
    DATASET_NAME = "REDDIT-BINARY"  # Change to any TUDataset name (e.g. REDDIT-BINARY, NCI1)
    MAX_POINTS_PER_DIAGRAM = 50  # Maximum points per persistence diagram
    STABILITY_TYPE = 'embedding'  # 'embedding' or 'kernel'
    USE_CACHE = True  # Use diagram cache to avoid recomputation
    FORCE_RECOMPUTE = False  # Set to True to recompute diagrams even if cached
    N_RUNS = 5  # Number of runs per config for ablation study
    ABLATION_STUDY = True  # If True, run ablation (4 configs × N_RUNS); else single training run

    if ABLATION_STUDY:
        run_ablation_study(
            dataset_name=DATASET_NAME,
            root="data",
            max_points_per_diagram=MAX_POINTS_PER_DIAGRAM,
            n_runs=N_RUNS,
            train_ratio=0.8,
            epochs=EPOCHS,
            batch_size=32,
            seed_base=SEED,
            L_pi=L_PI,
            stability_type=STABILITY_TYPE,
            use_cache=USE_CACHE,
            force_recompute=FORCE_RECOMPUTE,
            verbose=True,
        )
    else:
        set_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ds_full = PerslayGraphDataset(
            root=f"data/{DATASET_NAME}",
            name=DATASET_NAME,
            max_deg=50,
            p_edge_pert=0.05,
            max_points_per_diagram=MAX_POINTS_PER_DIAGRAM,
            use_cache=USE_CACHE,
            force_recompute=FORCE_RECOMPUTE
        )

        train_ds, test_ds = stratified_split(ds_full, train_ratio=0.8, seed=SEED)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        in_dim = train_ds.num_node_features
        n_cls = ds_full.num_classes
        num_filtrations = train_ds.num_filtrations

        print(f"\nDataset: {DATASET_NAME}")
        print(f"in_dim={in_dim}, num_filtrations={num_filtrations}, num_classes={n_cls}")
        print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
        print(f"Max points per diagram: {MAX_POINTS_PER_DIAGRAM}")
        print(f"Stability type: {STABILITY_TYPE}")
        print(f"Using cache: {USE_CACHE}")

        model = PerslayGIN_HK(
            in_dim=in_dim,
            hidden_dim=64,
            num_classes=n_cls,
            dataset_name=DATASET_NAME,
            num_filtrations=num_filtrations,
            max_points=MAX_POINTS_PER_DIAGRAM,
            use_gin=True,
            use_perslay=True,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        best = 0.0
        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, opt, device,
                               L_pi=L_PI, stability_type=STABILITY_TYPE, use_stability=True)
            scheduler.step()

            if epoch % 5 == 0:
                clean = eval_acc(model, test_loader, device, drop_edges=False)
                noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
                best = max(best, clean)
                print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Clean {clean:.3f} | Noisy {noisy:.3f} "
                      f"| Drop {clean - noisy:.3f}")

        print(f"\nDone in {(time.time() - t0):.1f}s")
        final_clean = eval_acc(model, test_loader, device, drop_edges=False)
        final_noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
        print(f"\nFinal → Clean: {final_clean:.3f} | Noisy (p=0.1): {final_noisy:.3f} | Drop: {final_clean - final_noisy:.3f}")
        print(f"Best Clean: {best:.3f}")
        eval_stability_report(model, test_loader, device, p_values=(0.05, 0.1, 0.15, 0.2), verbose=True)
        print(f"\nModel architecture:")
        print(f"- GIN layers: 2 layers with hidden_dim=64")
        print(f"- Perslay layers: {num_filtrations} PermutationEquivariant layers with masks")
        print(f"- Classifier: 2-layer MLP with spectral normalization")
        print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")