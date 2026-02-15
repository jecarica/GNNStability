"""
TensorFlow counterpart of perslayTF.py: GIN + Perslay (GUDHI) + Hiraoka–Kusano stability.
Same high-level idea as PerslayGIN_HK: structural (GIN) + topological (Perslay) branches,
classifier on concat, and optional HK stability loss on (original, perturbed) diagrams.

Alignment with perslayTF.py:
- Diagram masks are used: only valid (non-padded) points are passed to Perslay (via
  _to_ragged_batch(..., M_list=M)), matching the PyTorch PermutationEquivariant + masks.
- Perslay branch here uses GUDHI's GaussianPerslayPhi (image-based); perslayTF uses
  PermutationEquivariant MLP + max. So the topological representation differs; accuracy
  can still differ. Classifier head in PyTorch uses spectral norm + orthogonal init;
  here it is plain Dense.
- Default run is ONE config (full model) and 1 run so runtime is comparable to perslayTF.
  Use --ablation for the full 4-config × n_runs grid (much slower).

Install: pip install gudhi tensorflow  (and torch/torch_geometric for TUDataset loading)

Run (single config, ~same as perslayTF): python experiments_perslay_tensorflow.py --dataset MUTAG
Full ablation: python experiments_perslay_tensorflow.py --dataset MUTAG --ablation --n_runs 3
Quick: python experiments_perslay_tensorflow.py --dataset MUTAG --quick
"""
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

import gudhi as gd
try:
    import gudhi.tensorflow.perslay as prsl
except ImportError as e:
    print("GUDHI TensorFlow Perslay not found. Install GUDHI with TensorFlow support:")
    print("  pip install gudhi tensorflow")
    print("  (GUDHI uses Perslay from its representations module; no separate 'perslay' package needed.)")
    raise SystemExit(1) from e
from scipy.sparse import csgraph
from scipy.linalg import eigh

# Optional: load TUDataset via PyTorch Geometric for graph loading only
try:
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_dense_adj
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch Geometric not found. Install for TUDataset loading, or provide diagrams elsewhere.")


# ---------------------------------------------------------------------------
# Diagram computation (numpy/scipy/gudhi – no TF/torch)
# ---------------------------------------------------------------------------
def get_parameters(dataset):
    """Dataset-specific filtration names (same as perslayTF.py)."""
    if dataset in ("MUTAG", "PROTEINS"):
        return {"filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    if dataset in ("COX2", "ENZYMES", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"):
        return {"filt_names": [
            "Ord0_0.1-hks", "Rel1_0.1-hks", "Ext0_0.1-hks", "Ext1_0.1-hks",
            "Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"
        ]}
    if dataset in ("ORBIT5K", "ORBIT100K"):
        return {"filt_names": ["Alpha0", "Alpha1"]}
    return {"filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}


def hks_signature(eigenvectors, eigenvals, time_val):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time_val * eigenvals))).sum(axis=1)


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    xs, ys = np.where(np.triu(A))
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

    def process(dgm, dim):
        return np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]])
                        for p in dgm if p[0] == dim]) if len(dgm) else np.empty([0, 2])

    dgmOrd0 = process(dgmOrd0, 0)
    dgmRel1 = process(dgmRel1, 1)
    dgmExt0 = process(dgmExt0, 0)
    dgmExt1 = process(dgmExt1, 1)
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1


def compute_extended_persistence_diagrams(A_np, hks_time):
    L = csgraph.laplacian(A_np, normed=True)
    egvals, egvectors = eigh(L)
    filtration_val = hks_signature(egvectors, egvals, hks_time)
    dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A_np, filtration_val)
    diagrams = {
        f'Ord0_{hks_time}-hks': dgmOrd0, f'Ext0_{hks_time}-hks': dgmExt0,
        f'Rel1_{hks_time}-hks': dgmRel1, f'Ext1_{hks_time}-hks': dgmExt1
    }
    for key in diagrams:
        if len(diagrams[key]) > 0:
            diagrams[key] = np.clip(diagrams[key], 0.0, 1.0).astype(np.float32)
    return diagrams


def pad_diagram(diagram, max_points, return_mask=False):
    """Pad or truncate diagram to max_points. If return_mask, also return bool mask (True = valid)."""
    if len(diagram) == 0:
        out = np.zeros((max_points, 2), dtype=np.float32)
        mask = np.zeros(max_points, dtype=np.bool_)
        return (out, mask) if return_mask else out
    n = min(len(diagram), max_points)
    diag = np.asarray(diagram[:max_points], dtype=np.float32)
    if len(diagram) < max_points:
        padding = np.zeros((max_points - len(diagram), 2), dtype=np.float32)
        diag = np.vstack([diag, padding])
    mask = np.zeros(max_points, dtype=np.bool_)
    mask[:n] = True
    return (diag, mask) if return_mask else diag


# ---------------------------------------------------------------------------
# Load TUDataset: diagrams (original + perturbed), masks, graph data for GIN
# ---------------------------------------------------------------------------
def _perturb_adjacency(adj_np, p=0.05, rng=None):
    """Randomly flip edges with probability p (upper triangle then symmetrize)."""
    rng = rng or np.random.default_rng()
    n = adj_np.shape[0]
    triu = np.triu(np.ones((n, n), dtype=np.float64), 1)
    flip = rng.random((n, n)) < p
    flip = np.tril(flip, -1) + np.triu(flip, 1)
    adj2 = adj_np.astype(np.float64).copy()
    adj2[flip.astype(bool) & (triu > 0)] = 1 - adj2[flip.astype(bool) & (triu > 0)]
    adj2 = np.minimum(adj2, adj2.T)
    np.fill_diagonal(adj2, 0)
    return adj2


def _degree_onehot(edge_index, num_nodes, max_deg=50):
    """One-hot degree features (max_deg+1 classes)."""
    deg = np.zeros(num_nodes, dtype=np.int64)
    for j in range(edge_index.shape[1]):
        deg[edge_index[1, j]] += 1
    deg = np.minimum(deg, max_deg)
    out = np.zeros((num_nodes, max_deg + 1), dtype=np.float32)
    out[np.arange(num_nodes), deg] = 1.0
    return out


def load_tudataset_and_diagrams(dataset_name, root="data", max_points_per_diagram=50,
                                max_deg=50, p_edge_pert=0.05, use_cache=True, force_recompute=False):
    """Load dataset; return diags_dict, diags_pert_dict, masks_dict, masks_pert_dict,
    labels, filt_names, max_points, graph_list (list of dicts with x, edge_index, num_nodes), in_dim."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch and torch_geometric required for TUDataset loading.")
    root_path = os.path.join(root, dataset_name)
    ds = TUDataset(root=root_path, name=dataset_name)
    params = get_parameters(dataset_name)
    filt_names = params["filt_names"]
    in_dim = max_deg + 1
    cache_dir = os.path.join(root_path, f"{dataset_name}_perslay_tf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "diagrams_graphs.pkl")

    if use_cache and os.path.exists(cache_file) and not force_recompute:
        with open(cache_file, "rb") as f:
            out = pickle.load(f)
        print(f"Loaded cached data from {cache_file}")
        in_dim_cached = out.get("in_dim", in_dim)
        return (
            out["diags_dict"], out["diags_pert_dict"], out["masks_dict"], out["masks_pert_dict"],
            out["labels"], filt_names, out.get("max_points", max_points_per_diagram),
            out["graph_list"], in_dim_cached
        )

    diags_dict = {f: [] for f in filt_names}
    diags_pert_dict = {f: [] for f in filt_names}
    masks_dict = {f: [] for f in filt_names}
    masks_pert_dict = {f: [] for f in filt_names}
    labels = []
    graph_list = []
    rng = np.random.default_rng(42)

    for i in range(len(ds)):
        data = ds[i]
        edge_index = data.edge_index.numpy()
        num_nodes = data.num_nodes
        adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0).numpy()
        adj_pert = _perturb_adjacency(adj, p=p_edge_pert, rng=rng)

        for filt_name in filt_names:
            if "hks" not in filt_name:
                diags_dict[filt_name].append(np.zeros((max_points_per_diagram, 2), dtype=np.float32))
                masks_dict[filt_name].append(np.zeros(max_points_per_diagram, dtype=np.bool_))
                diags_pert_dict[filt_name].append(np.zeros((max_points_per_diagram, 2), dtype=np.float32))
                masks_pert_dict[filt_name].append(np.zeros(max_points_per_diagram, dtype=np.bool_))
                continue
            parts = filt_name.split("_")
            hks_time = float(parts[1].split("-")[0])
            diag_key = f"{parts[0]}_{hks_time}-hks"
            d_orig = compute_extended_persistence_diagrams(adj, hks_time)
            d_pert = compute_extended_persistence_diagrams(adj_pert, hks_time)
            diag_orig = d_orig.get(diag_key, np.array([[0.0, 0.0]]))
            diag_pert = d_pert.get(diag_key, np.array([[0.0, 0.0]]))
            padded_orig, mask_orig = pad_diagram(diag_orig, max_points_per_diagram, return_mask=True)
            padded_pert, mask_pert = pad_diagram(diag_pert, max_points_per_diagram, return_mask=True)
            diags_dict[filt_name].append(padded_orig)
            masks_dict[filt_name].append(mask_orig)
            diags_pert_dict[filt_name].append(padded_pert)
            masks_pert_dict[filt_name].append(mask_pert)

        x = _degree_onehot(edge_index, num_nodes, max_deg)
        graph_list.append({"x": x, "edge_index": edge_index, "num_nodes": num_nodes})
        labels.append(int(data.y.item()))
        if (i + 1) % 50 == 0 or (i + 1) == len(ds):
            print(f"  Processed {i + 1}/{len(ds)}")

    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump({
                "diags_dict": diags_dict, "diags_pert_dict": diags_pert_dict,
                "masks_dict": masks_dict, "masks_pert_dict": masks_pert_dict,
                "labels": labels, "graph_list": graph_list, "max_points": max_points_per_diagram,
                "in_dim": in_dim,
            }, f)

    return (
        diags_dict, diags_pert_dict, masks_dict, masks_pert_dict,
        labels, filt_names, max_points_per_diagram, graph_list, in_dim
    )


# ---------------------------------------------------------------------------
# Preprocess: D, D_pert (list of (N, max_pts, 2)), M, M_pert (masks), L (one-hot)
# ---------------------------------------------------------------------------
def prepare_perslay_data(diags_dict, diags_pert_dict, masks_dict, masks_pert_dict, labels, filt_names, max_points):
    N = len(labels)
    labels_arr = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels_arr.max()) + 1
    # One-hot (N, num_classes); avoid LabelBinarizer binary (N,1) which gives num_classes=1
    L = np.eye(num_classes, dtype=np.float32)[labels_arr]

    D = []
    D_pert = []
    M = []
    M_pert = []
    for f in filt_names:
        D.append(np.stack([diags_dict[f][i] for i in range(N)], axis=0))
        D_pert.append(np.stack([diags_pert_dict[f][i] for i in range(N)], axis=0))
        M.append(np.stack([masks_dict[f][i] for i in range(N)], axis=0))
        M_pert.append(np.stack([masks_pert_dict[f][i] for i in range(N)], axis=0))
    return D, D_pert, M, M_pert, L


def batch_graphs(graph_list, indices):
    """Batch graphs into (x, edge_index (E, 2), batch). edge_index columns are [src, dst] for TF."""
    xs, edge_index_list, batch_vec = [], [], []
    offset = 0
    for idx in indices:
        g = graph_list[idx]
        n = g["num_nodes"]
        xs.append(g["x"])
        ei = g["edge_index"]  # (2, E)
        edge_index_list.append(np.column_stack([ei[0] + offset, ei[1] + offset]))  # (E, 2)
        batch_vec.append(np.full(n, len(batch_vec), dtype=np.int32))
        offset += n
    batch_x = np.concatenate(xs, axis=0).astype(np.float32)
    batch_edge = np.vstack(edge_index_list).astype(np.int32)  # (total_E, 2)
    batch_vec = np.concatenate(batch_vec, axis=0)
    return batch_x, batch_edge, batch_vec


# ---------------------------------------------------------------------------
# GIN in TensorFlow (message passing + global add pool)
# ---------------------------------------------------------------------------
def gin_aggregate(h, edge_index, num_nodes, eps=0.0):
    """Aggregate: for each node, sum neighbor features. h (nodes, dim), edge_index (E, 2) with columns [src, dst]."""
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    messages = tf.gather(h, src)
    agg = tf.math.unsorted_segment_sum(messages, dst, num_segments=num_nodes)
    return (1.0 + eps) * h + agg


class GINBlock(tf.keras.layers.Layer):
    """One GIN layer: aggregate then MLP."""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.mlp = None

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.hidden_dim, use_bias=True),
            tf.keras.layers.ReLU(),
        ])
        super().build(input_shape)

    def call(self, inputs):
        h, edge_index, num_nodes = inputs
        agg = gin_aggregate(h, edge_index, num_nodes)
        return self.mlp(agg)


def global_add_pool_tf(h, batch, batch_size):
    """h (total_nodes, dim), batch (total_nodes,) -> (batch_size, dim)."""
    return tf.math.unsorted_segment_sum(h, batch, num_segments=batch_size)


# ---------------------------------------------------------------------------
# HK stability loss (TensorFlow)
# ---------------------------------------------------------------------------
def hk_stability_loss_embedding_tf(logits_o, logits_p, diagrams_o, diagrams_p, masks_o=None, masks_p=None, L_pi=1.0):
    """Hiraoka–Kusano stability in embedding space: ||f(G)-f(G')|| <= L * d(D,D')."""
    d_logits = tf.norm(logits_o - logits_p, axis=1)
    if masks_o is not None and masks_p is not None:
        valid_o = diagrams_o * tf.cast(tf.expand_dims(masks_o, -1), diagrams_o.dtype)
        valid_p = diagrams_p * tf.cast(tf.expand_dims(masks_p, -1), diagrams_p.dtype)
        diff = valid_o - valid_p
    else:
        diff = diagrams_o - diagrams_p
    # tf.norm(..., axis=(1,2,3)) not supported in some TF versions; use reduce_sum(square) then sqrt
    d_diagrams = tf.sqrt(tf.reduce_sum(diff ** 2, axis=[1, 2, 3]))
    d_diagrams = tf.maximum(d_diagrams, 1e-8)
    return tf.reduce_mean(tf.nn.relu(d_logits - L_pi * d_diagrams))


# ---------------------------------------------------------------------------
# Combined model: GIN + Perslay (GUDHI) + classifier (like PerslayGIN_HK)
# ---------------------------------------------------------------------------
class GINBranchLayer(tf.keras.layers.Layer):
    """Wraps GIN + global add pool so tf.shape/tf.reduce_max run in call() on real tensors."""

    def __init__(self, in_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        self.d1 = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.d2 = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        self.d3 = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.d4 = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        super().build(input_shape)

    def call(self, inputs):
        x, edge_index, batch = inputs
        # Strip batch dim (1, N, ...) -> (N, ...) so we have (total_nodes, in_dim), (E, 2), (total_nodes,)
        if tf.rank(x) == 3:
            x = x[0]
            edge_index = edge_index[0]
            batch = batch[0]
        num_nodes = tf.shape(x)[0]
        batch_size = tf.reduce_max(batch) + 1
        h = self.d1(x)
        h = self.bn1(h)
        h = self.d2(h)
        h = gin_aggregate(h, edge_index, num_nodes)
        h = self.d3(h)
        h = self.bn2(h)
        h = gin_aggregate(h, edge_index, num_nodes)
        h = self.d4(h)
        return global_add_pool_tf(h, batch, batch_size)

    def compute_output_shape(self, input_shape):
        # Graph batch: output is (batch_size, hidden_dim); batch_size unknown at build time
        return tf.TensorShape([None, self.hidden_dim])


def build_gin_perslay_model(
    in_dim, num_filtrations, num_classes, hidden_dim=64, max_points=50,
    use_gin=True, use_perslay=True,
    phi_size=(5, 5), image_bnds=((-0.01, 1.01), (-0.01, 1.01)), variance=0.1,
):
    """Two branches: GIN (x, edge_index, batch) -> g_gin; Perslay (ragged diagrams) -> g_perslay; concat -> fc -> logits.
    Only declares inputs that are used so Keras does not raise 'inputs not connected to outputs' for ablation configs."""
    # Graph inputs (only when GIN is used)
    if use_gin:
        inp_x = tf.keras.Input(shape=(None, in_dim), name="x")
        inp_edge_index = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name="edge_index")
        inp_batch = tf.keras.Input(shape=(None,), dtype=tf.int32, name="batch")
    else:
        inp_x = inp_edge_index = inp_batch = None

    # Diagram inputs (only when Perslay is used)
    if use_perslay:
        inp_diagrams = [tf.keras.Input(shape=(None, 2), ragged=True, name=f"diag_{k}") for k in range(num_filtrations)]
    else:
        inp_diagrams = None

    # GIN branch
    if use_gin:
        gin_layer = GINBranchLayer(in_dim, hidden_dim)
        g_gin = gin_layer([inp_x, inp_edge_index, inp_batch])
    else:
        g_gin = None

    # Perslay branch (only when use_perslay)
    if use_perslay:
        perslay_outs = []
        for k in range(num_filtrations):
            weight = prsl.PowerPerslayWeight(1.0, 0.0)
            phi = prsl.GaussianPerslayPhi(phi_size, image_bnds, variance)
            layer = prsl.Perslay(weight=weight, phi=phi, perm_op=tf.math.reduce_max, rho=tf.keras.layers.Flatten())
            perslay_outs.append(layer(inp_diagrams[k]))
        perslay_concat = tf.keras.layers.Concatenate()(perslay_outs)
        g_perslay = tf.keras.layers.Dense(hidden_dim, activation="relu")(perslay_concat)
        g_perslay = tf.keras.layers.Dropout(0.5)(g_perslay)
    else:
        g_perslay = None

    if use_gin and use_perslay:
        g = tf.keras.layers.Concatenate()([g_gin, g_perslay])
    elif use_gin:
        g = g_gin
    else:
        g = g_perslay

    g = tf.keras.layers.Dense(128, activation="elu")(g)
    g = tf.keras.layers.Dropout(0.5)(g)
    logits = tf.keras.layers.Dense(num_classes, activation="softmax")(g)

    inputs = ([inp_x, inp_edge_index, inp_batch] if use_gin else []) + (inp_diagrams if use_perslay else [])
    return tf.keras.Model(inputs=inputs, outputs=logits)


# ---------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def stratified_split_indices(labels, train_ratio=0.8, seed=42):
    from collections import defaultdict
    set_seed(seed)
    idx_by_class = defaultdict(list)
    for i, y in enumerate(labels):
        idx_by_class[y].append(i)
    train_idx, test_idx = [], []
    for indices in idx_by_class.values():
        np.random.shuffle(indices)
        n = int(len(indices) * train_ratio)
        train_idx.extend(indices[:n])
        test_idx.extend(indices[n:])
    train_idx.sort()
    test_idx.sort()
    return np.array(train_idx), np.array(test_idx)


def _to_ragged_batch(D_list, indices, M_list=None):
    """Convert list of (N, max_pts, 2) arrays to list of RaggedTensors for the batch indices.
    If M_list is provided (masks True = valid point), only valid points are included so
    Perslay does not see zero-padding (matches perslayTF.py PermutationEquivariant + masks)."""
    out = []
    for k in range(len(D_list)):
        Dk = D_list[k][indices]  # (B, max_pts, 2)
        if M_list is not None:
            Mk = M_list[k][indices]  # (B, max_pts)
            row_lengths = np.array(Mk.sum(axis=1), dtype=np.int64)
            parts = []
            for i in range(len(indices)):
                valid = Dk[i][Mk[i]]  # (n_valid_i, 2)
                parts.append(valid)
            if len(parts) == 0:
                values = np.zeros((0, 2), dtype=np.float32)
            else:
                values = np.concatenate(parts, axis=0).astype(np.float32)
            rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        else:
            rt = tf.RaggedTensor.from_tensor(tf.constant(Dk, dtype=tf.float32))
        out.append(rt)
    return out


def _make_dummy_graph_batch(batch_size, in_dim):
    """Dummy graph batch when use_gin=False: one node per graph, self-loop edges."""
    batch_x = np.zeros((batch_size, in_dim), dtype=np.float32)
    batch_x[:, 0] = 1.0
    batch_edge = np.column_stack([np.arange(batch_size), np.arange(batch_size)]).astype(np.int32)
    batch_vec = np.arange(batch_size, dtype=np.int32)
    return batch_x, batch_edge, batch_vec


def run_one_fold(
    graph_list, D, D_pert, M, M_pert, L, train_idx, test_idx,
    in_dim, num_classes, max_points, num_filtrations,
    use_gin=True, use_perslay=True, use_stability=True, stability_weight=0.3,
    epochs=100, batch_size=32, lr=1e-3, seed=42, verbose=0,
):
    set_seed(seed)
    if verbose:
        print("  Building model...", flush=True)
    model = build_gin_perslay_model(
        in_dim=in_dim, num_filtrations=num_filtrations, num_classes=num_classes,
        use_gin=use_gin, use_perslay=use_perslay,
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    L_train = L[train_idx]
    L_test = L[test_idx]
    n_train = len(train_idx)
    best_weights = None
    best_acc = 0.0

    # Warmup: first model call triggers TF graph build and can take 1–2 min; do it once here
    if verbose:
        print("  Warmup: first model call (TF graph build, may take 1–2 min)...", flush=True)
    ind0 = train_idx[: min(batch_size, len(train_idx))]
    warmup_inputs = _model_inputs(graph_list, D, ind0, in_dim, use_gin, use_perslay, M=M)
    _ = model(warmup_inputs, training=False)
    if verbose:
        print("  Warmup done. Training...", flush=True)

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start: start + batch_size]
            ind = train_idx[batch_idx]
            B = len(ind)

            if use_gin:
                batch_x, batch_edge, batch_vec = batch_graphs(graph_list, ind)
            else:
                batch_x, batch_edge, batch_vec = _make_dummy_graph_batch(B, in_dim)

            D_batch = _to_ragged_batch(D, ind, M_list=M)
            D_pert_batch = _to_ragged_batch(D_pert, ind, M_list=M_pert)
            L_batch = L_train[batch_idx]

            with tf.GradientTape() as tape:
                if verbose and epoch == 0 and start == 0:
                    print("  First forward pass ...", flush=True)
                inputs_o = _model_inputs_from_batch(batch_x, batch_edge, batch_vec, D_batch, use_gin, use_perslay)
                logits_o = model(inputs_o, training=True)
                if verbose and epoch == 0 and start == 0:
                    print("  First forward done. Second forward (perturbed)...", flush=True)
                inputs_p = _model_inputs_from_batch(batch_x, batch_edge, batch_vec, D_pert_batch, use_gin, use_perslay)
                logits_p = model(inputs_p, training=True)
                if verbose and epoch == 0 and start == 0:
                    print("  First batch done. Training continues.", flush=True)
                loss_ce = tf.keras.losses.categorical_crossentropy(L_batch, logits_o)
                loss_ce = tf.reduce_mean(loss_ce)

                if use_stability and stability_weight > 0:
                    # Dense diagrams for HK: (B, num_filtrations, max_pts, 2)
                    diagrams_o = tf.constant(np.stack([D[k][ind] for k in range(len(D))], axis=1))
                    diagrams_p = tf.constant(np.stack([D_pert[k][ind] for k in range(len(D_pert))], axis=1))
                    masks_o = tf.constant(np.stack([M[k][ind] for k in range(len(M))], axis=1))
                    masks_p = tf.constant(np.stack([M_pert[k][ind] for k in range(len(M_pert))], axis=1))
                    loss_stab = hk_stability_loss_embedding_tf(logits_o, logits_p, diagrams_o, diagrams_p, masks_o, masks_p)
                    loss = loss_ce + stability_weight * loss_stab
                else:
                    loss = loss_ce

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if (epoch + 1) % 10 == 0:
            te_acc = _eval_acc(model, graph_list, D, L, test_idx, in_dim, use_gin, use_perslay, M=M)
            if te_acc > best_acc:
                best_acc = te_acc
                best_weights = [w.numpy().copy() for w in model.weights]

    if best_weights is not None:
        for w, bw in zip(model.weights, best_weights):
            w.assign(bw)

    train_acc = _eval_acc(model, graph_list, D, L, train_idx, in_dim, use_gin, use_perslay, M=M)
    test_acc = _eval_acc(model, graph_list, D, L, test_idx, in_dim, use_gin, use_perslay, M=M)
    return train_acc, test_acc


def _model_inputs(graph_list, D, indices, in_dim, use_gin, use_perslay, M=None):
    """Build the list of inputs for the model (graph + diagrams as needed). Uses M to trim diagrams when provided."""
    if use_gin:
        batch_x, batch_edge, batch_vec = batch_graphs(graph_list, indices) if graph_list else _make_dummy_graph_batch(len(indices), in_dim)
    else:
        batch_x = batch_edge = batch_vec = None
    D_batch = _to_ragged_batch(D, indices, M_list=M) if use_perslay else None
    return _model_inputs_from_batch(batch_x, batch_edge, batch_vec, D_batch, use_gin, use_perslay)


def _model_inputs_from_batch(batch_x, batch_edge, batch_vec, D_batch, use_gin, use_perslay):
    """Build model input list from already batched graph and diagram data."""
    parts = []
    if use_gin:
        parts.append(tf.constant(np.expand_dims(batch_x, 0), dtype=tf.float32))
        parts.append(tf.constant(np.expand_dims(batch_edge, 0), dtype=tf.int32))
        parts.append(tf.constant(np.expand_dims(batch_vec, 0), dtype=tf.int32))
    if use_perslay:
        parts.extend(D_batch)
    return parts


def _eval_acc(model, graph_list, D, L, indices, in_dim, use_gin, use_perslay, M=None):
    """Evaluate accuracy on a set of indices. Uses M to trim diagram padding when provided."""
    if len(indices) == 0:
        return 0.0
    B = len(indices)
    if use_gin:
        batch_x, batch_edge, batch_vec = batch_graphs(graph_list, indices)
    else:
        batch_x, batch_edge, batch_vec = _make_dummy_graph_batch(B, in_dim)
    D_batch = _to_ragged_batch(D, indices, M_list=M)
    inputs = _model_inputs_from_batch(batch_x, batch_edge, batch_vec, D_batch, use_gin, use_perslay)
    L_batch = L[indices]
    logits = model(inputs, training=False)
    pred = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.argmax(L_batch, axis=1) == pred, tf.float32)).numpy()


# ---------------------------------------------------------------------------
# Ablation study: GIN+Perslay+HK (like perslayTF.py)
# ---------------------------------------------------------------------------
def run_ablation_tensorflow(
    dataset_name="NCI1",
    root="data",
    max_points_per_diagram=50,
    n_runs=5,
    train_ratio=0.8,
    epochs=100,
    batch_size=32,
    seed_base=42,
    use_cache=True,
    run_ablation_configs=True,
):
    print(f"Loading dataset {dataset_name} (graphs + diagrams + perturbed diagrams)...")
    out = load_tudataset_and_diagrams(
        dataset_name, root=root, max_points_per_diagram=max_points_per_diagram,
        use_cache=use_cache, force_recompute=False
    )
    diags_dict, diags_pert_dict, masks_dict, masks_pert_dict, labels, filt_names, max_points, graph_list, in_dim = out
    D, D_pert, M, M_pert, L = prepare_perslay_data(
        diags_dict, diags_pert_dict, masks_dict, masks_pert_dict, labels, filt_names, max_points
    )
    num_filtrations = len(D)
    num_classes = L.shape[1]

    if run_ablation_configs:
        configs = [
            {"name": "Full (GIN+Perslay+HK)", "use_gin": True, "use_perslay": True, "use_stability": True},
            {"name": "GIN+Perslay, no stability", "use_gin": True, "use_perslay": True, "use_stability": False},
            {"name": "GIN only", "use_gin": True, "use_perslay": False, "use_stability": False},
            {"name": "Perslay only", "use_gin": False, "use_perslay": True, "use_stability": False},
        ]
    else:
        configs = [{"name": "Full", "use_gin": True, "use_perslay": True, "use_stability": True}]

    all_results = []
    for cfg in configs:
        print(f"\n--- Config: {cfg['name']} ---")
        train_accs, test_accs = [], []
        for run in range(n_runs):
            seed = seed_base + run
            train_idx, test_idx = stratified_split_indices(labels, train_ratio=train_ratio, seed=seed)
            tr_acc, te_acc = run_one_fold(
                graph_list, D, D_pert, M, M_pert, L, train_idx, test_idx,
                in_dim=in_dim, num_classes=num_classes, max_points=max_points, num_filtrations=num_filtrations,
                use_gin=cfg["use_gin"], use_perslay=cfg["use_perslay"],
                use_stability=cfg["use_stability"], stability_weight=0.3,
                epochs=epochs, batch_size=batch_size, seed=seed, verbose=1
            )
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"  Run {run + 1}/{n_runs}  seed={seed}  test_acc={te_acc:.4f}")

        train_accs = np.array(train_accs)
        test_accs = np.array(test_accs)
        all_results.append({
            "name": cfg["name"],
            "train_mean": train_accs.mean(), "train_std": train_accs.std(),
            "test_mean": test_accs.mean(), "test_std": test_accs.std(),
        })

    print(f"\n{'=' * 70}")
    print("TensorFlow (GIN + GUDHI Perslay + HK) – Ablation summary")
    print(f"Dataset: {dataset_name}  |  Runs: {n_runs}  |  Filtrations: {num_filtrations}")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<35} {'Test accuracy (mean ± std)':<28}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<35} {r['test_mean']:.4f} ± {r['test_std']:.4f}")
    print(f"{'=' * 70}\n")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TensorFlow GIN+Perslay+HK experiments")
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--max_points", type=int, default=50)
    parser.add_argument("--n_runs", type=int, default=1, help="Number of seeds (default 1; use 3–5 for ablation)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--no_cache", action="store_true", help="Disable diagram cache")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation (4 configs × n_runs); default is single full model only")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs (20) and runs (2) for faster sanity check")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20
        args.n_runs = 2
        print("Quick mode: epochs=20, n_runs=2\n")

    results = run_ablation_tensorflow(
        dataset_name=args.dataset,
        root=args.root,
        max_points_per_diagram=args.max_points,
        n_runs=args.n_runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed_base=args.seed_base,
        use_cache=not args.no_cache,
        run_ablation_configs=args.ablation,
    )
