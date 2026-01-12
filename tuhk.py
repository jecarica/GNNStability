# hk_stability_tudataset.py
# Hiraoka–Kusano–inspired training with stability regularization for TUDatasets

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gudhi as gd

from gudhi.representations import PersistenceImage
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse, dropout_edge
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import TUDataset

from torch.nn.utils.parametrizations import spectral_norm


# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(dataset, train_ratio=0.8, seed=17):
    # Generic stratified split for multi-class datasets
    set_seed(seed)
    labels = [int(data.y.item()) for data in dataset]

    # Count samples per class
    from collections import Counter
    class_counts = Counter(labels)

    # Create indices per class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Shuffle and split each class
    train_idx = []
    test_idx = []

    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_idx.extend(indices[:split_idx])
        test_idx.extend(indices[split_idx:])

    train_idx.sort()
    test_idx.sort()

    # Create new datasets
    train_ds = dataset.__class__(dataset.root, name=dataset.name)
    test_ds = dataset.__class__(dataset.root, name=dataset.name)

    # Slice into new InMemoryDatasets
    train_list = [dataset[i] for i in train_idx]
    test_list = [dataset[i] for i in test_idx]

    train_ds.data, train_ds.slices = dataset.collate(train_list)
    test_ds.data, test_ds.slices = dataset.collate(test_list)

    # Sanity report
    trc = Counter([int(d.y.item()) for d in train_list])
    tec = Counter([int(d.y.item()) for d in test_list])
    print(f"Stratified split: train per class = {dict(trc)} test per class = {dict(tec)}")
    return train_ds, test_ds


# ---------------------------------------------
# 1) Dataset wrapper (HK-inspired persistence)
# ---------------------------------------------
class HKGraphDataset(TUDataset):
    """
    Builds per-graph node features and HK-style
    windowed persistence images (H0+H1 on [r0, r1]) plus a small
    perturbed PI for stability regularization.
    """

    def __init__(self, root, name, r0=0.4, r1=1.2, res=10, sigma=0.1, max_deg=50, p_edge_pert=0.05):
        super().__init__(root, name)
        self.r0, self.r1 = float(r0), float(r1)
        self.res = int(res)
        self.sigma = float(sigma)
        self.max_deg = int(max_deg)
        self.p_edge_pert = float(p_edge_pert)
        self.pers = PersistenceImage(bandwidth=self.sigma, resolution=[self.res, self.res])
        self._attach_all_features()

    # ---- helpers for PH ----
    def _shortest_path_matrix(self, edge_index, num_nodes):
        adj = np.zeros((num_nodes, num_nodes), dtype=float)
        ei = edge_index.cpu().numpy()
        adj[ei[0], ei[1]] = 1.0
        adj[ei[1], ei[0]] = 1.0

        dist = np.full((num_nodes, num_nodes), np.inf, dtype=float)
        np.fill_diagonal(dist, 0.0)
        dist[adj > 0] = 1.0
        # Floyd–Warshall
        for k in range(num_nodes):
            dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
        return dist

    @staticmethod
    def _window_intervals(dgm, r0, r1):
        out = []
        for b, d in dgm:
            if d <= r0 or b >= r1:
                continue
            out.append([max(b, r0), min(d, r1)])
        return np.array(out, dtype=float) if len(out) else np.empty((0, 2), dtype=float)

    def _persistence_image_H0H1(self, dist, r0, r1):
        rc = gd.RipsComplex(distance_matrix=dist, max_edge_length=r1)
        st = rc.create_simplex_tree(max_dimension=2)
        st.persistence()

        dgm0 = st.persistence_intervals_in_dimension(0)
        dgm1 = st.persistence_intervals_in_dimension(1)

        rel0 = self._window_intervals(dgm0, r0, r1)
        rel1 = self._window_intervals(dgm1, r0, r1)

        def pi_vec(intervals):
            if intervals.shape[0] == 0:
                return np.zeros(self.res * self.res, dtype=float)
            return self.pers.fit_transform([intervals])[0]

        pi0 = pi_vec(rel0)
        pi1 = pi_vec(rel1)
        return np.concatenate([pi0, pi1], axis=0)  # 2*(res^2)

    # ---- helpers for features ----
    def _onehot_degree(self, data):
        deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg = torch.clamp(deg, max=self.max_deg)
        return F.one_hot(deg, num_classes=self.max_deg + 1).float()

    def _perturb_edge_index(self, data, p=0.05):
        A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        M = (torch.rand_like(A) < p).triu(1)
        A2 = A.clone()
        A2[M] = 1 - A2[M]
        A2.fill_diagonal_(0)
        A2 = torch.minimum(A2, A2.t())
        return dense_to_sparse(A2)[0]

    def _attach_all_features(self):
        print("Attaching node features + HK-style windowed persistence images…")
        start = time.time()
        data_list = []
        for i in range(len(self)):
            d = self.get(i)

            # Node features (use existing features or create one-hot degree)
            if d.x is None or d.x.size(1) == 0:
                d.x = self._onehot_degree(d)

            # Topo features (original)
            dist = self._shortest_path_matrix(d.edge_index, d.num_nodes)
            topo = self._persistence_image_H0H1(dist, self.r0, self.r1)
            d.topo = torch.tensor(topo, dtype=torch.float)  # (2*res^2,)

            # Topo features (perturbed graph)
            ei_pert = self._perturb_edge_index(d, p=self.p_edge_pert)
            dist_p = self._shortest_path_matrix(ei_pert, d.num_nodes)
            topo_p = self._persistence_image_H0H1(dist_p, self.r0, self.r1)
            d.topo_pert = torch.tensor(topo_p, dtype=torch.float)

            data_list.append(d)
            if (i + 1) % 50 == 0 or (i + 1) == len(self):
                print(f"  • {i + 1}/{len(self)} done")

        # Replace in-memory storage with enriched data:
        self.data, self.slices = self.collate(data_list)
        print(f"Attached in {time.time() - start:.1f}s | topo_dim={2 * (self.res * self.res)}")


# --------------------------------------------
# 2) Model: GIN + topo fusion + spectral_norm
# --------------------------------------------
class TopoGIN_HK(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, topo_dim):
        super().__init__()
        self.topo_dim = topo_dim
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

        # Topology branch with spectral normalization (encourages Lipschitz control)
        self.topo_net = nn.Sequential(
            spectral_norm(nn.Linear(topo_dim, hidden_dim)),
            nn.ReLU(),
        )

        # Classifier with spectral normalization
        self.fc1 = spectral_norm(nn.Linear(2 * hidden_dim, 128))
        self.fc2 = spectral_norm(nn.Linear(128, num_classes))
        self.dropout = nn.Dropout(0.5)

        # Orthogonal init (helps keep operator norms tame)
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight)

    def _reshape_topo(self, topo_vec, batch):
        """Turn concatenated 1-D topo (length B*D) back into (B, D)."""
        B = int(batch.max().item()) + 1
        D = self.topo_dim
        if topo_vec.dim() == 1:
            if topo_vec.numel() == B * D:
                return topo_vec.view(B, D)
            if topo_vec.numel() == D:
                return topo_vec.view(1, D).expand(B, -1)
            return topo_vec.view(B, -1)
        if topo_vec.dim() == 2:
            if topo_vec.size(0) == B:
                return topo_vec
            if topo_vec.size(0) == 1 and B > 1:
                return topo_vec.expand(B, -1)
            if topo_vec.numel() == B * D:
                return topo_vec.reshape(B, D)
        flat = topo_vec.view(-1)
        if flat.numel() < B * D:
            pad = flat.new_zeros(B * D - flat.numel())
            flat = torch.cat([flat, pad], dim=0)
        return flat[: B * D].view(B, D)

    def forward(self, x, edge_index, batch, topo_vec):
        # Structural
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        g_struct = global_add_pool(h, batch)  # (B, hidden)

        # Topological (reshape → encode)
        topo = self._reshape_topo(topo_vec, batch)  # (B, topo_dim)
        g_topo = self.topo_net(topo)  # (B, hidden)

        # Fuse + classify
        g = torch.cat([g_struct, g_topo], dim=1)  # (B, 2*hidden)
        g = F.elu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g)


# -------------------------------------------------------
# 3) Stability loss & a simple certified-radius estimator
# -------------------------------------------------------
def hk_stability_loss(logits_o, logits_p, topo_o, topo_p, batch, topo_dim, L_pi=1.0, lam_kld=0.0):
    """
    Hiraoka–Kusano inspired: penalize violations of ||Δf|| ≤ L_pi * ||ΔPI||.

    loss = ReLU( ||Δlogits||_2 - L_pi * ||ΔPI||_2 ) + lam_kld * symKL(p_o || p_p)

    All terms are averaged over graphs in the batch.
    """
    B = int(batch.max().item()) + 1

    # Per-graph Δlogits (L2)
    dlog = torch.norm(logits_o - logits_p, dim=1)  # (B,)

    # Per-graph ΔPI (L2)
    def reshape_topo(t):
        if t.dim() == 1 and t.numel() == B * topo_dim:
            return t.view(B, topo_dim)
        if t.dim() == 2 and t.size(0) == B:
            return t
        return t.view(B, -1)

    to = reshape_topo(topo_o)
    tp = reshape_topo(topo_p)
    dpi = torch.norm(to - tp, dim=1).clamp(min=1e-8)  # (B,)

    # Inequality penalty
    stab_ineq = torch.relu(dlog - L_pi * dpi).mean()

    # Optional symmetric KL between predictions
    if lam_kld > 0:
        po = F.log_softmax(logits_o, dim=1)
        pp = F.log_softmax(logits_p, dim=1)
        sym_kld = (po.exp() * (po - pp)).sum(dim=1).mean() + (pp.exp() * (pp - po)).sum(dim=1).mean()
    else:
        sym_kld = torch.tensor(0.0, device=logits_o.device)

    return stab_ineq + lam_kld * sym_kld


@torch.no_grad()
def certify_radius(logits, topo, topo_pert, L_pi=1.0, batch=None, topo_dim=None):
    """
    Simple proxy: E[ (margin) / (L_pi * ||ΔPI||_1) ] over a loader batch.
    """
    probs = F.softmax(logits, dim=1)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])  # (B,)

    if topo.dim() == 1 and batch is not None and topo_dim is not None:
        B = int(batch.max().item()) + 1
        topo = topo.view(B, topo_dim)
        topo_pert = topo_pert.view(B, topo_dim)

    d_pi = (topo - topo_pert).abs().sum(dim=1).clamp(min=1e-8)
    return (margin / (L_pi * d_pi)).mean().item()


# --------------------------
# 4) Train / evaluate loops
# --------------------------
def train_epoch(model, loader, opt, device, topo_dim, L_pi=1.0, lam_kld=0.0):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)

        out_o = model(data.x, data.edge_index, data.batch, data.topo)
        out_p = model(data.x, data.edge_index, data.batch, data.topo_pert)

        ce = F.cross_entropy(out_o, data.y)
        stab = hk_stability_loss(out_o, out_p, data.topo, data.topo_pert, data.batch,
                                 topo_dim=topo_dim, L_pi=L_pi, lam_kld=lam_kld)
        loss = ce + 0.3 * stab

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_acc(model, loader, device, drop_edges=False, p=0.1):
    model.eval()
    correct = tot = 0
    for data in loader:
        data = data.to(device)
        ei = data.edge_index
        if drop_edges:
            ei, _ = dropout_edge(ei, p=p, force_undirected=True, training=True)
        out = model(data.x, ei, data.batch, data.topo)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        tot += data.num_graphs
    return correct / tot


@torch.no_grad()
def eval_cert(model, loader, device, topo_dim, L_pi=1.0):
    model.eval()
    vals = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.topo)
        vals.append(certify_radius(out,
                                   data.topo, data.topo_pert,
                                   L_pi=L_pi, batch=data.batch, topo_dim=topo_dim))
    return float(np.mean(vals)) if len(vals) else 0.0


# ---------------
# 5) Main script
# ---------------
if __name__ == "__main__":
    SEED = 17  # change to try other stratified splits
    L_PI = 1.0  # Lipschitz factor used in the inequality loss
    LAM_KLD = 0.0  # set >0 to add symmetric KL drift term (optional)
    EPOCHS = 50
    DATASET_NAME = "NCI1"  # Change to any TUDataset name

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds_full = HKGraphDataset(
        root=f"data/{DATASET_NAME}", name=DATASET_NAME,
        r0=0.4, r1=1.2, res=10, sigma=0.1,
        max_deg=50, p_edge_pert=0.05
    )

    # Stratified split (80% train, 20% test)
    train_ds, test_ds = stratified_split(ds_full, train_ratio=0.8, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    topo_dim = train_ds[0].topo.numel()  # = 2*(res^2) = 200 if res=10
    in_dim = train_ds.num_node_features  # node feature dimension
    n_cls = ds_full.num_classes
    print(f"Dataset: {DATASET_NAME}, in_dim={in_dim}, topo_dim={topo_dim}, num_classes={n_cls}")

    model = TopoGIN_HK(in_dim=in_dim, hidden_dim=64, num_classes=n_cls, topo_dim=topo_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = 0.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, opt, device, topo_dim, L_pi=L_PI, lam_kld=LAM_KLD)
        if epoch % 5 == 0:
            clean = eval_acc(model, test_loader, device, drop_edges=False)
            noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
            cert = eval_cert(model, test_loader, device, topo_dim=topo_dim, L_pi=L_PI)
            best = max(best, clean)
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Clean {clean:.3f} | Noisy {noisy:.3f} "
                  f"| Drop {clean - noisy:.3f} | CertRadius~ {cert:.4f}")

    print(f"\nDone in {(time.time() - t0):.1f}s")
    final_clean = eval_acc(model, test_loader, device, drop_edges=False)
    final_noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
    final_cert = eval_cert(model, test_loader, device, topo_dim=topo_dim, L_pi=L_PI)

    print(f"\nFinal → Clean: {final_clean:.3f} | Noisy: {final_noisy:.3f} | Drop: {final_clean - final_noisy:.3f}")
    print(f"Best Clean: {best:.3f} | Avg Certified Radius (proxy): {final_cert:.4f}")