# hk_stability_imdb.py
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

from process_networkrepo import NetworkRepoDataset


# -------------------------------
# 1) Dataset wrapper (HK-inspired)
# -------------------------------
class HKIMDBDataset(NetworkRepoDataset):
    """
    Builds per-graph node features (one-hot degree) and HK-style
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
        print("⏳ Attaching node features + HK-style windowed persistence images…")
        start = time.time()
        data_list = []
        for i in range(len(self)):
            d = self.get(i)

            # Node features
            d.x = self._onehot_degree(d)

            # Topo features (original)
            dist = self._shortest_path_matrix(d.edge_index, d.num_nodes)
            topo = self._persistence_image_H0H1(dist, self.r0, self.r1)
            d.topo = torch.tensor(topo, dtype=torch.float)

            # Topo features (perturbed)
            ei_pert = self._perturb_edge_index(d, p=self.p_edge_pert)
            dist_p = self._shortest_path_matrix(ei_pert, d.num_nodes)
            topo_p = self._persistence_image_H0H1(dist_p, self.r0, self.r1)
            d.topo_pert = torch.tensor(topo_p, dtype=torch.float)

            data_list.append(d)
            if (i + 1) % 50 == 0 or (i + 1) == len(self):
                print(f"  • {i + 1}/{len(self)} done")

        # Replace in-memory storage with enriched data:
        self.data, self.slices = self.collate(data_list)
        print(f"✅ Attached in {time.time() - start:.1f}s | topo_dim={2*(self.res*self.res)}")


# ----------------------------------
# 2) Model: GIN + topo fusion (HK-ish)
# ----------------------------------
class TopoGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, topo_dim):
        super().__init__()
        self.topo_dim = topo_dim
        self.hidden = hidden_dim

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

        self.topo_net = nn.Sequential(
            nn.Linear(topo_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def _reshape_topo(self, topo_vec, batch, topo_dim):
        """Turn concatenated 1-D topo (length B*D) back into (B, D)."""
        B = int(batch.max().item()) + 1
        if topo_vec.dim() == 1:
            # Most common case with PyG batching
            if topo_vec.numel() == B * topo_dim:
                return topo_vec.view(B, topo_dim)
            elif topo_vec.numel() == topo_dim:
                return topo_vec.view(1, topo_dim).expand(B, -1)
            else:
                # Fallback: try (B, -1)
                return topo_vec.view(B, -1)
        elif topo_vec.dim() == 2:
            if topo_vec.size(0) == B:
                return topo_vec
            if topo_vec.size(0) == 1 and B > 1:
                return topo_vec.expand(B, -1)
            if topo_vec.numel() == B * topo_dim:
                return topo_vec.reshape(B, topo_dim)
        # Last resort: pad/truncate to (B, topo_dim)
        flat = topo_vec.view(-1)
        if flat.numel() < B * topo_dim:
            pad = flat.new_zeros(B * topo_dim - flat.numel())
            flat = torch.cat([flat, pad], dim=0)
        return flat[: B * topo_dim].view(B, topo_dim)

    def forward(self, x, edge_index, batch, topo_vec):
        # Structural
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        g_struct = global_add_pool(h, batch)  # (B, hidden)

        # Topological (robust reshape)
        topo_vec = self._reshape_topo(topo_vec, batch, self.topo_dim)  # (B, topo_dim)
        g_topo = self.topo_net(topo_vec)                               # (B, hidden)

        # Fuse
        g = torch.cat([g_struct, g_topo], dim=1)  # (B, 2*hidden)
        return self.classifier(g)


# ----------------------------------
# 3) Stability loss & certification
# ----------------------------------
def stability_regularizer(logits_orig, logits_pert, topo, topo_pert, lam_pred=1.0, lam_pi=0.05):
    # Symmetric KL on predictions
    p = F.log_softmax(logits_orig, dim=1)
    q = F.log_softmax(logits_pert, dim=1)
    pred_drift = (p.exp() * (p - q)).sum(dim=1).mean() + (q.exp() * (q - p)).sum(dim=1).mean()

    # L1 distance on persistence images
    # (reshape both back to (B, D) so batches match)
    def reshape_any(t, batch, topo_dim):
        B = int(batch.max().item()) + 1
        if t.dim() == 1 and t.numel() == B * topo_dim:
            return t.view(B, topo_dim)
        if t.dim() == 2 and t.size(0) == B:
            return t
        return t.view(B, -1)
    topo = reshape_any(topo,  torch.arange(len(topo), device=topo.device), topo.size(-1) if topo.dim()==2 else topo.numel())
    topo_pert = reshape_any(topo_pert, torch.arange(len(topo_pert), device=topo_pert.device),
                            topo_pert.size(-1) if topo_pert.dim()==2 else topo_pert.numel())

    pi_drift = (topo - topo_pert).abs().mean()
    return lam_pred * pred_drift + lam_pi * pi_drift


@torch.no_grad()
def certify_radius(logits, topo, topo_pert, L=1.0):
    probs = F.softmax(logits, dim=1)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])  # (B,)
    d_pi = (topo - topo_pert).abs().sum(dim=1).clamp(min=1e-8)
    return (margin / (L * d_pi)).mean().item()


# --------------------------
# 4) Train / evaluate loops
# --------------------------
def train_epoch(model, loader, opt, device, topo_dim):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.topo)
        out_pert = model(data.x, data.edge_index, data.batch, data.topo_pert)

        ce = F.cross_entropy(out, data.y)
        # use simple logits drift + small PI drift
        stab = stability_regularizer(out, out_pert, data.topo, data.topo_pert, lam_pred=1.0, lam_pi=0.05)
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
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        ei = data.edge_index
        if drop_edges:
            ei, _ = dropout_edge(ei, p=p, force_undirected=True, training=True)
        out = model(data.x, ei, data.batch, data.topo)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return correct / total


@torch.no_grad()
def eval_cert(model, loader, device):
    model.eval()
    vals = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.topo)
        # reshape to (B, D) just for radius calc
        B = int(data.batch.max().item()) + 1
        topo = data.topo.view(B, -1) if data.topo.dim() == 1 else data.topo
        topo_pert = data.topo_pert.view(B, -1) if data.topo_pert.dim() == 1 else data.topo_pert
        vals.append(certify_radius(out, topo, topo_pert, L=1.0))
    return float(np.mean(vals)) if len(vals) else 0.0


# ---------------
# 5) Main script
# ---------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = HKIMDBDataset(
        root="data/IMDB-BINARY", name="IMDB-BINARY",
        r0=0.4, r1=1.2, res=10, sigma=0.1,
        max_deg=50, p_edge_pert=0.05
    )

    ds = ds.shuffle()
    split = int(0.8 * len(ds))
    train_ds, test_ds = ds[:split], ds[split:]
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64)

    topo_dim = train_ds[0].topo.numel()  # = 2*(res^2)
    in_dim   = train_ds.num_node_features
    n_cls    = ds.num_classes
    print(f"in_dim={in_dim}, topo_dim={topo_dim}, num_classes={n_cls}")

    model = TopoGIN(in_dim=in_dim, hidden_dim=64, num_classes=n_cls, topo_dim=topo_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = 0.0
    t0 = time.time()
    for epoch in range(1, 51):
        loss = train_epoch(model, train_loader, opt, device, topo_dim)
        if epoch % 5 == 0:
            clean = eval_acc(model, test_loader, device, drop_edges=False)
            noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
            cert  = eval_cert(model, test_loader, device)
            best  = max(best, clean)
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Clean {clean:.3f} | Noisy {noisy:.3f} | "
                  f"Drop {clean-noisy:.3f} | CertRadius~ {cert:.4f}")

    print(f"\nDone in {(time.time()-t0):.1f}s")
    final_clean = eval_acc(model, test_loader, device, drop_edges=False)
    final_noisy = eval_acc(model, test_loader, device, drop_edges=True, p=0.1)
    final_cert  = eval_cert(model, test_loader, device)

    print(f"\nFinal → Clean: {final_clean:.3f} | Noisy: {final_noisy:.3f} | Drop: {final_clean-final_noisy:.3f}")
    print(f"Best Clean: {best:.3f} | Avg Certified Radius (proxy): {final_cert:.4f}")
