# baselines_imdb.py
import os, time, math, statistics as stats, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, TransformerConv,
    global_add_pool, global_max_pool
)
from torch_geometric.utils import degree, dropout_edge
from torch_geometric.loader import DataLoader

# Your locally processed dataset wrapper:
from process_networkrepo import NetworkRepoDataset


# ---------- Utils ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_degree_onehot_features(ds, max_deg=50):
    # In-place: attach one-hot degree features to every graph
    for data in ds:
        deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg = torch.clamp(deg, max=max_deg)
        data.x = F.one_hot(deg, num_classes=max_deg + 1).float()
    return ds


def stratified_split_indices(labels, train_per_class=400, test_per_class=100, rng: random.Random = None):
    # IMDB-BINARY has ~500 graphs per class, so 400 train + 100 test per class uses all.
    if rng is None:
        rng = random
    labels = torch.as_tensor(labels).view(-1).tolist()
    per_class_idx = {}
    for i, y in enumerate(labels):
        per_class_idx.setdefault(int(y), []).append(i)
    train_idx, test_idx = [], []
    for y, idxs in per_class_idx.items():
        rng.shuffle(idxs)
        assert len(idxs) >= (train_per_class + test_per_class), \
            f"Class {y} has only {len(idxs)} samples."
        train_idx += idxs[:train_per_class]
        test_idx  += idxs[train_per_class:train_per_class + test_per_class]
    return sorted(train_idx), sorted(test_idx)


@torch.no_grad()
def evaluate(model, loader, device, drop_p: float = 0.0):
    model.eval()
    correct, total = 0, 0
    for data in loader:
        data = data.to(device)
        ei = data.edge_index
        if drop_p > 0:
            ei, _ = dropout_edge(ei, p=drop_p, force_undirected=True, training=True)
        out = model(data.x, ei, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return correct / total


# ---------- Models ----------
class Readout(nn.Module):
    def forward(self, x, batch):
        g_sum = global_add_pool(x, batch)
        g_max = global_max_pool(x, batch)
        return torch.cat([g_sum, g_max], dim=1)  # concat sum+max


class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.read = Readout()
        self.lin = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        g = self.read(x, batch)
        return self.lin(g)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.read = Readout()
        self.lin = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        g = self.read(x, batch)
        return self.lin(g)


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=2, heads1=8):
        super().__init__()
        # first layer expands to hidden via multi-head concat
        self.gat1 = GATConv(in_dim, hidden // 2, heads=heads1, concat=True)  # (hidden//2)*heads1 ≈ hidden
        out1 = (hidden // 2) * heads1
        self.gat2 = GATConv(out1, hidden, heads=1, concat=False)
        self.read = Readout()
        self.lin = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        g = self.read(x, batch)
        return self.lin(g)


class TransformerConvModel(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=2, heads=4):
        super().__init__()
        self.t1 = TransformerConv(in_dim, hidden // heads, heads=heads, dropout=0.2, beta=False)
        out1 = (hidden // heads) * heads
        self.t2 = TransformerConv(out1, hidden // heads, heads=heads, dropout=0.2, beta=False)
        out2 = (hidden // heads) * heads
        assert out2 == hidden
        self.read = Readout()
        self.lin = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.t1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.t2(x, edge_index))
        g = self.read(x, batch)
        return self.lin(g)


# ---------- Train one run ----------
def run_once(model_name, seed, device, epochs=50, batch_size=64, lr=1e-3, wd=1e-4,
             train_per_class=400, test_per_class=100, max_deg=50, drop_p_eval=0.1):
    set_seed(seed)

    # Dataset (locally processed IMDB-BINARY)
    ds = NetworkRepoDataset(root="data/IMDB-BINARY", name="IMDB-BINARY")
    ds = build_degree_onehot_features(ds, max_deg=max_deg)

    # Stratified split (uses all graphs: 400+100 per class)
    labels = [int(ds[i].y.item()) for i in range(len(ds))]
    rng = random.Random(seed)
    train_idx, test_idx = stratified_split_indices(labels, train_per_class, test_per_class, rng)
    train_ds = ds[train_idx]
    test_ds  = ds[test_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    in_dim = train_ds.num_node_features
    num_classes = ds.num_classes

    # Pick model
    if model_name == "GCN":
        model = GCNModel(in_dim, hidden=64, num_classes=num_classes)
    elif model_name == "GraphSAGE":
        model = GraphSAGEModel(in_dim, hidden=64, num_classes=num_classes)
    elif model_name == "GAT":
        model = GATModel(in_dim, hidden=64, num_classes=num_classes, heads1=8)
    elif model_name == "Transformer":
        model = TransformerConvModel(in_dim, hidden=64, num_classes=num_classes, heads=4)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)

    # Train
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * data.num_graphs
        train_loss = total / len(train_ds)

        if epoch % 5 == 0 or epoch == epochs:
            acc = evaluate(model, test_loader, device, drop_p=0.0)
            acc_noisy = evaluate(model, test_loader, device, drop_p=drop_p_eval)
            best = max(best, acc)
            sched.step(acc)
            print(f"[{model_name}] Seed {seed} | Ep {epoch:03d} | "
                  f"Loss {train_loss:.4f} | Clean {acc:.3f} | Noisy {acc_noisy:.3f} | Drop {acc-acc_noisy:.3f}")

    clean = evaluate(model, test_loader, device, drop_p=0.0)
    noisy = evaluate(model, test_loader, device, drop_p=drop_p_eval)
    return clean, noisy, best


# ---------- Multi-seed runner ----------
def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = ["GCN", "GraphSAGE", "GAT", "Transformer"]
    seeds = [13, 17, 42, 123, 777]   # change/add as you like
    epochs = 50

    results = {m: {"clean": [], "noisy": [], "best": []} for m in models}

    t0 = time.time()
    for m in models:
        print(f"\n=== {m} ===")
        for s in seeds:
            clean, noisy, best = run_once(
                m, s, device,
                epochs=epochs, batch_size=64, lr=1e-3, wd=1e-4,
                train_per_class=400, test_per_class=100, max_deg=50, drop_p_eval=0.10
            )
            results[m]["clean"].append(clean)
            results[m]["noisy"].append(noisy)
            results[m]["best"].append(best)

    print(f"\nFinished in {time.time()-t0:.1f}s\n")

    # Summaries
    def msd(x): return (stats.mean(x), stats.pstdev(x))
    for m in models:
        mc, sc = msd(results[m]["clean"])
        mn, sn = msd(results[m]["noisy"])
        mb, sb = msd(results[m]["best"])
        print(f"{m:12s} | Clean {mc:.3f}±{sc:.3f} | Noisy {mn:.3f}±{sn:.3f} | Best {mb:.3f}±{sb:.3f} | "
              f"Drop {mc-mn:+.3f}")

if __name__ == "__main__":
    run_all()
