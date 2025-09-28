# gcn_imdb_onehot_fixed.py

import torch
import torch.nn.functional as F
import networkx as nx
import time

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, dropout_edge, to_networkx
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool

from process_networkrepo import NetworkRepoDataset

# 1) Load the locally‐processed IMDB‑BINARY
ds = NetworkRepoDataset(root="data/IMDB-BINARY", name="IMDB-BINARY")

# 2) Enhanced feature engineering
max_deg = 100
print("Computing graph features...")
start_time = time.time()

for data in ds:
    # Original one-hot degree features
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    deg = torch.clamp(deg, max=max_deg)
    data.x = F.one_hot(deg, num_classes=max_deg + 1).float()

    # Add graph-level structural features to all nodes
    G = to_networkx(data, to_undirected=True)

    # Edge count (scaled)
    edge_count = data.edge_index.shape[1] / 100.0

    # Average clustering coefficient
    clust_coeffs = list(nx.clustering(G).values())
    avg_clust = sum(clust_coeffs) / len(clust_coeffs) if clust_coeffs else 0

    # Create additional feature tensor
    extra_feats = torch.tensor([[edge_count, avg_clust]] * data.num_nodes, dtype=torch.float)

    # Append to existing features
    data.x = torch.cat([data.x, extra_feats], dim=1)

print(f"Feature engineering completed in {time.time() - start_time:.2f}s")
print(f"New feature dimension: {ds[0].x.shape[1]}")

# 3) Train/test split and loaders
ds = ds.shuffle()
split = int(0.8 * len(ds))
train_ds, test_ds = ds[:split], ds[split:]
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)


# 4) Define GIN (Graph Isomorphism Network)
class GIN(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_feats, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        )
        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU()
            )
        )
        self.lin = torch.nn.Linear(hidden_dim * 2, num_classes)  # Double features for concat pooling

    def forward(self, x, edge_index, batch):
        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = F.dropout(x1, p=0.5, training=self.training)

        # Layer 2
        x2 = self.conv2(x1, edge_index)
        x2 = F.dropout(x2, p=0.5, training=self.training)

        # Layer 3
        x3 = self.conv3(x2, edge_index)

        # Concatenated pooling (sum + max)
        g_sum = global_add_pool(x3, batch)
        g_max = global_max_pool(x3, batch)
        g = torch.cat([g_sum, g_max], dim=1)

        return self.lin(g)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GIN(
    in_feats=ds.num_node_features,
    hidden_dim=64,
    num_classes=ds.num_classes
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# 5) Enhanced training & evaluation
def train_one():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        opt.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_ds)


@torch.no_grad()
def test(loader, noisy=False):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        ei = data.edge_index
        if noisy:
            ei, _ = dropout_edge(ei, p=0.1, force_undirected=True)
        out = model(data.x, ei, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return correct / total


# Training loop with validation monitoring
best_test_acc = 0
for epoch in range(1, 101):
    loss = train_one()

    if epoch % 5 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        scheduler.step(test_acc)  # Update learning rate

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
              f"LR: {opt.param_groups[0]['lr']:.6f}")

# Load best model for final evaluation
model.load_state_dict(torch.load('best_model.pth'))
clean_acc = test(test_loader, noisy=False)
noisy_acc = test(test_loader, noisy=True)

print(f"\nFinal Results → Clean Acc: {clean_acc:.3f}, "
      f"Noisy Acc: {noisy_acc:.3f}, "
      f"Drop: {clean_acc - noisy_acc:.3f}")
print(f"Best Test Accuracy: {best_test_acc:.3f}")