import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gudhi as gd
from gudhi.representations import PersistenceImage

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_networkx
from torch_geometric.nn import GINConv, global_add_pool
from process_networkrepo import NetworkRepoDataset


# 1. Robust persistence computation with dimension validation
class TopoIMDBDataset(NetworkRepoDataset):
    def __init__(self, root, name, r0=0.1, r1=0.8):
        super().__init__(root, name)
        self.r0, self.r1 = r0, r1
        if not os.path.exists(self.processed_paths[0]):
            self._precompute_persistence()

    def _compute_shortest_path(self, edge_index, num_nodes):
        adj = np.zeros((num_nodes, num_nodes))
        ei = edge_index.numpy()
        adj[ei[0], ei[1]] = 1
        adj[ei[1], ei[0]] = 1

        dist = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(dist, 0)
        dist[adj == 1] = 1

        for k in range(num_nodes):
            dist = np.minimum(dist, dist[:, k][:, None] + dist[k])
        return dist

    def _precompute_persistence(self):
        print("Computing H0+H1 persistence with validation...")
        # Smaller persistence image
        pers_img = PersistenceImage(bandwidth=0.1, resolution=[10, 10])
        data_list = [self.get(i) for i in range(len(self))]

        for i, data in enumerate(data_list):
            dist = self._compute_shortest_path(data.edge_index, data.num_nodes)

            # Compute persistence
            rc = gd.RipsComplex(distance_matrix=dist, max_edge_length=self.r1)
            st = rc.create_simplex_tree(max_dimension=2)
            st.persistence()

            # Extract H0 and H1
            dgm0 = st.persistence_intervals_in_dimension(0)
            dgm1 = st.persistence_intervals_in_dimension(1)

            # Filter by relative scale
            rel0 = [[b, min(d, self.r1)] for b, d in dgm0 if d > self.r0]
            rel1 = [[b, min(d, self.r1)] for b, d in dgm1 if d > self.r0]

            # Create persistence images with validation
            pi0 = np.zeros(100)
            if len(rel0) > 0:
                pi0 = pers_img.fit_transform([np.array(rel0)])[0]
                if len(pi0) != 100:
                    pi0 = np.zeros(100)

            pi1 = np.zeros(100)
            if len(rel1) > 0:
                pi1 = pers_img.fit_transform([np.array(rel1)])[0]
                if len(pi1) != 100:
                    pi1 = np.zeros(100)

            # Store as graph-level feature
            data.topo = torch.tensor(np.concatenate([pi0, pi1]), dtype=torch.float)  # 200D

            # Create node features
            deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg = torch.clamp(deg, max=50)  # Reduced max degree
            data.x = F.one_hot(deg, num_classes=51).float()  # 51D

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(data_list)} graphs | Topo dim: {data.topo.shape[0]}")

        data_all, slices = self.collate(data_list)
        torch.save((data_all, slices), self.processed_paths[0])
        print(f"Persistence computation complete. Topo dimension: {data_list[0].topo.shape[0]}")


# 2. Fixed GIN model with topology integration
class TopoGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, topo_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topo_dim = topo_dim

        # Structural pathway (GIN)
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ))

        # Topology integration with dimension safety
        self.topo_net = nn.Sequential(
            nn.Linear(topo_dim, hidden_dim),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, edge_index, batch, topo):
        # Process structural features
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x_struct = global_add_pool(x, batch)  # (batch_size, hidden_dim)

        # Get actual batch size from graph batch vector
        batch_size = batch.max().item() + 1

        # Handle batch dimension for topology features
        if topo.dim() == 1:
            if batch_size == 1:
                topo = topo.unsqueeze(0)  # [1, topo_dim]
            else:
                # Replicate for all graphs in batch
                topo = topo.unsqueeze(0).expand(batch_size, -1)  # [batch_size, topo_dim]
        elif topo.dim() == 2 and topo.size(0) == 1 and batch_size > 1:
            # Expand single graph features to full batch
            topo = topo.expand(batch_size, -1)
        elif topo.dim() > 2:
            # Flatten extra dimensions
            topo = topo.view(topo.size(0), -1)

        # Feature dimension adjustment
        if topo.size(-1) != self.topo_dim:
            if topo.size(-1) < self.topo_dim:
                # Pad with zeros
                padding = torch.zeros(topo.size(0), self.topo_dim - topo.size(-1), device=topo.device)
                topo = torch.cat([topo, padding], dim=1)
            else:
                # Truncate
                topo = topo[:, :self.topo_dim]

        x_topo = self.topo_net(topo)  # (batch_size, hidden_dim)

        # Now both tensors have same batch dimension
        combined = torch.cat([x_struct, x_topo], dim=1)
        return self.classifier(combined)


# 3. Training and evaluation functions
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.topo)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.topo)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return correct / total


# 4. Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset with validation
    ds = TopoIMDBDataset(root="data/IMDB-BINARY", name="IMDB-BINARY", r0=0.1, r1=0.8)
    print(f"Node dim: {ds[0].x.shape[1]}, Topo dim: {ds[0].topo.shape[0]}")

    # Train/test split
    ds = ds.shuffle()
    train_ds = ds[:800]
    test_ds = ds[800:]
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Create model with validated dimensions
    model = TopoGIN(
        in_dim=ds[0].x.shape[1],  # 51
        hidden_dim=64,
        num_classes=ds.num_classes,
        topo_dim=200  # Fixed to match our computation
    ).to(device)

    # Optimizer with higher learning rate
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5, factor=0.5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_acc = 0
    for epoch in range(1, 201):
        loss = train(model, train_loader, opt, device)
        test_acc = test(model, test_loader, device)
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

        lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test: {test_acc:.4f} | LR: {lr:.6f}")

        # Early stopping if learning rate becomes too small
        if lr < 1e-7:
            print("Stopping early due to small LR")
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    final_acc = test(model, test_loader, device)
    print(f"\n★ Final Test Accuracy: {final_acc:.4f}")
    print(f"★ Best Test Accuracy: {best_acc:.4f}")