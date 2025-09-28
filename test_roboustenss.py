# test_robustness.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge

from process_networkrepo import NetworkRepoDataset
from models import GCN, GAT, StabilityGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, loader, optimizer, epochs=40):
    model.to(device)
    for _ in range(epochs):
        model.train()
        for data in loader:
            data = data.to(device)
            # ----- RESHAPE TOPO -----
            # data.topo comes in as a 1-D vector of length batch_size * topo_dim
            B = data.num_graphs
            total_elems = data.topo.numel()
            topo_dim = total_elems // B
            data.topo = data.topo.view(B, topo_dim)
            # ------------------------

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, data.topo)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

@torch.no_grad()
def evaluate(model, loader, perturb=False):
    model.to(device).eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        # reshape topo exactly as in training
        B = data.num_graphs
        total_elems = data.topo.numel()
        topo_dim = total_elems // B
        data.topo = data.topo.view(B, topo_dim)

        ei = data.edge_index
        if perturb:
            ei, _ = dropout_edge(ei, p=0.1,
                                 force_undirected=True, training=True)
        out = model(data.x, ei, data.batch, data.topo)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total   += data.num_graphs
    return correct / total

if __name__ == "__main__":
    print("=== IMDB-BINARY ===")
    ds = NetworkRepoDataset(root="data/IMDB-BINARY", name="IMDB-BINARY")
    ds = ds.shuffle()
    split = int(0.8 * len(ds))
    train_ds, test_ds = ds[:split], ds[split:]
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32)

    in_feats  = ds.num_node_features
    n_classes = ds.num_classes
    # get topo_dim from a single graph
    sample = ds[0]
    topo_dim = sample.topo.numel()

    # instantiate
    gcn  = GCN(in_feats,   64, n_classes).to(device)
    gat  = GAT(in_feats,   64, n_classes).to(device)
    stab = StabilityGNN(in_feats, 64, n_classes, topo_dim).to(device)

    opt_gcn  = torch.optim.Adam(gcn.parameters(),  lr=1e-2)
    opt_gat  = torch.optim.Adam(gat.parameters(),  lr=1e-2)
    opt_stab = torch.optim.Adam(stab.parameters(), lr=1e-2)

    # train
    train(gcn,  train_loader, opt_gcn)
    train(gat,  train_loader, opt_gat)
    train(stab, train_loader, opt_stab)

    # evaluate
    for name, model in [("GCN", gcn), ("GAT", gat), ("Stab", stab)]:
        a_clean = evaluate(model, test_loader, perturb=False)
        a_noisy = evaluate(model, test_loader, perturb=True)
        print(f"{name:4s} clean={a_clean:.3f} noisy={a_noisy:.3f} drop={a_clean-a_noisy:.3f}")
