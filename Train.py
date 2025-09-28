import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# import your model classes
from models import GCN, GAT, StabilityGNN

def train_model(model, train_loader, optimizer, epochs=50, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)  # cross-entropy for classification
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # (Optional) print loss per epoch or other metrics if needed
        # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    model.eval()  # set to eval mode for evaluation

def evaluate_model(model, loader, device='cpu', perturb=False):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        if perturb:
            # Create a perturbed edge_index by dropping 10% edges for each graph in the batch
            # We will manually drop edges for each graph in the batch.
            # Since DataLoader batches multiple graphs, we drop edges per graph, then merge.
            # Simpler approach: drop edges from the whole batch edge_index (treating it as one graph),
            # which approximates dropping edges across all graphs uniformly.
            edge_index, _ = dropout_edge(data.edge_index, p=0.1, force_undirected=True, training=True)
            out = model(data.x, edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index, data.batch)
        # Predicted class is the index with max logit
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs  # number of graphs in this batch
    return correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}  # to store accuracy results
for name, loaders_dict in loaders.items():
    train_loader = loaders_dict["train"]
    test_loader = loaders_dict["test"]
    print(f"\n** Dataset: {name} **")
    results[name] = {}
    # Define model instances
    gcn = GCN(num_features=train_loader.dataset.num_features, hidden_dim=64, num_classes=train_loader.dataset.num_classes)
    gat = GAT(num_features=train_loader.dataset.num_features, hidden_dim=64, num_classes=train_loader.dataset.num_classes)
    stability_gnn = StabilityGNN(num_features=train_loader.dataset.num_features, hidden_dim=64, num_classes=train_loader.dataset.num_classes)
    # Optimizers for each
    optim_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01)
    optim_gat = torch.optim.Adam(gat.parameters(), lr=0.01)
    optim_stab = torch.optim.Adam(stability_gnn.parameters(), lr=0.01)
    # Train each model
    train_model(gcn, train_loader, optim_gcn, epochs=50, device=device)
    train_model(gat, train_loader, optim_gat, epochs=50, device=device)
    train_model(stability_gnn, train_loader, optim_stab, epochs=50, device=device)
    # Evaluate on clean test data
    acc_clean_gcn = evaluate_model(gcn, test_loader, device=device, perturb=False)
    acc_clean_gat = evaluate_model(gat, test_loader, device=device, perturb=False)
    acc_clean_stab = evaluate_model(stability_gnn, test_loader, device=device, perturb=False)
    # Evaluate on perturbed test data (10% edges removed)
    acc_pert_gcn = evaluate_model(gcn, test_loader, device=device, perturb=True)
    acc_pert_gat = evaluate_model(gat, test_loader, device=device, perturb=True)
    acc_pert_stab = evaluate_model(stability_gnn, test_loader, device=device, perturb=True)
    # Calculate accuracy drops
    drop_gcn = acc_clean_gcn - acc_pert_gcn
    drop_gat = acc_clean_gat - acc_pert_gat
    drop_stab = acc_clean_stab - acc_pert_stab
    # Store and print results
    results[name]['GCN'] = (acc_clean_gcn, acc_pert_gcn, drop_gcn)
    results[name]['GAT'] = (acc_clean_gat, acc_pert_gat, drop_gat)
    results[name]['StabilityGNN'] = (acc_clean_stab, acc_pert_stab, drop_stab)
    print(f"GCN – Test Acc: {acc_clean_gcn*100:.2f}%, Perturbed Acc: {acc_pert_gcn*100:.2f}%, Drop: {drop_gcn*100:.2f}%")
    print(f"GAT – Test Acc: {acc_clean_gat*100:.2f}%, Perturbed Acc: {acc_pert_gat*100:.2f}%, Drop: {drop_gat*100:.2f}%")
    print(f"StabilityGNN – Test Acc: {acc_clean_stab*100:.2f}%, Perturbed Acc: {acc_pert_stab*100:.2f}%, Drop: {drop_stab*100:.2f}%")
