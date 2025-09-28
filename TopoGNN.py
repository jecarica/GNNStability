import torch
import sklearn
import torch.nn as nn
import torch.nn.functional as F

# (Assume Gudhi is installed for persistent homology computations)
from gudhi import RipsComplex
from gudhi.representations import PersistenceImage


class StabilityGNN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes,
                 r0=0.5, r1=1.0, pers_img_res=20, pers_img_sigma=0.1):
        """
        GNN with stability-preserving topology integration.
        :param in_feats: Dimension of input node features.
        :param hidden_dim: Hidden dimension for node embeddings.
        :param num_classes: Number of output classes.
        :param r0: Initial lower filtration radius for relative homology.
        :param r1: Initial upper filtration radius for relative homology.
        :param pers_img_res: Resolution of persistence image (square grid size).
        :param pers_img_sigma: Gaussian kernel sigma for persistence image.
        """
        super(StabilityGNN, self).__init__()
        # GNN layers (using a simple 1-layer GAT for illustration)
        self.W = nn.Linear(in_feats, hidden_dim, bias=False)  # feature transform
        self.att_vec = nn.Parameter(torch.randn(hidden_dim * 2 + pers_img_res ** 2))
        # ^ attention weight vector 'a' concatenating [Wh_i || Wh_j || topo_emb]
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.out_linear = nn.Linear(hidden_dim, num_classes)

        # Topology-related parameters
        self.r0 = r0
        self.r1 = r1
        # Initialize persistence image transformer
        self.pers_img = PersistenceImage(bandwidth=pers_img_sigma,
                                         resolution=[pers_img_res, pers_img_res])
        # Lipschitz constant (assume we can estimate or enforce this; set 1.0 if 1-Lipschitz network)
        self.Lipschitz_const = 1.0

    def compute_persistence_embedding(self, adj_matrix):
        """
        Compute the persistence image embedding for the graph described by adj_matrix.
        adj_matrix: N x N (numpy or torch) adjacency (with weights or 1 for edges, inf or 0 for no edge).
        Returns: torch.Tensor of shape (pers_img_res**2,) representing the persistence image.
        """
        # Compute shortest-path distance matrix (use Floyd-Warshall or BFS since graph may be unweighted)
        # Here we'll assume adj_matrix is a numpy array for simplicity.
        import numpy as np
        N = adj_matrix.shape[0]
        # Initialize distance matrix
        dist = np.full((N, N), np.inf)
        np.fill_diagonal(dist, 0.0)
        # Use adjacency to set initial distances
        for i in range(N):
            for j in range(N):
                if adj_matrix[i, j] > 0:  # assume >0 means an edge with that weight
                    dist[i, j] = min(dist[i, j], adj_matrix[i, j])
        # Floyd-Warshall algorithm for all-pairs shortest paths
        for k in range(N):
            for i in range(N):
                # small optimization: skip if dist[i,k] is inf
                if dist[i, k] == np.inf:
                    continue
                for j in range(N):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # Build Vietoris-Rips complex up to r1
        rips = RipsComplex(distance_matrix=dist, max_edge_length=self.r1)
        st = rips.create_simplex_tree(max_dimension=2)
        persistence = st.persistence()  # compute persistence for all dimensions
        # Extract H1 persistence intervals (loops)
        H1_intervals = st.persistence_intervals_in_dimension(1)
        # Also build Rips complex up to r0 for relative reference
        rips0 = RipsComplex(distance_matrix=dist, max_edge_length=self.r0)
        st0 = rips0.create_simplex_tree(max_dimension=2)
        st0.persistence()  # compute persistence up to r0 (not used directly, but ensures complex built)
        # Now construct relative persistence intervals:
        rel_intervals = []
        for (birth, death) in H1_intervals:
            # Clamp the interval to [r0, r1]
            if death <= self.r0 or birth >= self.r1:
                continue  # interval lies completely outside [r0, r1]
            b_rel = max(birth, self.r0)
            d_rel = min(death, self.r1)
            if b_rel < d_rel:
                rel_intervals.append([b_rel, d_rel])
        # Compute persistence image for the relative intervals (as a list of diagrams of one graph)
        if len(rel_intervals) == 0:
            # If no features, we can return a zero vector
            pi_vector = np.zeros(self.pers_img.resolution[0] * self.pers_img.resolution[1])
        else:
            pi_vector = self.pers_img.fit_transform([np.array(rel_intervals)])[0]
        # Convert to torch tensor
        pi_tensor = torch.tensor(pi_vector, dtype=torch.float32)
        return pi_tensor

    def forward(self, node_features, adj_matrix):
        """
        Forward pass for graph classification.
        :param node_features: Tensor of shape (N, in_feats) for N nodes.
        :param adj_matrix: Adjacency matrix (NxN, numpy or tensor) of the graph.
        """
        N = node_features.size(0)
        # 1. Compute topology embedding (persistence image) for the graph
        with torch.no_grad():  # persistence computation is not differentiable, treat as constant for backprop
            topo_emb = self.compute_persistence_embedding(adj_matrix)
        # If needed, project topo_emb to same device and dtype as model
        topo_emb = topo_emb.to(node_features.device)

        # 2. Compute initial node embeddings via linear projection
        h = self.W(node_features)  # shape (N, hidden_dim)
        # 3. Compute attention coefficients for each pair of connected nodes
        # We'll use the adjacency to get edges; for efficiency assume adjacency is small or use sparse operations
        att_scores = torch.zeros((N, N), device=node_features.device)
        for i in range(N):
            for j in range(N):
                if adj_matrix[i, j] == 0 or i == j:
                    continue  # no direct edge (we could also attend to non-neighbors if desired in a fully-connected graph transformer)
                # Concatenate [h_i || h_j || topo_emb]
                concat_feat = torch.cat([h[i], h[j], topo_emb], dim=0)
                e_ij = torch.dot(self.att_vec, concat_feat)  # a^T [Wh_i || Wh_j || z_topo]
                att_scores[i, j] = self.leaky_relu(e_ij)
        # 4. Normalize attention scores (softmax) for each target node i
        att_weights = F.softmax(att_scores, dim=1)  # softmax over j for each i
        # 5. Message passing: aggregate neighbors' messages weighted by attention
        h_out = torch.zeros_like(h)
        for i in range(N):
            # aggregate from all j
            agg = torch.zeros_like(h[i])
            for j in range(N):
                if adj_matrix[i, j] == 0 or i == j:
                    continue
                agg += att_weights[i, j] * h[j]
            # combine with self (optional, as in GAT one can include self-message)
            h_out[i] = F.elu(agg)
        # 6. Readout: here we do simple average pooling of node embeddings for graph-level representation
        graph_emb = torch.mean(h_out, dim=0)  # shape (hidden_dim,)
        # (We could also concatenate topo_emb here as part of graph_emb if desired)
        # 7. Classification layer
        logits = self.out_linear(graph_emb)
        return logits

    def compute_topological_stability_loss(self, node_features, adj_matrix, true_label, perturb_fn):
        """
        Compute the stability regularization loss.
        :param perturb_fn: a function that takes adj_matrix and returns a perturbed adjacency (e.g., add a small edge).
        """
        # Forward on original graph
        logits = self.forward(node_features, adj_matrix)
        orig_loss = F.cross_entropy(logits.unsqueeze(0), true_label.unsqueeze(0))
        # Forward on perturbed graph
        pert_adj = perturb_fn(adj_matrix)
        pert_logits = self.forward(node_features, pert_adj)
        stab_loss = F.mse_loss(F.softmax(logits, dim=-1), F.softmax(pert_logits, dim=-1))
        # Alternatively or additionally, ensure the perturbed graph is classified correctly:
        robust_loss = F.cross_entropy(pert_logits.unsqueeze(0), true_label.unsqueeze(0))
        return stab_loss + robust_loss + orig_loss  # total loss includes original CE and stability terms

    def certify_stability(self, node_features, adj_matrix):
        """
        Certify the robustness radius for this graph input.
        Returns: certified radius delta (float).
        """
        # Get model prediction
        logits = self.forward(node_features, adj_matrix)
        probs = F.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs).item()
        pred_conf = probs[pred_label].item()
        # If multiple classes, find second highest confidence
        sorted_probs, _ = torch.sort(probs, descending=True)
        if len(sorted_probs) > 1:
            runnerup_conf = sorted_probs[1].item()
        else:
            runnerup_conf = 0.0
        margin = pred_conf - runnerup_conf  # confidence gap
        if margin <= 0:
            return 0.0  # no robustness if not confidently classified
        # Known Lipschitz constant of network (approx or enforced)
        L = self.Lipschitz_const
        # Using PH stability: the persistence diagram won't change significantly under perturbations of size delta.
        # We compute certified delta as margin / L (assuming persistence image change <= perturbation size).
        certified_delta = margin / L
        return certified_delta


