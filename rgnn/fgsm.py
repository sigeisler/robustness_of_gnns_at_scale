"""Contains a greedy FGSM implementation. In each iteration the edge is flipped, determined by the largest gradient
towards increasing the loss.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from rgnn.models import DenseGCN


class FGSM():
    """Greedy Fast Gradient Signed Method.

    Parameters
    ----------
    adj : torch.sparse.FloatTensor
        [n, n] sparse adjacency matrix.
    X : torch.Tensor
        [n, d] feature matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked [?].
    model : DenseGCN
        Model to be attacked.
    """

    def __init__(self,
                 adj: torch.sparse.FloatTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 **kwargs):
        super().__init__()
        assert adj.device == X.device, 'The device of the features and adjacency matrix must match'
        self.device = X.device
        self.original_adj = adj.to_dense()
        self.adj = self.original_adj.clone().requires_grad_(True)
        self.X = X
        self.labels = labels
        self.idx_attack = idx_attack
        self.model = deepcopy(model).to(self.device)
        self.attr_adversary = None
        self.adj_adversary = None

    def attack(self,
               n_perturbations: int,
               **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        for i in range(n_perturbations):
            logits = self.model.to(self.device)(self.X, self.adj)

            loss = F.cross_entropy(logits[self.idx_attack], self.labels[self.idx_attack])

            gradient = torch.autograd.grad(loss, self.adj)[0]
            gradient[self.original_adj != self.adj] = 0
            gradient *= 2 * (0.5 - self.adj)

            # assert torch.all(gradient.nonzero()[:, 0] < gradient.nonzero()[:, 1]),\
            #     'Only upper half should get nonzero gradient'

            maximum = torch.max(gradient)
            edge_pert = (maximum == gradient).nonzero()

            with torch.no_grad():
                new_edge_value = -self.adj[edge_pert[0][0], edge_pert[0][1]] + 1
                self.adj[edge_pert[0][0], edge_pert[0][1]] = new_edge_value
                self.adj[edge_pert[0][1], edge_pert[0][0]] = new_edge_value

        self.attr_adversary = self.X
        self.adj_adversary = self.adj.to_sparse().detach()
