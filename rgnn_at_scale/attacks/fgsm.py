"""Contains a greedy FGSM implementation. In each iteration the edge is flipped, determined by the largest gradient
towards increasing the loss.
"""
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from rgnn_at_scale.models import DenseGCN


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
                 device: Union[str, int, torch.device],
                 stop_optimizing_if_label_flipped: bool = False,
                 **kwargs):
        super().__init__()
        assert adj.device == X.device, 'The device of the features and adjacency matrix must match'
        self.device = device
        self.original_adj = adj.to_dense().to(device)
        self.adj = self.original_adj.clone().requires_grad_(True)
        self.X = X.to(device)
        self.labels = labels.to(device)
        self.idx_attack = idx_attack
        self.model = deepcopy(model).to(self.device)
        self.stop_optimizing_if_label_flipped = stop_optimizing_if_label_flipped

        self.attr_adversary = None
        self.adj_adversary = None
        self.n_perturbations = 0

    def attack(self, n_perturbations: int):
        """Perform attack

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        assert n_perturbations > self.n_perturbations, (
            f'Number of perturbations must be bigger as this attack is greedy (current {n_perturbations}, '
            f'previous {self.n_perturbations})'
        )
        n_perturbations -= self.n_perturbations
        self.n_perturbations += n_perturbations

        for i in range(n_perturbations):
            logits = self.model.to(self.device)(self.X, self.adj)

            not_yet_flipped_mask = logits[self.idx_attack].argmax(-1) == self.labels[self.idx_attack]
            if self.stop_optimizing_if_label_flipped and not_yet_flipped_mask.sum() > 0:
                loss = F.cross_entropy(logits[self.idx_attack][not_yet_flipped_mask],
                                       self.labels[self.idx_attack][not_yet_flipped_mask])
            else:
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
