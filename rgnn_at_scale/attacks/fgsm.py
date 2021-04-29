"""Contains a greedy FGSM implementation. In each iteration the edge is flipped, determined by the largest gradient
towards increasing the loss.
"""
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from rgnn_at_scale.models import DenseGCN
from rgnn_at_scale.attacks.base_attack import DenseAttack


class FGSM(DenseAttack):
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
                 adj: Union[SparseTensor, torch.Tensor],
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 device: Union[str, int, torch.device],
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 stop_optimizing_if_label_flipped: bool = False,
                 **kwargs):
        super().__init__(adj, X, labels, idx_attack, model, device, loss_type, **kwargs)

        self.adj_tmp = self.adj.clone().requires_grad_(True)
        self.stop_optimizing_if_label_flipped = stop_optimizing_if_label_flipped
        self.n_perturbations = 0

    def _attack(self, n_perturbations: int):
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
            logits = self.surrogate_model.to(self.device)(self.X, self.adj_tmp)

            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])

            gradient = torch.autograd.grad(loss, self.adj_tmp)[0]
            gradient[self.adj != self.adj_tmp] = 0
            gradient *= 2 * (0.5 - self.adj_tmp)

            # assert torch.all(gradient.nonzero()[:, 0] < gradient.nonzero()[:, 1]),\
            #     'Only upper half should get nonzero gradient'

            maximum = torch.max(gradient)
            edge_pert = (maximum == gradient).nonzero()

            with torch.no_grad():
                new_edge_value = -self.adj_tmp[edge_pert[0][0], edge_pert[0][1]] + 1
                self.adj_tmp[edge_pert[0][0], edge_pert[0][1]] = new_edge_value
                self.adj_tmp[edge_pert[0][1], edge_pert[0][0]] = new_edge_value

        self.attr_adversary = self.X
        self.adj_adversary = SparseTensor.from_dense(self.adj_tmp.detach())
