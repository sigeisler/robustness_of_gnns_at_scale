"""Contains a greedy FGSM implementation. In each iteration the edge is flipped, determined by the largest gradient
towards increasing the loss.
"""
from tqdm import tqdm
import torch
from torch_sparse import SparseTensor

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.make_undirected, 'Attack only implemented for undirected graphs'

        self.adj_perturbed = self.adj.clone().requires_grad_(True).to(self.device)
        self.n_perturbations = 0

        self.adj = self.adj.to(self.device)
        self.attr = self.attr.to(self.device)
        self.attacked_model = self.attacked_model.to(self.device)

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

        for i in tqdm(range(n_perturbations)):
            logits = self.attacked_model(self.attr, self.adj_perturbed)

            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])

            gradient = torch.autograd.grad(loss, self.adj_perturbed)[0]
            gradient[self.adj != self.adj_perturbed] = 0
            gradient *= 2 * (0.5 - self.adj_perturbed)

            maximum = torch.max(gradient)
            edge_pert = (maximum == gradient).nonzero()

            with torch.no_grad():
                new_edge_value = -self.adj_perturbed[edge_pert[0][0], edge_pert[0][1]] + 1
                self.adj_perturbed[edge_pert[0][0], edge_pert[0][1]] = new_edge_value
                self.adj_perturbed[edge_pert[0][1], edge_pert[0][0]] = new_edge_value

        self.attr_adversary = self.attr
        self.adj_adversary = SparseTensor.from_dense(self.adj_perturbed.detach())
