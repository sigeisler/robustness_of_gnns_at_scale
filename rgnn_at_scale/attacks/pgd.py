"""This file contains the Projected Gradient Descent attack as proposed in:
Kaidi Xu, Hongge Chen, Sijia Liu, Pin Yu Chen, Tsui Wei Weng, Mingyi Hong, and Xue Lin.Topology attack and defense
for graph neural networks: An optimization perspective. IJCAI International Joint Conference on Artificial
Intelligence, 2019-Augus:3961–3967, 2019. ISSN10450823. doi: 10.24963/ijcai.2019/550.

The Subsequent code build upon the implementation https://github.com/DSE-MSU/DeepRobust (under MIT License). We did
not intent to unify the code style, programming paradigms, etc. with the rest of the code base.

"""
import numpy as np
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

from rgnn_at_scale.attacks.base_attack import DenseAttack


class PGD(DenseAttack):
    """L_0 norm Projected Gradient Descent (PGD) attack as proposed in:
    Kaidi Xu, Hongge Chen, Sijia Liu, Pin Yu Chen, Tsui Wei Weng, Mingyi Hong, and Xue Lin.Topology attack and defense
    for graph neural networks: An optimization perspective. IJCAI International Joint Conference on Artificial
    Intelligence, 2019-Augus:3961–3967, 2019. ISSN10450823. doi: 10.24963/ijcai.2019/550.

    Parameters
    ----------
    X : torch.Tensor
        [n, d] feature matrix.
    adj : Union[SparseTensor, torch.Tensor]
        [n, n] adjacency matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked [?].
    model : DenseGCN
        Model to be attacked.
    epochs : int, optional
        Number of epochs to attack the adjacency matrix, by default 200.
    loss_type : str, optional
        'CW' for Carlini and Wagner or 'CE' for cross entropy, by default 'CE'.
    """

    def __init__(self,
                 epochs: int = 200,
                 epsilon: float = 1e-5,
                 base_lr: float = 1e-2,
                 **kwargs):
        super().__init__(**kwargs)

        assert self.make_undirected, 'Attack only implemented for undirected graphs'

        self.epochs = epochs
        self.epsilon = epsilon
        self.base_lr = base_lr

        self.adj = self.adj.to(self.device)
        self.attr = self.attr.to(self.device)
        self.attacked_model = self.attacked_model.to(self.device)

    def _attack(self, n_perturbations: int, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        self.complementary = None
        self.adj_changes = torch.zeros(int(self.n * (self.n - 1) / 2), dtype=torch.float, device=self.device)
        self.adj_changes.requires_grad = True

        self.attacked_model.eval()
        for t in tqdm(range(self.epochs)):
            modified_adj = self.get_modified_adj()
            logits = self.attacked_model(self.attr, modified_adj)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            lr = n_perturbations * self.base_lr / np.sqrt(t + 1)
            self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)

        self.random_sample(n_perturbations)
        self.adj_adversary = SparseTensor.from_dense(self.get_modified_adj().detach())

    def random_sample(self, n_perturbations: int):
        K = 20
        best_loss = float('-Inf')
        with torch.no_grad():
            while best_loss == float('-Inf'):
                s = self.adj_changes.cpu().detach().numpy()
                for i in range(K):
                    sampled = np.random.binomial(1, s)

                    if sampled.sum() > n_perturbations:
                        continue
                    self.adj_changes.data.copy_(torch.tensor(sampled))
                    modified_adj = self.get_modified_adj()
                    logits = self.attacked_model(self.attr, modified_adj)
                    loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
                    if best_loss < loss:
                        best_loss = loss
                        best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def projection(self, n_perturbations: int):
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = PGD.bisection(left, right, self.adj_changes, n_perturbations, self.epsilon)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self):
        if self.complementary is None:
            self.complementary = torch.ones_like(self.adj) - torch.eye(self.n, device=self.device) - 2 * self.adj

        m = torch.zeros_like(self.adj)
        tril_indices = torch.tril_indices(row=self.n, col=self.n, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + self.adj

        return modified_adj

    @staticmethod
    def bisection(a: float, b: float, adj_changes: torch.Tensor, n_perturbations: int, epsilon: float):
        def func(x):
            return torch.clamp(adj_changes - x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            if (func(miu) == 0.0):
                break
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu
