from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from torch_sparse import SparseTensor
from rgnn_at_scale.models import MODEL_TYPE, DenseGCN


class Attack(ABC):

    def __init__(self,
                 adj: Union[SparseTensor, torch.Tensor],
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 **kwargs):
        self.device = device
        self.idx_attack = idx_attack
        self.loss_type = loss_type

        self.model = deepcopy(model).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.labels = labels.to(device)
        self.labels_attack = self.labels[self.idx_attack]
        self.X = X.to(device)
        self.adj = adj.to(device)

        self.attr_adversary = self.X
        self.adj_adversary = None

    @abstractmethod
    def attack(self, n_perturbations: int, **kwargs):
        pass

    def calculate_loss(self, logits, labels):
        if self.loss_type == 'CW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -torch.clamp(margin, min=0).mean()
        elif self.loss_type == 'LCW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
        elif self.loss_type == 'tanhCW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.nll_loss(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'WCE':
            not_flipped = logits.argmax(-1) == labels
            weighting_not_flipped = not_flipped.sum().item() / not_flipped.shape[0]
            weighting_flipped = (not_flipped.shape[0] - not_flipped.sum().item()) / not_flipped.shape[0]
            loss_not_flipped = F.nll_loss(logits[not_flipped], labels[not_flipped])
            loss_flipped = F.nll_loss(logits[~not_flipped], labels[~not_flipped])
            loss = (
                weighting_not_flipped * loss_not_flipped
                + 0.25 * weighting_flipped * loss_flipped
            )
        elif self.loss_type == 'SCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.nll_loss(logits, best_non_target_class)
        # TODO: Is it worth trying? CW should be quite similar
        # elif self.loss_type == 'Margin':
        #    loss = F.multi_margin_loss(torch.exp(logits), labels)
        else:
            loss = F.nll_loss(logits, labels)
        return loss

    @staticmethod
    def project(n_perturbations: int, values: torch.tensor, eps: float = 0, inplace: bool = False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = Attack.bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    @staticmethod
    def bisection(pos_modified_edge_weight_diff, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
        def func(x):
            return torch.clamp(pos_modified_edge_weight_diff - x, 0, 1).sum() - n_perturbations

        miu = a
        for i in range(int(iter_max)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) >= epsilon):
                break
        return miu


class SparseAttack(Attack):
    def __init__(self,
                 adj: Union[SparseTensor, torch.Tensor, sp.csr_matrix],
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 **kwargs):

        if isinstance(adj, torch.Tensor):
            adj = SparseTensor.from_dense(adj)
        elif isinstance(adj, sp.csr_matrix):
            adj = SparseTensor.from_scipy(adj)

        super().__init__(adj, X, labels, idx_attack, model, device, loss_type, **kwargs)

        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(self.device)
        self.edge_weight = edge_weight.to(self.device)
        self.n = adj.size(0)
        self.d = X.shape[1]


class DenseAttack(Attack):
    def __init__(self,
                 adj: Union[SparseTensor, torch.Tensor],
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 device: Union[str, int, torch.device],
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 **kwargs):
        assert isinstance(model, DenseGCN), "DenseAttacks can only attack the DenseGCN model"

        if isinstance(adj, SparseTensor):
            adj = adj.to_dense()

        super().__init__(adj, X, labels, idx_attack, model, device, loss_type, **kwargs)

        self.n = adj.shape[0]
