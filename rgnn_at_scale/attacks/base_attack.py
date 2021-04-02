from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

from torch.nn import functional as F
import numpy as np
import torch
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
        self.X = X.to(device)
        self.adj = adj.to(device)

        self.attr_adversary = self.X
        self.adj_adversary = None

    @abstractmethod
    def attack(self, n_perturbations: int, **kwargs):
        pass

    def calculate_loss(self, logits, labels):
        if self.loss_type == 'CW':
            second_best_class = logits.argsort(-1)[:, -2]
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


class SparseAttack(Attack):
    def __init__(self,
                 adj: Union[SparseTensor, torch.Tensor],
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 **kwargs):

        if not isinstance(adj, SparseTensor):
            adj = SparseTensor.from_dense(adj)

        super().__init__(adj, X, labels, idx_attack, model, device, loss_type, **kwargs)

        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(self.device)
        self.edge_weight = edge_weight.to(self.device)
        self.n = adj.size(0)


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
