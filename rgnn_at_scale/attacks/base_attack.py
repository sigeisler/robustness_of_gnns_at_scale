import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, List

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from torch_sparse import SparseTensor
from rgnn_at_scale.models import MODEL_TYPE, DenseGCN, GCN, BATCHED_PPR_MODELS
from rgnn_at_scale.helper.utils import accuracy


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

        # TODO: should this be an assert instead of a warning?
        if not (isinstance(model, GCN) or isinstance(model, DenseGCN)):
            warnings.warn("The attack will fail if the gradient w.r.t. the adjacency can't be computed.")

        if isinstance(model, GCN):
            assert model.gdc_params is None, "GDC doesn't support a gradient w.r.t. the adjacency"
            assert model.svd_params is None, "GDC doesn't support a gradient w.r.t. the adjacency"
            assert model.jaccard_params is None, "GDC doesn't support a gradient w.r.t. the adjacency"

        self.device = device
        self.idx_attack = idx_attack
        self.loss_type = loss_type

        self.surrogate_model = deepcopy(model).to(self.device)
        self.surrogate_model.eval()
        for p in self.surrogate_model.parameters():
            p.requires_grad = False

        self.set_eval_model(model)

        self.labels = labels.to(device)
        self.labels_attack = self.labels[self.idx_attack]
        self.X = X.to(device)
        self.adj = adj.to(device)

        self.attr_adversary = self.X
        self.adj_adversary = self.adj

    @abstractmethod
    def _attack(self, n_perturbations: int, **kwargs):
        pass

    def attack(self, n_perturbations: int, **kwargs):
        if n_perturbations > 0:
            return self._attack(n_perturbations, **kwargs)
        else:
            self.attr_adversary = self.X
            self.adj_adversary = self.adj

    def set_pertubations(self, adj_perturbed, attr_perturbed):
        self.adj_adversary = adj_perturbed.to(self.device)
        self.attr_adversary = attr_perturbed.to(self.device)

    def get_pertubations(self):
        if isinstance(self.adj_adversary, torch.Tensor):
            self.adj_adversary = SparseTensor.from_dense(self.adj_adversary)

        if isinstance(self.attr_adversary, SparseTensor):
            self.attr_adversary = self.attr_adversary.to_dense()

        return self.adj_adversary, self.attr_adversary

    def set_eval_model(self, model):
        self.eval_model = deepcopy(model).to(self.device)

    def evaluate_global(self, eval_idx: List[int]):
        with torch.no_grad():
            self.eval_model.eval()
            if hasattr(self.eval_model, 'release_cache'):
                self.eval_model.release_cache()

            if type(self.eval_model) in BATCHED_PPR_MODELS.__args__:
                pred_logits_target = self.eval_model.forward(self.attr_adversary,
                                                             self.adj_adversary,
                                                             ppr_idx=np.array(eval_idx))
                acc_test_target = (pred_logits_target.cpu().argmax(
                    1) == self.labels.cpu()[eval_idx]).float().mean().item()
            else:
                pred_logits_target = self.eval_model(self.attr_adversary, self.adj_adversary)
                acc_test_target = accuracy(pred_logits_target.cpu(), self.labels.cpu(), eval_idx)

        return pred_logits_target, acc_test_target

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


class SparseLocalAttack(SparseAttack):
    @abstractmethod
    def get_perturbed_edges(self, node_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None):
        pass

    def get_surrogate_logits(self, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        return self.get_logits(self.surrogate_model, node_idx, perturbed_graph)

    def get_eval_logits(self, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        return self.get_logits(self.eval_model, node_idx, perturbed_graph)

    def evaluate_local(self, node_idx: int):
        with torch.no_grad():
            self.eval_model.eval()
            if hasattr(self.eval_model, 'release_cache'):
                self.eval_model.release_cache()

            initial_logits = self.get_eval_logits(node_idx)
            logits = self.get_eval_logits(node_idx, self.adj_adversary)
        return logits, initial_logits


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

        assert isinstance(adj,  torch.Tensor)
        super().__init__(adj, X, labels, idx_attack, model, device, loss_type, **kwargs)

        self.n = adj.shape[0]
