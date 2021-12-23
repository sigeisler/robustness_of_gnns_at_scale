import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union, List, Dict
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import logging

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from torch_sparse import SparseTensor
from rgnn_at_scale.models import MODEL_TYPE, DenseGCN, GCN, RGNN, BATCHED_PPR_MODELS
from rgnn_at_scale.helper.utils import accuracy

patch_typeguard()


@typechecked
class Attack(ABC):
    """
    Base class for all attacks providing a uniform interface for all attacks
    as well as the implementation for all losses proposed or mentioned in our paper.

    Parameters
    ----------
    adj : SparseTensor or torch.Tensor
        [n, n] (sparse) adjacency matrix.
    attr : torch.Tensor
        [n, d] feature/attribute matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked.
    model : MODEL_TYPE
        Model to be attacked.
    device : Union[str, int, torch.device]
        The cuda device to use for the attack
    data_device : Union[str, int, torch.device]
        The cuda device to use for storing the dataset.
        For batched models (like PPRGo) this may differ from the device parameter.
        Other models require the dataset and model to be on the same device.
    make_undirected: bool
        Wether the perturbed adjacency matrix should be made undirected (symmetric degree normalization)
    binary_attr: bool
        If true the perturbed attributes are binarized (!=0)
    loss_type: str
        The loss to be used by a gradient based attack, can be one of the following loss types:
            - CW: Carlini-Wagner
            - LCW: Leaky Carlini-Wagner
            - Margin: Negative classification margin
            - tanhMargin: Negative TanH of classification margin
            - eluMargin: Negative Exponential Linear Unit (ELU) of classification margin
            - CE: Cross Entropy
            - MCE: Masked Cross Entropy
            - NCE: Negative Cross Entropy
    """

    def __init__(self,
                 adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
                 attr: TensorType["n_nodes", "n_features"],
                 labels: TensorType["n_nodes"],
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 data_device: Union[str, int, torch.device],
                 make_undirected: bool,
                 binary_attr: bool,
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 **kwargs):
        if not (isinstance(model, GCN) or isinstance(model, DenseGCN) or isinstance(model, RGNN)):
            warnings.warn("The attack will fail if the gradient w.r.t. the adjacency can't be computed.")

        if isinstance(model, GCN) or isinstance(model, RGNN):
            assert (
                model.gdc_params is None
                or 'use_cpu' not in model.gdc_params
                or not model.gdc_params['use_cpu']
            ), "GDC doesn't support a gradient w.r.t. the adjacency"
            assert model.svd_params is None, "SVD preproc. doesn't support a gradient w.r.t. the adjacency"
            assert model.jaccard_params is None, "Jaccard preproc. doesn't support a gradient w.r.t. the adjacency"
        if isinstance(model, RGNN):
            assert model._mean in ['dimmedian', 'medoid', 'soft_median'],\
                "Agg. doesn't support a gradient w.r.t. the adjacency"

        self.device = device
        self.data_device = data_device
        self.idx_attack = idx_attack
        self.loss_type = loss_type

        self.make_undirected = make_undirected
        self.binary_attr = binary_attr

        self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model.eval()
        for p in self.attacked_model.parameters():
            p.requires_grad = False
        self.eval_model = self.attacked_model

        self.labels = labels.to(torch.long).to(self.device)
        self.labels_attack = self.labels[self.idx_attack]
        self.attr = attr.to(self.data_device)
        self.adj = adj.to(self.data_device)

        self.attr_adversary = self.attr
        self.adj_adversary = self.adj

    @abstractmethod
    def _attack(self, n_perturbations: int, **kwargs):
        pass

    def attack(self, n_perturbations: int, **kwargs):
        """
        Executes the attack on the model updating the attributes
        self.adj_adversary and self.attr_adversary accordingly.

        Parameters
        ----------
        n_perturbations : int
            number of perturbations (attack budget in terms of node additions/deletions) that constrain the atack
        """
        if n_perturbations > 0:
            return self._attack(n_perturbations, **kwargs)
        else:
            self.attr_adversary = self.attr
            self.adj_adversary = self.adj

    def set_pertubations(self, adj_perturbed: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
                         attr_perturbed: TensorType["n_nodes", "n_features"]):
        self.adj_adversary = adj_perturbed.to(self.data_device)
        self.attr_adversary = attr_perturbed.to(self.data_device)

    def get_pertubations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary

        if isinstance(self.adj_adversary, torch.Tensor):
            adj_adversary = SparseTensor.from_dense(self.adj_adversary)

        if isinstance(self.attr_adversary, SparseTensor):
            attr_adversary = self.attr_adversary.to_dense()

        return adj_adversary, attr_adversary

    @staticmethod
    @torch.no_grad()
    def evaluate_global(model,
                        attr: TensorType["n_nodes", "n_features"],
                        adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
                        labels: TensorType["n_nodes"],
                        eval_idx: Union[List[int], np.ndarray]):
        """
        Evaluates any model w.r.t. accuracy for a given (perturbed) adjacency and attribute matrix.
        """
        model.eval()
        if hasattr(model, 'release_cache'):
            model.release_cache()

        if type(model) in BATCHED_PPR_MODELS.__args__:
            pred_logits_target = model.forward(attr, adj, ppr_idx=np.array(eval_idx))
        else:
            pred_logits_target = model(attr, adj)[eval_idx]

        acc_test_target = accuracy(pred_logits_target.cpu(), labels.cpu()[eval_idx],
                                   np.arange(pred_logits_target.shape[0]))

        return pred_logits_target, acc_test_target

    def calculate_loss(self, logits, labels):
        """
        TODO: maybe add formal definition for all losses? or maybe don't
        """
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
        elif self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'Margin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -margin.mean()
        elif self.loss_type.startswith('tanhMarginCW-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = (alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)).mean()
        elif self.loss_type.startswith('tanhMarginMCE-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )

            not_flipped = logits.argmax(-1) == labels

            loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * \
                F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'eluMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.elu(margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    @staticmethod
    def project(n_perturbations: int, values: torch.Tensor, eps: float = 0, inplace: bool = False):
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
    def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
        def func(x):
            return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

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
            if ((b - a) <= epsilon):
                break
        return miu


@typechecked
class SparseAttack(Attack):
    """
    Base class for all sparse attacks.
    Just like the base attack class but automatically casting the adjacency to sparse format.
    """

    def __init__(self,
                 adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"], sp.csr_matrix],
                 **kwargs):

        if isinstance(adj, torch.Tensor):
            adj = SparseTensor.from_dense(adj)
        elif isinstance(adj, sp.csr_matrix):
            adj = SparseTensor.from_scipy(adj)

        super().__init__(adj, **kwargs)

        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(self.data_device)
        self.edge_weight = edge_weight.to(self.data_device)
        self.n = adj.size(0)
        self.d = self.attr.shape[1]


@typechecked
class SparseLocalAttack(SparseAttack):
    """
    Base class for all local sparse attacks
    """

    @abstractmethod
    def get_perturbed_edges(self) -> torch.Tensor:
        """
        returns the edge (in coo format) that should be perturbed (added/deleted)
        """
        pass

    @abstractmethod
    def get_logits(self,
                   model: MODEL_TYPE,
                   node_idx: int,
                   perturbed_graph: Optional[Union[SparseTensor,
                                                   Tuple[TensorType[2, "nnz"],
                                                         TensorType["nnz"]]]] = None):
        pass

    def get_surrogate_logits(self,
                             node_idx: int,
                             perturbed_graph: Optional[Union[SparseTensor,
                                                             Tuple[TensorType[2, "nnz"],
                                                                   TensorType["nnz"]]]] = None) -> torch.Tensor:
        return self.get_logits(self.attacked_model, node_idx, perturbed_graph)

    def get_eval_logits(self,
                        node_idx: int,
                        perturbed_graph: Optional[Union[SparseTensor,
                                                        Tuple[TensorType[2, "nnz"],
                                                              TensorType["nnz"]]]] = None) -> torch.Tensor:
        return self.get_logits(self.eval_model, node_idx, perturbed_graph)

    @torch.no_grad()
    def evaluate_local(self, node_idx: int):
        self.eval_model.eval()
        if hasattr(self.eval_model, 'release_cache'):
            self.eval_model.release_cache()
        if hasattr(self.eval_model, "deactivate_caching"):
            self.eval_model.deactivate_caching()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024 ** 3)
            logging.info(f"Cuda Memory before local evaluation on clean adjacency {memory}")

        initial_logits = self.get_eval_logits(node_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024 ** 3)
            logging.info(f"Cuda Memory before local evaluation on perturbed adjacency {memory}")

        logits = self.get_eval_logits(node_idx, self.adj_adversary)
        return logits, initial_logits

    @staticmethod
    def evaluate_global(model, attr, adj, labels: torch.Tensor, eval_idx: List[int]):
        raise NotImplementedError("Can't evaluate globally for a local attack")

    def set_eval_model(self, model):
        self.eval_model = deepcopy(model).to(self.device)

    @staticmethod
    def classification_statistics(logits: TensorType[1, "n_classes"],
                                  label: TensorType[()]) -> Dict[str, float]:
        logits, label = F.log_softmax(logits.cpu(), dim=-1), label.cpu()
        logits = logits[0]
        logit_target = logits[label].item()
        sorted = logits.argsort()
        logit_best_non_target = (logits[sorted[sorted != label][-1]]).item()
        confidence_target = np.exp(logit_target)
        confidence_non_target = np.exp(logit_best_non_target)
        margin = confidence_target - confidence_non_target
        return {
            'logit_target': logit_target,
            'logit_best_non_target': logit_best_non_target,
            'confidence_target': confidence_target,
            'confidence_non_target': confidence_non_target,
            'margin': margin
        }

    def adj_adversary_for_poisoning(self):
        return self.adj_adversary


class DenseAttack(Attack):

    @typechecked
    def __init__(self,
                 adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
                 attr: TensorType["n_nodes", "n_features"],
                 labels: TensorType["n_nodes"],
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 device: Union[str, int, torch.device],
                 data_device: Union[str, int, torch.device],
                 loss_type: str = 'CE',
                 **kwargs):
        assert isinstance(model, DenseGCN), "DenseAttacks can only attack the DenseGCN model"

        if isinstance(adj, SparseTensor):
            adj = adj.to_dense()

        super().__init__(adj, attr, labels, idx_attack, model, device, data_device, loss_type=loss_type, **kwargs)

        self.n = adj.shape[0]
