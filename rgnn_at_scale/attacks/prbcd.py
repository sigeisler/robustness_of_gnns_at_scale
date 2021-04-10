import logging

from collections import defaultdict
import math
from typing import Tuple, Union

from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor
from rgnn_at_scale.helper import utils
from rgnn_at_scale.models import MODEL_TYPE
from rgnn_at_scale.attacks.base_attack import Attack, SparseAttack

"""
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
    Tensorflow Implementation:
        https://github.com/KaidiXu/GCN_ADV_Train
"""


class PRBCD(SparseAttack):
    """Sampled and hence scalable PGD attack for graph data.
    """

    def __init__(self,
                 adj: SparseTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 keep_heuristic: str = 'WeightOnly',  # 'InvWeightGradient' 'Gradient', 'WeightOnly'
                 keep_weight: float = .1,
                 lr_n_perturbations_factor: float = 0.1,
                 lr_factor: float = 1,
                 display_step: int = 20,
                 epochs: int = 400,
                 fine_tune_epochs: int = 100,
                 search_space_size: int = 1_000_000,
                 search_space_dropout: float = 0,
                 with_early_stropping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-7,
                 K: int = 20,
                 **kwargs):

        super().__init__(adj, X, labels, idx_attack, model, device, loss_type, **kwargs)

        self.n_possible_edges = self.n * (self.n - 1) // 2
        self.keep_heuristic = keep_heuristic
        self.keep_weight = keep_weight
        self.lr_n_perturbations_factor = lr_n_perturbations_factor
        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.search_space_size = search_space_size
        self.search_space_dropout = search_space_dropout
        self.with_early_stropping = with_early_stropping
        self.eps = eps
        self.do_synchronize = do_synchronize
        # TODO: Rename
        self.K = K

        self.current_search_space: torch.Tensor = None
        self.modified_edge_index: torch.Tensor = None
        self.modified_edge_weight_diff: torch.Tensor = None

        if self.loss_type == 'CW':
            self.lr_factor = .5 * lr_factor
        else:
            self.lr_factor = 10 * lr_factor
        self.lr_factor *= max(math.log2(self.n_possible_edges / self.search_space_size), 1.)

    def _attack(self, n_perturbations, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        assert self.search_space_size > n_perturbations, \
            f'The search space size ({self.search_space_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'
        self.sample_search_space(n_perturbations)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')
        self.attack_statistics = defaultdict(list)

        mn = self.modified_edge_weight_diff.mean()
        logging.info(f'modified_edge_weight_diff mean is {mn}')

        for epoch in tqdm(range(self.epochs + self.fine_tune_epochs)):
            self.modified_edge_weight_diff.requires_grad = True
            edge_index, edge_weight = self.get_modified_adj()

            if self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self.surrogate_model(data=self.X.to(self.device), adj=(edge_index, edge_weight))
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])

            gradient = utils.grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            if self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                self.modified_edge_weight_diff.requires_grad = False
                edge_weight = self.update_edge_weights(n_perturbations, epoch, gradient)[1]
                self.projection(n_perturbations, edge_index, edge_weight)

                edge_index, edge_weight = self.get_modified_adj()
                logits = self.surrogate_model(data=self.X.to(self.device), adj=(edge_index, edge_weight))
                accuracy = (
                    logits.argmax(-1)[self.idx_attack] == self.labels[self.idx_attack]
                ).float().mean().item()
                if epoch % self.display_step == 0:
                    print(f'\nEpoch: {epoch} Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

                if self.with_early_stropping and best_accuracy > accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_search_space = self.current_search_space.clone().cpu()
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.modified_edge_weight_diff.detach().clone().cpu()

                self._append_attack_statistics(loss.item(), accuracy)

                if epoch < self.epochs - 1:
                    self.resample_search_space(n_perturbations, edge_index, edge_weight, gradient)
                    mn = self.modified_edge_weight_diff.mean()
                    t = (self.modified_edge_weight_diff > 0.4).sum()
                    logging.info(f'modified_edge_weight_diff mean is {mn} of this {t} over 0.4')
                elif self.with_early_stropping and epoch == self.epochs - 1:
                    print(f'Loading search space of epoch {best_epoch} (accuarcy={best_accuracy}) for fine tuning\n')
                    self.current_search_space = best_search_space.to(self.device)
                    self.modified_edge_index = best_edge_index.to(self.device)
                    self.modified_edge_weight_diff = best_edge_weight_diff.to(self.device)
                    self.modified_edge_weight_diff.requires_grad = True
                    mn = self.modified_edge_weight_diff.mean()
                    logging.info(f'modified_edge_weight_diff mean is {mn}')

            del logits
            del loss
            del gradient

        if self.with_early_stropping:
            self.current_search_space = best_search_space.to(self.device)
            self.modified_edge_index = best_edge_index.to(self.device)
            self.modified_edge_weight_diff = best_edge_weight_diff.to(self.device)

        edge_index = self.sample_edges(n_perturbations)[0]

        self.adj_adversary = SparseTensor.from_edge_index(
            edge_index,
            torch.ones_like(edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.X

    def sample_edges(self, n_perturbations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        best_accuracy = float('Inf')
        with torch.no_grad():
            s = self.modified_edge_weight_diff.abs().detach()
            s[s == self.eps] = 0
            # TODO: Why numpy?
            s = s.cpu().numpy()
            while best_accuracy == float('Inf'):
                for i in range(self.K):
                    sampled = np.random.binomial(1, s)

                    if sampled.sum() > n_perturbations:
                        n_samples = sampled.sum()
                        logging.info(f'{i}-th sampling: too many samples {n_samples}')
                        continue
                    pos_modified_edge_weight_diff = torch.from_numpy(sampled).to(self.device)
                    self.modified_edge_weight_diff = torch.where(
                        self.modified_edge_weight_diff > 0,
                        pos_modified_edge_weight_diff,
                        -pos_modified_edge_weight_diff
                    ).float()
                    edge_index, edge_weight = self.get_modified_adj()
                    logits = self.surrogate_model(data=self.X.to(self.device), adj=(edge_index, edge_weight))
                    accuracy = (
                        logits.argmax(-1)[self.idx_attack] == self.labels[self.idx_attack]
                    ).float().mean().item()
                    if best_accuracy > accuracy:
                        best_accuracy = accuracy
                        best_s = self.modified_edge_weight_diff.clone().cpu()
            self.modified_edge_weight_diff.data.copy_(best_s.to(self.device))
            edge_index, edge_weight = self.get_modified_adj(is_final=True)
            is_edge_set = torch.isclose(edge_weight, torch.tensor(1.))
            edge_index = edge_index[:, is_edge_set]
            edge_weight = edge_weight[is_edge_set]
        return edge_index, edge_weight

    def match_search_space_on_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_in_search_space = (edge_weight != 1) & (edge_index[0] < edge_index[1])
        assert is_in_search_space.sum() == self.current_search_space.size(0), \
            f'search space size mismatch: {is_in_search_space.sum()} vs. {self.current_search_space.size(0)}'
        modified_edge_weight = edge_weight[is_in_search_space]
        original_edge_weight = modified_edge_weight - self.modified_edge_weight_diff
        does_original_edge_exist = torch.isclose(original_edge_weight, torch.tensor(1.))
        # for source, dest in self.modified_edge_index[:, does_original_edge_exist].T:
        #     weight = self.edge_weight[(self.edge_index[0] == source) & (self.edge_index[1] == dest)]
        #     assert weight == 1, f'For edge ({source}, {dest}) the weight is not 1, it is {weight}'
        return does_original_edge_exist, is_in_search_space

    def projection(self, n_perturbations: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        does_original_edge_exist, is_in_search_space = self.match_search_space_on_edges(edge_index, edge_weight)

        pos_modified_edge_weight_diff = torch.where(
            does_original_edge_exist, -self.modified_edge_weight_diff, self.modified_edge_weight_diff
        )
        pos_modified_edge_weight_diff = Attack.project(n_perturbations, pos_modified_edge_weight_diff, self.eps)

        self.modified_edge_weight_diff = torch.where(
            does_original_edge_exist, -pos_modified_edge_weight_diff, pos_modified_edge_weight_diff
        )

    def handle_zeros_and_ones(self):
        # Handling edge case to detect an unchanged edge via its value 1
        self.modified_edge_weight_diff.data[
            (self.modified_edge_weight_diff <= self.eps)
            & (self.modified_edge_weight_diff >= -self.eps)
        ] = self.eps
        self.modified_edge_weight_diff.data[self.modified_edge_weight_diff >= 1 - self.eps] = 1 - self.eps
        self.modified_edge_weight_diff.data[self.modified_edge_weight_diff <= -1 + self.eps] = -1 + self.eps

    def get_modified_adj(self, is_final: bool = False):
        if self.search_space_dropout > 0 and not is_final:
            self.modified_edge_weight_diff.data = F.dropout(self.modified_edge_weight_diff.data)

        if not is_final:
            self.handle_zeros_and_ones()

        if (
            not self.modified_edge_weight_diff.requires_grad
            or not hasattr(self.surrogate_model, 'do_checkpoint')
            or not self.surrogate_model.do_checkpoint
        ):
            modified_edge_index, modified_edge_weight = utils.to_symmetric(
                self.modified_edge_index, self.modified_edge_weight_diff, self.n
            )
            edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
            edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

            edge_index, edge_weight = torch_sparse.coalesce(
                edge_index,
                edge_weight,
                m=self.n,
                n=self.n,
                op='sum'
            )
        else:
            from torch.utils import checkpoint

            def fuse_edges_run(modified_edge_weight_diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                modified_edge_index, modified_edge_weight = utils.to_symmetric(
                    self.modified_edge_index, modified_edge_weight_diff, self.n
                )
                edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

                # FIXME: This seems to be the current bottle neck. Maybe change to merge of sorted lists
                edge_index, edge_weight = torch_sparse.coalesce(
                    edge_index,
                    edge_weight,
                    m=self.n,
                    n=self.n,
                    op='sum'
                )
                return edge_index, edge_weight

            # Due to bottleneck...
            if len(self.edge_weight) > 100_000_000:
                device = self.device
                self.device = 'cpu'
                self.modified_edge_index = self.modified_edge_index.to(self.device)
                edge_index, edge_weight = fuse_edges_run(self.modified_edge_weight_diff.cpu())
                self.device = device
                self.modified_edge_index = self.modified_edge_index.to(self.device)
                return edge_index.to(self.device), edge_weight.to(self.device)

            # Currently (1.6.0) PyTorch does not support return arguments of `checkpoint` that do not require gradient.
            # For this reason we need to execute the code twice (due to checkpointing in fact three times...)
            with torch.no_grad():
                edge_index = fuse_edges_run(self.modified_edge_weight_diff)[0]

            edge_weight = checkpoint.checkpoint(
                lambda *input: fuse_edges_run(*input)[1],
                self.modified_edge_weight_diff
            )

        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations: int, epoch: int, gradient: torch.Tensor):
        lr_factor = max(1., n_perturbations / self.n / 2 / self.lr_n_perturbations_factor) * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs) + 1)
        self.modified_edge_weight_diff.data.add_(lr * gradient)

        return self.get_modified_adj()

    def sample_search_space(self, n_perturbations: int = 0):
        while True:
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            self.modified_edge_weight_diff = torch.ones_like(
                self.current_search_space, dtype=torch.float32, requires_grad=True)
            self.modified_edge_weight_diff.data.mul_(self.eps)
            if self.current_search_space.size(0) >= n_perturbations:
                break

    def resample_search_space(self, n_perturbations: int, edge_index: torch.Tensor,
                              edge_weight: torch.Tensor, gradient: torch.Tensor):
        does_original_edge_exist = self.match_search_space_on_edges(edge_index, edge_weight)[0]
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.modified_edge_weight_diff.abs())
            idx_keep = (self.modified_edge_weight_diff <= self.eps).sum().long()
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            if self.keep_heuristic == 'InvWeightGradient':
                gradient = gradient * 1 / self.modified_edge_weight_diff.abs()

            attack_gradient = torch.where(does_original_edge_exist, -gradient, gradient)

            keep_edges = self.modified_edge_weight_diff.abs() > self.keep_weight
            if self.keep_weight is not None and self.keep_weight > 0 and keep_edges.sum() > 0:
                if keep_edges.sum() < keep_edges.size(0):
                    sorted_idx = torch.cat((keep_edges.nonzero().reshape(-1),
                                            torch.argsort(attack_gradient[~keep_edges])))
                else:
                    sorted_idx = keep_edges.nonzero().squeeze()
                n_keep_edges = keep_edges.sum()
                idx_keep = (self.modified_edge_weight_diff.size(0) - n_keep_edges) // 2 + n_keep_edges
            else:
                sorted_idx = torch.argsort(attack_gradient)
                idx_keep = sorted_idx.size(0) // 2

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.modified_edge_weight_diff = self.modified_edge_weight_diff[sorted_idx]

        # Sample until enough edges were drawn
        while True:
            lin_index = torch.randint(self.n_possible_edges, (self.search_space_size -
                                                              self.current_search_space.size(0),), device=self.device)
            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )
            self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            # Merge existing weights with new edge weights
            modified_edge_weight_diff_old = self.modified_edge_weight_diff.clone()
            self.modified_edge_weight_diff = self.eps * torch.ones_like(self.current_search_space, dtype=torch.float32)
            self.modified_edge_weight_diff[
                unique_idx[:modified_edge_weight_diff_old.size(0)]
            ] = modified_edge_weight_diff_old

            if self.current_search_space.size(0) > n_perturbations:
                break

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = (
            n
            - 2
            - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        ).long()
        col_idx = (
            lin_idx
            + row_idx
            + 1 - n * (n - 1) // 2
            + (n - row_idx) * ((n - row_idx) - 1) // 2
        )
        return torch.stack((row_idx, col_idx))

    def _append_attack_statistics(self, loss, accuracy):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['accuracy'].append(accuracy)
        self.attack_statistics['nonzero_weights'].append((self.modified_edge_weight_diff.abs() > self.eps).sum().item())
        self.attack_statistics['expected_perturbations'].append(self.modified_edge_weight_diff.sum().item())
