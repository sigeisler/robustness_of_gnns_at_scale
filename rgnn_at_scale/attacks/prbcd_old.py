import logging

from collections import defaultdict
import math
from typing import Tuple
from typeguard import typechecked

from tqdm import tqdm
import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor
from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.base_attack import Attack, SparseAttack


class PRBCD(SparseAttack):
    """Sampled and hence scalable PGD attack for graph data.
    """

    @typechecked
    def __init__(self,
                 keep_heuristic: str = 'WeightOnly',  # 'InvWeightGradient' 'Gradient', 'WeightOnly'
                 keep_weight: float = .1,
                 lr_factor: float = 100,
                 display_step: int = 20,
                 epochs: int = 400,
                 fine_tune_epochs: int = 100,
                 block_size: int = 1_000_000,
                 with_early_stopping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-7,
                 max_resamples: int = 20,
                 **kwargs):
        super().__init__(**kwargs)

        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        self.keep_heuristic = keep_heuristic
        self.keep_weight = keep_weight
        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.block_size = block_size
        self.with_early_stopping = with_early_stopping
        self.eps = eps
        self.do_synchronize = do_synchronize
        self.max_resamples = max_resamples

        self.current_search_space: torch.Tensor = None
        self.modified_edge_index: torch.Tensor = None
        self.modified_edge_weight_diff: torch.Tensor = None

        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.block_size), 1.)

    def _attack(self, n_perturbations, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        assert self.block_size > n_perturbations, \
            f'The search space size ({self.block_size}) must be ' \
            + f'greater than the number of permutations ({n_perturbations})'
        self.sample_search_space(n_perturbations)
        best_accuracy = float('Inf')
        best_epoch = float('-Inf')
        self.attack_statistics = defaultdict(list)

        with torch.no_grad():
            logits = self.attacked_model(
                data=self.attr.to(self.device),
                adj=(self.edge_index.to(self.device), self.edge_weight.to(self.device))
            )
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            accuracy = (
                logits.argmax(-1)[self.idx_attack] == self.labels[self.idx_attack]
            ).float().mean().item()
            logging.info(f'\nBefore the attack - Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')
            self._append_attack_statistics(loss.item(), accuracy, 0., 0.)
            del logits
            del loss

        for epoch in tqdm(range(self.epochs + self.fine_tune_epochs)):
            self.modified_edge_weight_diff.requires_grad = True
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])

            gradient = utils.grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                self.modified_edge_weight_diff.requires_grad = False
                edge_weight = self.update_edge_weights(n_perturbations, epoch, gradient)[1]
                probability_mass, probability_mass_projected = self.projection(n_perturbations, edge_index, edge_weight)

                edge_index, edge_weight = self.get_modified_adj()
                logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
                accuracy = (
                    logits.argmax(-1)[self.idx_attack] == self.labels[self.idx_attack]
                ).float().mean().item()
                if epoch % self.display_step == 0:
                    logging.info(f'\nEpoch: {epoch} Loss: {loss.item()} Accuracy: {100 * accuracy:.3f} %\n')

                if self.with_early_stopping and best_accuracy > accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_search_space = self.current_search_space.clone().cpu()
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.modified_edge_weight_diff.detach().clone().cpu()

                self._append_attack_statistics(loss.item(), accuracy, probability_mass, probability_mass_projected)

                if epoch < self.epochs - 1:
                    self.resample_search_space(n_perturbations, edge_index, edge_weight, gradient)
                elif self.with_early_stopping and epoch == self.epochs - 1:
                    logging.info(
                        f'Loading search space of epoch {best_epoch} (accuarcy={best_accuracy}) for fine tuning\n')
                    self.current_search_space = best_search_space.to(self.device)
                    self.modified_edge_index = best_edge_index.to(self.device)
                    self.modified_edge_weight_diff = best_edge_weight_diff.to(self.device)
                    self.modified_edge_weight_diff.requires_grad = True

            del logits
            del loss
            del gradient

        if self.with_early_stopping:
            self.current_search_space = best_search_space.to(self.device)
            self.modified_edge_index = best_edge_index.to(self.device)
            self.modified_edge_weight_diff = best_edge_weight_diff.to(self.device)

        edge_index = self.sample_final_edges(n_perturbations)[0]

        self.adj_adversary = SparseTensor.from_edge_index(
            edge_index,
            torch.ones_like(edge_index[0], dtype=torch.float32),
            (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.attr

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        best_accuracy = float('Inf')
        s = self.modified_edge_weight_diff.abs().detach()
        s[s == self.eps] = 0
        while best_accuracy == float('Inf'):
            for i in range(self.max_resamples):
                if best_accuracy == float('Inf'):
                    sampled = torch.zeros_like(s)
                    sampled[torch.topk(s, n_perturbations).indices] = 1
                else:
                    sampled = torch.bernoulli(s).float()

                if sampled.sum() > n_perturbations:
                    n_samples = sampled.sum()
                    logging.info(f'{i}-th sampling: too many samples {n_samples}')
                    continue
                pos_modified_edge_weight_diff = sampled
                self.modified_edge_weight_diff = torch.where(
                    self.modified_edge_weight_diff > 0,
                    pos_modified_edge_weight_diff,
                    -pos_modified_edge_weight_diff
                ).float()
                edge_index, edge_weight = self.get_modified_adj()
                logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
                accuracy = (
                    logits.argmax(-1)[self.idx_attack] == self.labels[self.idx_attack]
                ).float().mean().item()
                if best_accuracy > accuracy:
                    best_accuracy = accuracy
                    best_s = self.modified_edge_weight_diff.clone().cpu()
        self.modified_edge_weight_diff.data.copy_(best_s.to(self.device))
        edge_index, edge_weight = self.get_modified_adj(is_final=True)

        edge_weight = edge_weight.round()
        edge_mask = edge_weight == 1
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def match_search_space_on_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.make_undirected:
            is_in_search_space = (edge_weight != 1) & (edge_index[0] < edge_index[1])
        else:
            is_in_search_space = (edge_weight != 1) & (edge_index[0] != edge_index[1])
        assert is_in_search_space.sum() == self.current_search_space.size(0), \
            f'search space size mismatch: {is_in_search_space.sum()} vs. {self.current_search_space.size(0)}'
        modified_edge_weight = edge_weight[is_in_search_space]
        original_edge_weight = modified_edge_weight - self.modified_edge_weight_diff
        does_original_edge_exist = torch.isclose(original_edge_weight.float(), torch.tensor(1.))

        return does_original_edge_exist, is_in_search_space

    def projection(self, n_perturbations: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> float:
        does_original_edge_exist, is_in_search_space = self.match_search_space_on_edges(edge_index, edge_weight)

        pos_modified_edge_weight_diff = torch.where(
            does_original_edge_exist, -self.modified_edge_weight_diff, self.modified_edge_weight_diff
        )
        probability_mass = pos_modified_edge_weight_diff.sum().item()

        pos_modified_edge_weight_diff = Attack.project(n_perturbations, pos_modified_edge_weight_diff, self.eps)

        self.modified_edge_weight_diff = torch.where(
            does_original_edge_exist, -pos_modified_edge_weight_diff, pos_modified_edge_weight_diff
        )

        return probability_mass, pos_modified_edge_weight_diff.sum().item()

    def handle_zeros_and_ones(self):
        # Handling edge case to detect an unchanged edge via its value 1
        self.modified_edge_weight_diff.data[
            (self.modified_edge_weight_diff <= self.eps)
            & (self.modified_edge_weight_diff >= -self.eps)
        ] = self.eps
        self.modified_edge_weight_diff.data[self.modified_edge_weight_diff >= 1 - self.eps] = 1 - self.eps
        self.modified_edge_weight_diff.data[self.modified_edge_weight_diff <= -1 + self.eps] = -1 + self.eps

    def get_modified_adj(self, is_final: bool = False):
        if not is_final:
            self.handle_zeros_and_ones()

        if (
            not self.modified_edge_weight_diff.requires_grad
            or not hasattr(self.attacked_model, 'do_checkpoint')
            or not self.attacked_model.do_checkpoint
        ):
            if self.make_undirected:
                modified_edge_index, modified_edge_weight = utils.to_symmetric(
                    self.modified_edge_index, self.modified_edge_weight_diff, self.n
                )
            else:
                modified_edge_index, modified_edge_weight = self.modified_edge_index, self.modified_edge_weight_diff
            edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
            edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

            edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
        else:
            # Currently (1.6.0) PyTorch does not support return arguments of `checkpoint` that do not require gradient.
            # For this reason we need this extra code and to execute it twice (due to checkpointing in fact 3 times...)
            from torch.utils import checkpoint

            def fuse_edges_run(modified_edge_weight_diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if self.make_undirected:
                    modified_edge_index, modified_edge_weight = utils.to_symmetric(
                        self.modified_edge_index, modified_edge_weight_diff, self.n
                    )
                else:
                    modified_edge_index, modified_edge_weight = self.modified_edge_index, self.modified_edge_weight_diff
                edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
                edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

                edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
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

            with torch.no_grad():
                edge_index = fuse_edges_run(self.modified_edge_weight_diff)[0]

            edge_weight = checkpoint.checkpoint(
                lambda *input: fuse_edges_run(*input)[1],
                self.modified_edge_weight_diff
            )

        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations: int, epoch: int,
                            gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the edge weights and adaptively, heuristically refined the learning rate such that (1) it is
        independent of the number of perturbations (assuming an undirected adjacency matrix) and (2) to decay learning
        rate during fine-tuning (i.e. fixed search space).

        Parameters
        ----------
        n_perturbations : int
            Number of perturbations.
        epoch : int
            Number of epochs until fine tuning.
        gradient : torch.Tensor
            The current gradient.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated edge indices and weights.
        """
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs) + 1)
        self.modified_edge_weight_diff.data.add_(lr * gradient)

        return self.get_modified_adj()

    def sample_search_space(self, n_perturbations: int = 0):
        for i in range(self.max_resamples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.block_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.modified_edge_weight_diff = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                break

    def resample_search_space(self, n_perturbations: int, edge_index: torch.Tensor,
                              edge_weight: torch.Tensor, gradient: torch.Tensor):
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.modified_edge_weight_diff.abs())
            idx_keep = (self.modified_edge_weight_diff <= self.eps).sum().long()
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.modified_edge_weight_diff = self.modified_edge_weight_diff[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_resamples):
            n_edges_resample = self.block_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)
            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )
            if self.make_undirected:
                self.modified_edge_index = PRBCD.linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = PRBCD.linear_to_full_idx(self.n, self.current_search_space)
            # Merge existing weights with new edge weights
            modified_edge_weight_diff_old = self.modified_edge_weight_diff.clone()
            self.modified_edge_weight_diff = self.eps * torch.ones_like(self.current_search_space, dtype=torch.float32)
            self.modified_edge_weight_diff[
                unique_idx[:modified_edge_weight_diff_old.size(0)]
            ] = modified_edge_weight_diff_old

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.modified_edge_weight_diff = self.modified_edge_weight_diff[is_not_self_loop]

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

    @staticmethod
    def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    def _append_attack_statistics(self, loss: float, accuracy: float,
                                  probability_mass: float, probability_mass_projected: float):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['accuracy'].append(accuracy)
        self.attack_statistics['nonzero_weights'].append((self.modified_edge_weight_diff.abs() > self.eps).sum().item())
        self.attack_statistics['probability_mass'].append(probability_mass)
        self.attack_statistics['probability_mass_projected'].append(probability_mass_projected)
