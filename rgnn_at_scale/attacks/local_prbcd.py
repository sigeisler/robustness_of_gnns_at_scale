from collections import defaultdict

import math
import logging
from typeguard import typechecked

import numpy as np

import torch
import torch_sparse
from torch_sparse import SparseTensor

from tqdm import tqdm

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.helper.utils import grad_with_checkpoint, to_symmetric
from rgnn_at_scale.attacks.base_attack import Attack, SparseLocalAttack


class LocalPRBCD(SparseLocalAttack):

    @typechecked
    def __init__(self,
                 loss_type: str = 'Margin',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 lr_factor: float = 1.0,
                 display_step: int = 20,
                 epochs: int = 150,
                 fine_tune_epochs: int = 50,
                 block_size: int = 10_000,
                 with_early_stopping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-14,
                 final_samples: int = 20,
                 **kwargs):

        super().__init__(**kwargs)

        # Late import to prevent circular import
        from rgnn_at_scale.attacks.local_prbcd_batched import LocalBatchedPRBCD
        assert isinstance(self, LocalBatchedPRBCD) or type(
            self.attacked_model) not in BATCHED_PPR_MODELS.__args__, \
            "'LocalPRBCD' does not support batched models, use 'LocalBatchedPRBCD' instead"

        self.loss_type = loss_type
        self.n_possible_edges = self.n * (self.n - 1) // 2

        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.epochs_resampling = epochs - fine_tune_epochs
        self.block_size = block_size
        self.with_early_stopping = with_early_stopping
        self.eps = eps
        self.do_synchronize = do_synchronize
        self.final_samples = final_samples

        self.current_search_space: torch.Tensor = None
        self.modified_edge_weight_diff: torch.Tensor = None

        self.lr_factor = lr_factor
        self.lr_factor *= max(math.sqrt(self.n / self.block_size), 1.)

    def _attack(self, n_perturbations: int, node_idx: int, **kwargs):

        self.sample_search_space(node_idx, n_perturbations)
        best_margin = float('Inf')
        best_epoch = float('-Inf')
        self.attack_statistics = defaultdict(list)

        with torch.no_grad():
            logits_orig = self.get_surrogate_logits(node_idx).to(self.device)
            loss_orig = self.calculate_loss(logits_orig, self.labels[node_idx, None]).to(self.device)
            statistics_orig = LocalPRBCD.classification_statistics(logits_orig, self.labels[node_idx])
            logging.info(f'Original: Loss: {loss_orig.item()} Statstics: {statistics_orig}\n')
            del logits_orig
            del loss_orig

        for epoch in tqdm(range(self.epochs)):
            self.modified_edge_weight_diff.requires_grad = True
            perturbed_graph = self.perturb_graph(node_idx)

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self.get_surrogate_logits(node_idx, perturbed_graph).to(self.device)
            loss = self.calculate_loss(logits, self.labels[node_idx][None])

            if epoch == 0:
                classification_statistics = LocalPRBCD.classification_statistics(
                    logits, self.labels[node_idx].to(self.device))
                logging.info(f'Initial: Loss: {loss.item()} Statstics: {classification_statistics}\n')

            gradient = grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                self.modified_edge_weight_diff.requires_grad = False
                self.update_edge_weights(n_perturbations, epoch, gradient)

                self.modified_edge_weight_diff = Attack.project(
                    n_perturbations, self.modified_edge_weight_diff, self.eps
                )

                perturbed_graph = self.perturb_graph(node_idx)
                logits = self.get_surrogate_logits(node_idx, perturbed_graph).to(self.device)
                classification_statistics = LocalPRBCD.classification_statistics(
                    logits, self.labels[node_idx].to(self.device))
                if epoch % self.display_step == 0:
                    logging.info(f'\nEpoch: {epoch} Loss: {loss.item()} Statstics: {classification_statistics}\n')
                    logging.info(f"Gradient mean {gradient.abs().mean().item()} std {gradient.abs().std().item()} "
                                 f"with base learning rate {n_perturbations * self.lr_factor}")
                    if torch.cuda.is_available():
                        logging.info(f'Cuda memory {torch.cuda.memory_allocated() / (1024 ** 3)}')

                if self.with_early_stopping and best_margin > classification_statistics['margin']:
                    best_margin = classification_statistics['margin']
                    best_epoch = epoch
                    best_search_space = self.current_search_space.clone().cpu()
                    best_edge_weight_diff = self.modified_edge_weight_diff.detach().clone().cpu()

                self._append_attack_statistics(loss.item(), classification_statistics)

                if epoch < self.epochs_resampling - 1:
                    self.resample_search_space(node_idx, n_perturbations, gradient)
                elif self.with_early_stopping and epoch == self.epochs_resampling - 1:
                    logging.info(
                        f'Loading search space of epoch {best_epoch} (margin={best_margin}) for fine tuning\n')
                    self.current_search_space = best_search_space.clone().to(self.device)
                    self.modified_edge_weight_diff = best_edge_weight_diff.clone().to(self.device)

            del logits
            del loss
            del gradient

        # For the case that the attack was not successfull
        if best_margin > statistics_orig['margin']:
            self.perturbed_edges = torch.tensor([])
            self.adj_adversary = None
            self.attr_adversary = self.attr
            logging.info(f"Failed to attack node {node_idx} with n_perturbations={n_perturbations}")
            return None

        if self.with_early_stopping:
            self.current_search_space = best_search_space.to(self.device)
            self.modified_edge_weight_diff = best_edge_weight_diff.to(self.device)

        if torch.cuda.is_available() and self.do_synchronize:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.adj_adversary = self.sample_final_edges(node_idx, n_perturbations)
        self.perturbed_edges = self.calc_perturbed_edges(node_idx)

        if torch.cuda.is_available() and self.do_synchronize:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None):
        if perturbed_graph is None:
            perturbed_graph = self.adj

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return model.forward(self.attr.to(self.device), perturbed_graph, ppr_idx=np.array([node_idx]))
        else:
            return model(data=self.attr.to(self.device), adj=perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def calc_perturbed_edges(self, node_idx: int) -> torch.Tensor:
        source = torch.full_like(self.current_search_space, node_idx).cpu()
        target = self.current_search_space.cpu()

        return torch.stack((source, target), dim=0)

    def get_perturbed_edges(self) -> torch.Tensor:
        if not hasattr(self, "perturbed_edges"):
            return torch.tensor([])
        return self.perturbed_edges

    def perturb_graph(self, node_idx: int) -> SparseTensor:
        modified_edge_weight_diff = SparseTensor(row=torch.zeros_like(self.current_search_space),
                                                 col=self.current_search_space,
                                                 value=self.modified_edge_weight_diff,
                                                 sparse_sizes=(1, self.n))

        device = self.modified_edge_weight_diff.device
        updated_adj = LocalPRBCD.mod_row(modified_edge_weight_diff, self.adj.to(device), node_idx, self.make_undirected)

        return updated_adj

    @staticmethod
    def mod_row(modified: SparseTensor, adj: SparseTensor, row_idx: int, make_undirected: bool) -> SparseTensor:
        n = adj.size(0)

        v_rows, v_cols, v_vals = modified.coo()
        v_rows += row_idx
        v_idx = torch.stack([v_rows, v_cols], dim=0)

        A_rows, A_cols, A_vals = adj.coo()
        A_idx = torch.stack([A_rows, A_cols], dim=0)

        # select only changed row
        is_row = A_rows == row_idx
        A_idx_row = A_idx[:, is_row]
        A_vals_row = A_vals[is_row]

        # sparse addition: row = A[i] + v
        A_idx_row = torch.cat((v_idx, A_idx_row), dim=-1)
        A_vals_row = torch.cat((v_vals, A_vals_row))

        A_idx_row, A_vals_row = torch_sparse.coalesce(A_idx_row, A_vals_row, m=n, n=n, op='sum')

        is_before = A_rows < row_idx
        is_after = A_rows > row_idx

        A_idx = torch.cat((A_idx[:, is_before], A_idx_row, A_idx[:, is_after]), dim=-1)
        A_weights = torch.cat((A_vals[is_before], A_vals_row, A_vals[is_after]), dim=-1)

        # Works since the attack will always assign at least a small constant the elements in p
        A_weights[A_weights > 1] = -A_weights[A_weights > 1] + 2

        if make_undirected:
            A_idx, A_weights = to_symmetric(A_idx, A_weights, n, op='max')

        return SparseTensor.from_edge_index(A_idx, A_weights, (n, n))

    def update_edge_weights(self, n_perturbations: int, epoch: int, gradient: torch.Tensor):
        lr_factor = n_perturbations * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.modified_edge_weight_diff.data.add_(lr * gradient)

    def _append_attack_statistics(self, loss, statistics):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['perturbation_mass'].append(
            torch.clamp(self.modified_edge_weight_diff, 0, 1).sum().item()
        )
        for key, value in statistics.items():
            self.attack_statistics[key].append(value)

    def sample_search_space(self, node_idx: int, n_perturbations: int):
        while True:
            self.current_search_space = torch.randint(self.n - 1, (self.block_size,), device=self.device)
            self.current_search_space[self.current_search_space >= node_idx] += 1
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            self.modified_edge_weight_diff = torch.full_like(self.current_search_space, self.eps,
                                                             dtype=torch.float32, requires_grad=True)
            if self.current_search_space.size(0) >= n_perturbations:
                break

    def resample_search_space(self, node_idx: int, n_perturbations: int, gradient: torch.Tensor):
        sorted_idx = torch.argsort(self.modified_edge_weight_diff)
        idx_don_not_keep = (self.modified_edge_weight_diff <= self.eps).sum().long()
        if idx_don_not_keep < sorted_idx.size(0) // 2:
            idx_don_not_keep = sorted_idx.size(0) // 2

        sorted_idx = sorted_idx[idx_don_not_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_weight_diff = self.modified_edge_weight_diff[sorted_idx]

        # Sample until enough edges were drawn
        while True:
            number_new_edges = self.block_size - self.current_search_space.size(0)
            new_index = torch.randint(self.n - 1, (number_new_edges,), device=self.device)
            new_index[new_index >= node_idx] += 1
            self.current_search_space = torch.cat((self.current_search_space, new_index))

            self.current_search_space, unique_idx = torch.unique(
                self.current_search_space,
                sorted=True,
                return_inverse=True
            )
            # Merge existing weights with new edge weights
            modified_edge_weight_diff_old = self.modified_edge_weight_diff.clone()
            self.modified_edge_weight_diff = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.modified_edge_weight_diff[
                unique_idx[:modified_edge_weight_diff_old.size(0)]
            ] = modified_edge_weight_diff_old

            if self.current_search_space.size(0) > n_perturbations:
                break

    @torch.no_grad()
    def sample_final_edges(self, node_idx: int, n_perturbations: int) -> SparseTensor:
        best_margin = float('Inf')
        current_search_space = self.current_search_space.clone()

        s = self.modified_edge_weight_diff.abs().detach()
        s[s == self.eps] = 0
        while best_margin == float('Inf'):
            for i in range(self.final_samples):
                if best_margin == float('Inf'):
                    sampled = torch.zeros_like(s)
                    sampled[torch.topk(s, n_perturbations).indices] = 1
                else:
                    sampled = torch.bernoulli(s).float()

                if sampled.sum() > n_perturbations or sampled.sum() == 0:
                    continue

                self.modified_edge_weight_diff = sampled
                keep_mask = self.modified_edge_weight_diff == 1
                self.current_search_space = current_search_space[keep_mask].to(self.device)
                self.modified_edge_weight_diff = self.modified_edge_weight_diff[keep_mask].to(self.device)

                perturbed_graph = self.perturb_graph(node_idx)
                logits = self.get_surrogate_logits(node_idx, perturbed_graph)
                margin = LocalPRBCD.classification_statistics(logits, self.labels[node_idx])['margin']
                if best_margin > margin:
                    best_margin = margin
                    best_weights = self.modified_edge_weight_diff.cpu()
                    best_search_space = self.current_search_space.cpu()
        self.modified_edge_weight_diff = best_weights.to(self.device).float()
        self.current_search_space = best_search_space.to(self.device).long()

        perturbed_graph = self.perturb_graph(node_idx)
        return perturbed_graph
