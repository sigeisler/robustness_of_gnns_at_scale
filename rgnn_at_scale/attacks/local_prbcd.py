from collections import defaultdict
from copy import deepcopy
import math
from typing import Dict, Optional, Union
import warnings

from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.utils import calc_ppr_exact_row, calc_ppr_update_sparse_result, grad_with_checkpoint
from pprgo import ppr
from rgnn_at_scale.attacks.prbcd import PRBCD


class LocalPRBCD(PRBCD):

    def __init__(self,
                 adj: SparseTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 ppr_matrix: Optional[SparseTensor] = None,
                 loss_type: str = 'Margin',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 lr_factor: float = 1.0,
                 lr_n_perturbations_factor: float = 0.1,
                 display_step: int = 20,
                 epochs: int = 400,
                 fine_tune_epochs: int = 100,
                 search_space_size: int = 10_000,
                 with_early_stropping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-14,
                 K: int = 20,
                 **kwargs):

        super().__init__(adj, X, labels, idx_attack, model, device)
        self.device = device
        self.X = X
        self.model = deepcopy(model).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # TODO: Replace adj or remove
        edge_index_rows, edge_index_cols, edge_weight = adj.coo()
        #diagonal_mask = edge_index_rows == edge_index_cols
        #edge_index_rows = edge_index_rows[diagonal_mask]
        #edge_index_cols = edge_index_cols[diagonal_mask]
        #edge_weight = edge_weight[diagonal_mask]
        self.adj = SparseTensor(row=edge_index_rows, col=edge_index_cols, value=edge_weight, sparse_sizes=adj.sizes())

        self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).cpu()
        self.edge_weight = edge_weight.cpu()
        self.n = adj.size(0)
        self.n_possible_edges = self.n * (self.n - 1) // 2
        self.d = X.shape[1]
        self.labels = labels.to(self.device)
        self.idx_attack = idx_attack
        if type(model) in BATCHED_PPR_MODELS.__args__:
            self.ppr_alpha = model.alpha
            if ppr_matrix is None:
                self.ppr_matrix = SparseTensor.from_scipy(
                    ppr.topk_ppr_matrix(adj.to_scipy(layout="csr"), model.alpha, model.eps,
                                        np.arange(self.n), model.topk,  normalization=model.ppr_normalization)
                ).to(device)
            else:
                self.ppr_matrix = ppr_matrix

        self.loss_type = loss_type
        self.lr_n_perturbations_factor = lr_n_perturbations_factor
        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.search_space_size = search_space_size
        self.with_early_stropping = with_early_stropping
        self.eps = eps
        self.do_synchronize = do_synchronize
        # TODO: Rename
        self.K = K

        self.adj_adversary = None
        self.attr_adversary = None

        self.current_search_space: torch.Tensor = None
        self.modified_edge_index: torch.Tensor = None
        self.modified_edge_weight_diff: torch.Tensor = None

        self.lr_factor = lr_factor
        self.lr_factor *= max(math.sqrt(self.n / self.search_space_size), 1.)

    def attack(self, node_idx: int, n_perturbations: int):
        self.sample_search_space(node_idx, n_perturbations)
        self.attack_statistics = defaultdict(list)

        for epoch in tqdm(range(self.epochs + self.fine_tune_epochs)):
            self.modified_edge_weight_diff.requires_grad = True
            updated_vector_or_graph = self.get_updated_vector_or_graph(node_idx)

            if self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self.get_logits(node_idx, updated_vector_or_graph)
            loss = self.calculate_loss(logits, self.labels[node_idx][None])

            classification_statistics = LocalPRBCD.classification_statistics(logits, self.labels[node_idx])
            if epoch == 0:
                print(f'Initial: Loss: {loss.item()} Statstics: {classification_statistics}\n')
                with torch.no_grad():
                    if type(self.model) in BATCHED_PPR_MODELS.__args__:
                        updated_vector_or_graph_orig = self.ppr_matrix[node_idx]
                    else:
                        updated_vector_or_graph_orig = self.adj.to(self.device)
                    logits_orig = self.get_logits(node_idx, updated_vector_or_graph_orig)
                    loss_orig = self.calculate_loss(logits_orig, self.labels[node_idx][None])
                    statistics_orig = LocalPRBCD.classification_statistics(logits_orig,
                                                                           self.labels[node_idx])
                    print(f'Original: Loss: {loss_orig.item()} Statstics: {statistics_orig}\n')

            gradient = grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            if self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                self.modified_edge_weight_diff.requires_grad = False
                self.update_edge_weights(n_perturbations, epoch, gradient)

                self.modified_edge_weight_diff = PRBCD.project(
                    n_perturbations, self.modified_edge_weight_diff, self.eps
                )

                updated_vector_or_graph = self.get_updated_vector_or_graph(node_idx)
                logits = self.get_logits(node_idx, updated_vector_or_graph)
                classification_statistics = LocalPRBCD.classification_statistics(logits, self.labels[node_idx])
                if epoch % self.display_step == 0:
                    print(f'\nEpoch: {epoch} Loss: {loss.item()} Statstics: {classification_statistics}\n')

                self._append_attack_statistics(loss.item(), classification_statistics)

                self.resample_search_space(node_idx, n_perturbations, gradient)

            del logits
            del loss
            del gradient

        updated_vector_or_graph = self.sample_edges(node_idx, n_perturbations)
        logits = self.get_logits(node_idx, updated_vector_or_graph)
        self.perturbed_edges = self.get_perturbed_edges(node_idx)

        return logits

    def get_logits(self, node_idx: int, updated_vector_or_graph: SparseTensor) -> torch.Tensor:
        if type(self.model) in BATCHED_PPR_MODELS.__args__:
            return F.log_softmax(self.model.forward(self.X, None, ppr_scores=updated_vector_or_graph), dim=-1)
        else:
            return self.model(data=self.X.to(self.device), adj=updated_vector_or_graph)[node_idx:node_idx + 1]

    def get_perturbed_edges(self, node_idx: int) -> torch.Tensor:
        source = torch.full_like(self.current_search_space, node_idx).cpu()
        target = self.current_search_space.cpu()
        flip_order_mask = source > target
        source, target = torch.where(~flip_order_mask, source, target), torch.where(flip_order_mask, source, target)
        return torch.stack((source, target), dim=0)

    def get_updated_vector_or_graph(self, node_idx: int) -> SparseTensor:
        modified_edge_weight_diff = SparseTensor(row=torch.zeros_like(self.current_search_space),
                                                 col=self.current_search_space,
                                                 value=self.modified_edge_weight_diff,
                                                 sparse_sizes=(1, self.n))
        if type(self.model) in BATCHED_PPR_MODELS.__args__:
            A_row = self.adj[node_idx].to(self.device)
            updated_vector_or_graph = calc_ppr_update_sparse_result(self.ppr_matrix, A_row,
                                                                    modified_edge_weight_diff, node_idx, self.ppr_alpha)
            # updated_vector_or_graph /= updated_vector_or_graph.sum()

            # modified_adj = SparseTensor(
            #     row=torch.full_like(self.current_search_space, node_idx),
            #     col=self.current_search_space,
            #     value=self.modified_edge_weight_diff,
            #     sparse_sizes=(self.n, self.n)
            # ).to_scipy(layout="csr") + self.adj.to_scipy(layout="csr")

            u = torch.zeros((self.n, 1), dtype=torch.float32, device=self.device)
            u[node_idx] = 1
            p_dense = modified_edge_weight_diff.to_dense()
            A_dense = self.adj.to(self.device).to_dense()
            v = torch.where(A_dense[node_idx, :] > 0, -p_dense, p_dense)
            A_pert = A_dense + u @ v
            exact_ppr_vector = calc_ppr_exact_row(A_pert, alpha=self.ppr_alpha)[node_idx]

            approx_ppr_vector = SparseTensor.from_scipy(
                ppr.topk_ppr_matrix(SparseTensor.from_dense(A_pert).to_scipy(layout="csr"), self.model.alpha,
                                    self.model.eps, np.arange(self.n), self.model.topk,
                                    normalization=self.model.ppr_normalization)
            ).to(self.device)[node_idx]
            # approx_ppr_vector.storage._value /= approx_ppr_vector.sum()

            diff_approx_exact = torch.norm(approx_ppr_vector.to_dense() - exact_ppr_vector)
            diff_approx_updated = torch.norm(approx_ppr_vector.to_dense() - updated_vector_or_graph)
            diff_exact_updated = torch.norm(exact_ppr_vector - updated_vector_or_graph)

            if diff_exact_updated > diff_approx_exact or diff_exact_updated > diff_approx_updated:
                warnings.warn(f'Error os large diff_approx_exact={diff_approx_exact:.3f} '
                              f'diff_approx_updated={diff_approx_updated:.3f} '
                              f'diff_exact_updated={diff_exact_updated:.3f}')

            return SparseTensor.from_dense(updated_vector_or_graph)
        else:
            v_rows, v_cols, v_vals = modified_edge_weight_diff.coo()
            v_rows += node_idx
            v_idx = torch.stack([v_rows, v_cols], dim=0)

            A_rows, A_cols, A_vals = self.adj.to(self.device).coo()
            A_idx = torch.stack([A_rows, A_cols], dim=0)

            # sparse addition: row = A[i] + v
            A_idx = torch.cat((v_idx, A_idx), dim=-1)
            A_weights = torch.cat((v_vals, A_vals))
            A_idx, A_weights = torch_sparse.coalesce(
                A_idx,
                A_weights,
                m=1,
                n=self.n,
                op='sum'
            )

            # Works since the attack will always assign at least a small constant the elements in p
            A_weights[A_weights > 1] = -A_weights[A_weights > 1] + 2

            return SparseTensor.from_edge_index(A_idx, A_weights, (self.n, self.n))

    def update_edge_weights(self, n_perturbations: int, epoch: int, gradient: torch.Tensor):
        lr_factor = max(1., n_perturbations / self.n / 2 / self.lr_n_perturbations_factor) * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs) + 1)
        self.modified_edge_weight_diff.data.add_(lr * gradient)

    @ staticmethod
    def classification_statistics(logits, label) -> Dict[str, float]:
        logits = logits[0]
        logit_target = logits[label].item()
        sorted = logits.argsort()
        logit_best_non_target = logits[sorted[sorted != label][-1]].item()
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

    def _append_attack_statistics(self, loss, statistics):
        self.attack_statistics['loss'].append(loss)
        for key, value in statistics.items():
            self.attack_statistics[key].append(value)

    def sample_search_space(self, node_idx: int, n_perturbations: int):
        while True:
            self.current_search_space = torch.randint(self.n, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            self.current_search_space = self.current_search_space[node_idx != self.current_search_space]
            self.modified_edge_weight_diff = torch.full_like(self.current_search_space, self.eps,
                                                             dtype=torch.float32, requires_grad=True)
            if self.current_search_space.size(0) >= n_perturbations:
                break

    def resample_search_space(self, node_idx: int, n_perturbations: int, gradient: torch.Tensor):
        sorted_idx = torch.argsort(self.modified_edge_weight_diff)
        idx_keep_not = (self.modified_edge_weight_diff <= self.eps).sum().long()
        if idx_keep_not < sorted_idx.size(0) // 2:
            idx_keep_not = sorted_idx.size(0) // 2

        sorted_idx = sorted_idx[idx_keep_not:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_weight_diff = self.modified_edge_weight_diff[sorted_idx]

        # Sample until enough edges were drawn
        while True:
            new_index = torch.randint(self.n,
                                      (self.search_space_size - self.current_search_space.size(0),),
                                      device=self.device)
            self.current_search_space = torch.cat((self.current_search_space, new_index))
            self.current_search_space = self.current_search_space[node_idx != self.current_search_space]
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

    def sample_edges(self, node_idx: int, n_perturbations: int) -> SparseTensor:
        best_margin = float('Inf')
        with torch.no_grad():
            current_search_space = self.current_search_space.clone()
            s = self.modified_edge_weight_diff.abs().detach()
            s[s == self.eps] = 0
            # TODO: Why numpy?
            #s = s.cpu().numpy()
            while best_margin == float('Inf'):
                for i in range(self.K):
                    if best_margin == float('Inf'):
                        sampled = torch.zeros_like(s)
                        sampled[torch.topk(s, n_perturbations).indices] = 1
                    else:
                        sampled = torch.bernoulli(s).float()

                    if sampled.sum() > n_perturbations or sampled.sum() == 0:
                        continue

                    self.modified_edge_weight_diff = sampled
                    self.current_search_space = current_search_space[self.modified_edge_weight_diff == 1]
                    self.modified_edge_weight_diff = self.modified_edge_weight_diff[self.modified_edge_weight_diff == 1]

                    updated_vector_or_graph = self.get_updated_vector_or_graph(node_idx)
                    logits = self.get_logits(node_idx, updated_vector_or_graph)
                    margin = LocalPRBCD.classification_statistics(logits, self.labels[node_idx])['margin']
                    if best_margin > margin:
                        best_margin = margin
                        best_weights = self.modified_edge_weight_diff.clone().cpu()
                        best_search_space = self.current_search_space.clone().cpu()
            self.modified_edge_weight_diff = best_weights.to(self.device).float()
            self.current_search_space = best_search_space.to(self.device).long()

            updated_vector_or_graph = self.get_updated_vector_or_graph(node_idx)
        return updated_vector_or_graph
