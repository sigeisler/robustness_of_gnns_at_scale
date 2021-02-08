from collections import defaultdict
from copy import deepcopy
import math
import logging
from typing import Dict, Optional, Union
import warnings

import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F
import torch
import torch_sparse
from torch_sparse import SparseTensor
from tqdm import tqdm

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.utils import calc_ppr_exact_row, calc_ppr_update_sparse_result, grad_with_checkpoint
from pprgo import ppr
from rgnn_at_scale.attacks.prbcd import PRBCD
from rgnn_at_scale.load_ppr import load_ppr, load_ppr_csr

from pprgo import utils as ppr_utils


class LocalPRBCD():

    def __init__(self,
                 adj: SparseTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: MODEL_TYPE,
                 device: Union[str, int, torch.device],
                 attack_labeled_nodes_only: bool = False,
                 ppr_matrix: Optional[SparseTensor] = None,
                 ppr_recalc_at_end: bool = False,
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

        logging.info(f'Memory before loading ppr: {ppr_utils.get_max_memory_bytes() / (1024 ** 3)}')

        self.device = device
        self.X = X
        self.model = model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.n, self.d = X.shape
        self.n_possible_edges = self.n * (self.n - 1) // 2
        self.labels = labels.long().to(self.device)
        self.idx_attack = idx_attack
        self.ppr_recalc_at_end = False
        self.attack_labeled_nodes_only = attack_labeled_nodes_only

        if type(model) in BATCHED_PPR_MODELS.__args__:
            if isinstance(adj, SparseTensor):
                adj = adj.to_scipy(layout="csr")
            self.ppr_alpha = model.alpha
            if self.attack_labeled_nodes_only:
                ppr_nodes = idx_attack
            else:
                ppr_nodes = np.arange(self.n)
            if ppr_matrix is None:
                if self.n == 111059956:
                    logging.info(f'model.alpha={model.alpha}, eps={model.eps}, topk={model.topk}, '
                                 f'ppr_normalization={model.ppr_normalization}')
                    self.ppr_matrix = load_ppr_csr(alpha=model.alpha,
                                                   eps=model.eps,
                                                   topk=model.topk,
                                                   ppr_normalization=model.ppr_normalization)
                else:
                    self.ppr_matrix = ppr.topk_ppr_matrix(adj, model.alpha, model.eps, ppr_nodes,
                                                          model.topk, normalization=model.ppr_normalization)
                if self.attack_labeled_nodes_only:
                    relabeled_row = torch.from_numpy(ppr_nodes)[self.ppr_matrix.storage.row()]
                    self.ppr_matrix = SparseTensor(row=relabeled_row, col=self.ppr_matrix.storage.col(),
                                                   value=self.ppr_matrix.storage.value(), sparse_sizes=(self.n, self.n))
            else:
                self.ppr_matrix = ppr_matrix
            self.ppr_recalc_at_end = ppr_recalc_at_end

            logging.info(f'self.ppr_matrix is of shape {self.ppr_matrix.shape}')
            logging.info(f'Memory after loading ppr: {ppr_utils.get_max_memory_bytes() / (1024 ** 3)}')

        if isinstance(adj, SparseTensor):
            self.adj = adj
        else:
            self.adj = SparseTensor.from_scipy(adj)

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
        self.modified_edge_weight_diff: torch.Tensor = None

        self.lr_factor = lr_factor
        self.lr_factor *= max(math.sqrt(self.n / self.search_space_size), 1.)

    def attack(self, node_idx: int, n_perturbations: int):
        self.sample_search_space(node_idx, n_perturbations)
        best_margin = float('Inf')
        best_epoch = float('-Inf')
        self.attack_statistics = defaultdict(list)

        with torch.no_grad():
            if type(self.model) in BATCHED_PPR_MODELS.__args__:
                updated_vector_or_graph_orig = SparseTensor.from_scipy(self.ppr_matrix[node_idx])
            else:
                updated_vector_or_graph_orig = self.adj.to(self.device)
                if not isinstance(updated_vector_or_graph_orig, SparseTensor):
                    updated_vector_or_graph_orig = SparseTensor.from_scipy(updated_vector_or_graph_orig)
            logits_orig = self.get_logits(node_idx, updated_vector_or_graph_orig)
            loss_orig = self.calculate_loss(logits_orig, self.labels[node_idx][None])
            statistics_orig = LocalPRBCD.classification_statistics(logits_orig,
                                                                   self.labels[node_idx])
            logging.info(f'Original: Loss: {loss_orig.item()} Statstics: {statistics_orig}\n')

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
                logging.info(f'Initial: Loss: {loss.item()} Statstics: {classification_statistics}\n')

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
                    logging.info(f'\nEpoch: {epoch} Loss: {loss.item()} Statstics: {classification_statistics}\n')

                if self.with_early_stropping and best_margin > classification_statistics['margin']:
                    best_margin = classification_statistics['margin']
                    best_epoch = epoch
                    best_search_space = self.current_search_space.clone().cpu()
                    best_edge_weight_diff = self.modified_edge_weight_diff.detach().clone().cpu()

                self._append_attack_statistics(loss.item(), classification_statistics)

                if epoch < self.epochs - 1:
                    self.resample_search_space(node_idx, n_perturbations, gradient)
                elif self.with_early_stropping and epoch == self.epochs - 1:
                    print(f'Loading search space of epoch {best_epoch} (margin={best_margin}) for fine tuning\n')
                    self.current_search_space = best_search_space.clone().to(self.device)
                    self.modified_edge_weight_diff = best_edge_weight_diff.clone().to(self.device)

            del logits
            del loss
            del gradient

        # For the case that the attack was not successfull
        if best_margin > statistics_orig['margin']:
            self.perturbed_edges = torch.tensor([])
            return logits_orig, logits_orig

        if self.with_early_stropping:
            self.current_search_space = best_search_space.to(self.device)
            self.modified_edge_weight_diff = best_edge_weight_diff.to(self.device)

        updated_vector_or_graph = self.sample_edges(node_idx, n_perturbations)
        logits = self.get_logits(node_idx, updated_vector_or_graph)
        self.perturbed_edges = self.get_perturbed_edges(node_idx)

        if self.ppr_recalc_at_end:
            adj = self.get_updated_vector_or_graph(node_idx, only_update_adj=True)
            # Handle disconnected nodes
            disconnected_nodes = (adj.sum(0) == 0).nonzero().flatten()
            if disconnected_nodes.nelement():
                adj = SparseTensor(row=torch.cat((adj.storage.row(), disconnected_nodes)),
                                   col=torch.cat((adj.storage.col(), disconnected_nodes)),
                                   value=torch.cat((adj.storage.col(), torch.full_like(disconnected_nodes, 1e-9))))
            if type(self.model) in BATCHED_PPR_MODELS.__args__:
                self.model.topk = self.model.topk + n_perturbations
                try:
                    logits = F.log_softmax(self.model.forward(self.X, adj, ppr_idx=np.array([node_idx])), dim=-1)
                except:
                    print('sdf')
                self.model.topk = self.model.topk - n_perturbations
            else:
                return self.model(data=self.X.to(self.device), adj=updated_vector_or_graph)[node_idx:node_idx + 1]
            # from pprgo.predict import predict_power_iter
            # predict_power_iter(self.model, adj.to_scipy(layout="csr"),
            #                    SparseTensor.from_dense(self.X).to_scipy(layout="csr"), alpha=0.109536, nprop=5,
            #                    ppr_normalization='row')[0][node_idx]

        return logits.detach(), logits_orig.detach()

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

    def get_updated_vector_or_graph(self, node_idx: int, only_update_adj: bool = False) -> SparseTensor:

        if self.attack_labeled_nodes_only:
            current_search_space = torch.tensor(self.idx_attack, device=self.device)[self.current_search_space]
        else:
            current_search_space = self.current_search_space
        modified_edge_weight_diff = SparseTensor(row=torch.zeros_like(self.current_search_space),
                                                 col=current_search_space,
                                                 value=self.modified_edge_weight_diff,
                                                 sparse_sizes=(1, self.n))

        if type(self.model) in BATCHED_PPR_MODELS.__args__ and not only_update_adj:
            A_row = self.adj[node_idx].to(self.device)
            if not isinstance(A_row, SparseTensor):
                A_row = SparseTensor.from_scipy(A_row)
            updated_vector_or_graph = calc_ppr_update_sparse_result(self.ppr_matrix, A_row,
                                                                    modified_edge_weight_diff, node_idx, self.ppr_alpha)
            return updated_vector_or_graph
        else:
            v_rows, v_cols, v_vals = modified_edge_weight_diff.coo()
            v_rows += node_idx
            v_idx = torch.stack([v_rows, v_cols], dim=0)

            A_rows, A_cols, A_vals = self.adj.to(v_vals.device).coo()
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

            updated_adj = SparseTensor.from_edge_index(A_idx, A_weights, (self.n, self.n))

            return updated_adj.to_symmetric('max')

    def update_edge_weights(self, n_perturbations: int, epoch: int, gradient: torch.Tensor):
        lr_factor = n_perturbations * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs) + 1)
        self.modified_edge_weight_diff.data.add_(lr * gradient)

    @ staticmethod
    def classification_statistics(logits, label) -> Dict[str, float]:
        logits, label = logits.cpu(), label.cpu()
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

    def _append_attack_statistics(self, loss, statistics):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['perturbation_mass'].append(
            torch.clamp(self.modified_edge_weight_diff, 0, 1).sum().item()
        )
        for key, value in statistics.items():
            self.attack_statistics[key].append(value)

    def sample_search_space(self, node_idx: int, n_perturbations: int):
        if self.attack_labeled_nodes_only:
            n = len(self.idx_attack)
        else:
            n = self.n

        while True:
            self.current_search_space = torch.randint(n, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            #self.current_search_space = self.current_search_space[node_idx != self.current_search_space]
            self.modified_edge_weight_diff = torch.full_like(self.current_search_space, self.eps,
                                                             dtype=torch.float32, requires_grad=True)
            if self.current_search_space.size(0) >= n_perturbations:
                break

    def resample_search_space(self, node_idx: int, n_perturbations: int, gradient: torch.Tensor):
        if self.attack_labeled_nodes_only:
            n = len(self.idx_attack)
        else:
            n = self.n

        sorted_idx = torch.argsort(self.modified_edge_weight_diff)
        idx_keep_not = (self.modified_edge_weight_diff <= self.eps).sum().long()
        if idx_keep_not < sorted_idx.size(0) // 2:
            idx_keep_not = sorted_idx.size(0) // 2

        sorted_idx = sorted_idx[idx_keep_not:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_weight_diff = self.modified_edge_weight_diff[sorted_idx]

        # Sample until enough edges were drawn
        while True:
            new_index = torch.randint(n,
                                      (self.search_space_size - self.current_search_space.size(0),),
                                      device=self.device)
            self.current_search_space = torch.cat((self.current_search_space, new_index))
            #self.current_search_space = self.current_search_space[node_idx != self.current_search_space]
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
                    self.current_search_space = current_search_space[
                        self.modified_edge_weight_diff == 1
                    ].to(self.device)
                    self.modified_edge_weight_diff = self.modified_edge_weight_diff[
                        self.modified_edge_weight_diff == 1
                    ].to(self.device)

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

    # TODO: ...
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
        elif self.loss_type == 'Margin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels.long()]
                - logits[np.arange(logits.size(0)), best_non_target_class.long()]
            )
            loss = -margin.mean()
        # TODO: Is it worth trying? CW should be quite similar
        # elif self.loss_type == 'Margin':
        #    loss = F.multi_margin_loss(torch.exp(logits), labels)
        else:
            loss = F.nll_loss(logits, labels)
        return loss
