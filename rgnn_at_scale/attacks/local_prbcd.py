from collections import defaultdict
from typing import Dict, Union

import math
import logging

import numpy as np
import scipy.sparse as sp

import torch
from torch.nn import functional as F
import torch_sparse
from torch_sparse import SparseTensor

from tqdm import tqdm

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.helper.utils import grad_with_checkpoint

from rgnn_at_scale.attacks.base_attack import Attack, SparseLocalAttack


class LocalPRBCD(SparseLocalAttack):

    def __init__(self,
                 loss_type: str = 'Margin',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
                 attack_labeled_nodes_only: bool = False,
                 lr_factor: float = 1.0,
                 display_step: int = 20,
                 epochs: int = 150,
                 fine_tune_epochs: int = 50,
                 search_space_size: int = 10_000,
                 with_early_stropping: bool = True,
                 do_synchronize: bool = False,
                 eps: float = 1e-14,
                 final_samples: int = 20,
                 **kwargs):

        super().__init__(**kwargs)

        self.loss_type = loss_type
        self.n_possible_edges = self.n * (self.n - 1) // 2
        self.attack_labeled_nodes_only = attack_labeled_nodes_only

        self.display_step = display_step
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.search_space_size = search_space_size
        self.with_early_stropping = with_early_stropping
        self.eps = eps
        self.do_synchronize = do_synchronize
        # TODO: Rename
        self.final_samples = final_samples

        self.current_search_space: torch.Tensor = None
        self.modified_edge_weight_diff: torch.Tensor = None

        self.lr_factor = lr_factor
        self.lr_factor *= max(math.sqrt(self.n / self.search_space_size), 1.)

    def _attack(self, n_perturbations: int, node_idx: int, **kwargs):
        self.sample_search_space(node_idx, n_perturbations)
        best_margin = float('Inf')
        best_epoch = float('-Inf')
        self.attack_statistics = defaultdict(list)

        with torch.no_grad():
            logits_orig = self.get_surrogate_logits(node_idx).to(self.device)
            loss_orig = self.calculate_loss(logits_orig, self.labels[node_idx][None]).to(self.device)
            statistics_orig = LocalPRBCD.classification_statistics(logits_orig,
                                                                   self.labels[node_idx])
            logging.info(f'Original: Loss: {loss_orig.item()} Statstics: {statistics_orig}\n')

        for epoch in tqdm(range(self.epochs + self.fine_tune_epochs)):
            self.modified_edge_weight_diff.requires_grad = True
            perturbed_graph = self.perturb_graph(node_idx)

            if self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self.get_surrogate_logits(node_idx, perturbed_graph).to(self.device)
            loss = self.calculate_loss(logits, self.labels[node_idx][None])

            classification_statistics = LocalPRBCD.classification_statistics(
                logits, self.labels[node_idx].to(self.device))
            if epoch == 0:
                logging.info(f'Initial: Loss: {loss.item()} Statstics: {classification_statistics}\n')

            gradient = grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            if self.do_synchronize:
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

        self.adj_adversary = self.sample_final_edges(node_idx, n_perturbations)
        self.perturbed_edges = self.calc_perturbed_edges(node_idx)

    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None):
        if perturbed_graph is None:
            perturbed_graph = self.adj

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return F.log_softmax(model.forward(self.X.to(self.device), perturbed_graph, ppr_idx=np.array([node_idx])), dim=-1)
        else:
            return model(data=self.X.to(self.device), adj=perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def sample_final_edges(self, node_idx: int, n_perturbations: int):
        perturbed_graph = self.sample_edges(node_idx, n_perturbations)
        return perturbed_graph

    def calc_perturbed_edges(self, node_idx: int) -> torch.Tensor:
        source = torch.full_like(self.current_search_space, node_idx).cpu()
        target = self.current_search_space.cpu()
        #flip_order_mask = source > target
        #source, target = torch.where(~flip_order_mask, source, target), torch.where(flip_order_mask, source, target)
        return torch.stack((source, target), dim=0)

    def get_perturbed_edges(self) -> torch.Tensor:
        if not hasattr(self, "perturbed_edges"):
            return torch.tensor([])
        return self.perturbed_edges

    def perturb_graph(self, node_idx: int) -> SparseTensor:

        if self.attack_labeled_nodes_only:
            current_search_space = torch.tensor(self.idx_attack, device=self.device)[self.current_search_space]
        else:
            current_search_space = self.current_search_space
        modified_edge_weight_diff = SparseTensor(row=torch.zeros_like(self.current_search_space),
                                                 col=current_search_space,
                                                 value=self.modified_edge_weight_diff,
                                                 sparse_sizes=(1, self.n))

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
                    self.current_search_space = current_search_space[
                        self.modified_edge_weight_diff == 1
                    ].to(self.device)
                    self.modified_edge_weight_diff = self.modified_edge_weight_diff[
                        self.modified_edge_weight_diff == 1
                    ].to(self.device)

                    perturbed_graph = self.perturb_graph(node_idx)
                    logits = self.get_surrogate_logits(node_idx, perturbed_graph)
                    margin = LocalPRBCD.classification_statistics(logits, self.labels[node_idx])['margin']
                    if best_margin > margin:
                        best_margin = margin
                        best_weights = self.modified_edge_weight_diff.clone().cpu()
                        best_search_space = self.current_search_space.clone().cpu()
            self.modified_edge_weight_diff = best_weights.to(self.device).float()
            self.current_search_space = best_search_space.to(self.device).long()

            perturbed_graph = self.perturb_graph(node_idx)
        return perturbed_graph
