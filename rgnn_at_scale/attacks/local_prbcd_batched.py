from typing import Dict, Optional, Any

import logging

import numpy as np
import torch
import torch_sparse
from torch.nn import functional as F
from torch_sparse import SparseTensor

from rgnn_at_scale.helper.utils import calc_ppr_update_sparse_result

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.attacks.local_prbcd import LocalPRBCD
from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.data import CachedPPRMatrix


class LocalBatchedPRBCD(LocalPRBCD):

    def __init__(self,
                 ppr_matrix: Optional[SparseTensor] = None,
                 ppr_recalc_at_end: bool = False,
                 ppr_cache_params: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(**kwargs)

        assert type(self.surrogate_model) in BATCHED_PPR_MODELS.__args__, "LocalBatchedPRBCD Attack only supports PPRGo models"

        # if self.attack_labeled_nodes_only:
        #     ppr_nodes = self.idx_attack
        # else:
        #     ppr_nodes = np.arange(self.n)

        self.ppr_cache_params = ppr_cache_params
        if self.ppr_cache_params is None:
            self.ppr_cache_params = self.surrogate_model.ppr_cache_params

        # if self.attack_labeled_nodes_only:
        #     relabeled_row = torch.from_numpy(ppr_nodes)[self.ppr_matrix.storage.row()]
        #     self.ppr_matrix = SparseTensor(row=relabeled_row, col=self.ppr_matrix.storage.col(),
        #                                    value=self.ppr_matrix.storage.value(), sparse_sizes=(self.n, self.n))

        self.ppr_recalc_at_end = ppr_recalc_at_end
        self.ppr_matrix = CachedPPRMatrix(self.adj,
                                          self.ppr_cache_params,
                                          self.surrogate_model.alpha,
                                          self.surrogate_model.eps,
                                          self.surrogate_model.topk,
                                          self.surrogate_model.ppr_normalization)

        logging.info(f'self.ppr_matrix is of shape {self.ppr_matrix.shape}')
        logging.info(f'Memory after loading ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        if perturbed_graph is None:
            perturbed_graph = SparseTensor.from_scipy(self.ppr_matrix[node_idx])

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return F.log_softmax(model.forward(self.X, None, ppr_scores=perturbed_graph), dim=-1)
        else:
            return model(data=self.X.to(self.device), adj=perturbed_graph)[node_idx:node_idx + 1]

    def sample_final_edges(self, node_idx: int, n_perturbations: int):
        if self.ppr_recalc_at_end:
            adj = self.get_updated_vector_or_graph(node_idx, only_update_adj=True)
            # Handle disconnected nodes
            disconnected_nodes = (adj.sum(0) == 0).nonzero().flatten()
            if disconnected_nodes.nelement():
                adj = SparseTensor(row=torch.cat((adj.storage.row(), disconnected_nodes)),
                                   col=torch.cat((adj.storage.col(), disconnected_nodes)),
                                   value=torch.cat((adj.storage.col(), torch.full_like(disconnected_nodes, 1e-9))))
            sp_adj = self.adj.to_scipy(layout="csr")
            perturbed_graph = ppr.topk_ppr_matrix(sp_adj,
                                                  self.surrogate_model.alpha + n_perturbations,
                                                  self.surrogate_model.eps,
                                                  np.array([node_idx]),
                                                  self.surrogate_model.topk,
                                                  normalization=self.surrogate_model.ppr_normalization)
        else:
            perturbed_graph = self.perturbe_graph(node_idx)

        return perturbed_graph

    def perturbe_graph(self, node_idx: int, only_update_adj: bool = False) -> SparseTensor:
        if self.attack_labeled_nodes_only:
            current_search_space = torch.tensor(self.idx_attack, device=self.device)[self.current_search_space]
        else:
            current_search_space = self.current_search_space

        modified_edge_weight_diff = SparseTensor(row=torch.zeros_like(self.current_search_space),
                                                 col=current_search_space,
                                                 value=self.modified_edge_weight_diff,
                                                 sparse_sizes=(1, self.n))
        if not only_update_adj:
            A_row = self.adj[node_idx].to(self.device)
            if not isinstance(A_row, SparseTensor):
                A_row = SparseTensor.from_scipy(A_row)
            perturbed_graph = calc_ppr_update_sparse_result(self.ppr_matrix, A_row,
                                                            modified_edge_weight_diff, node_idx, self.surrogate_model.alpha)
            return perturbed_graph
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
