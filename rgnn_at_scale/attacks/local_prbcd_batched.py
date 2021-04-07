from collections import defaultdict
from typing import Dict, Optional, Union, Any

import math
import logging

import numpy as np
import scipy.sparse as sp

import torch
import torch_sparse
from torch.nn import functional as F
from torch_sparse import SparseTensor

from tqdm import tqdm

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.helper.utils import calc_ppr_update_sparse_result, grad_with_checkpoint

from rgnn_at_scale.attacks.local_prbcd import LocalPRBCD
from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr
from rgnn_at_scale.helper.io import Storage


class LocalBatchedPRBCD(LocalPRBCD):

    def __init__(self,
                 ppr_matrix: Optional[SparseTensor] = None,
                 ppr_recalc_at_end: bool = False,
                 ppr_cache_params: Dict[str, Any] = None,
                 **kwargs):

        super().__init__(**kwargs)

        if self.attack_labeled_nodes_only:
            ppr_nodes = self.idx_attack
        else:
            ppr_nodes = np.arange(self.n)

        self.ppr_cache_params = ppr_cache_params
        if self.ppr_cache_params is None:
            self.ppr_cache_params = self.model.ppr_cache_params

        self.ppr_matrix = None

        if self.ppr_cache_params is not None:
            storage = Storage(self.ppr_cache_params["data_artifact_dir"])
            params = dict(dataset=self.ppr_cache_params["dataset"],
                          alpha=self.model.alpha,
                          ppr_idx=list(map(int, ppr_nodes)),
                          eps=self.model.eps,
                          topk=self.model.topk,
                          ppr_normalization=self.model.ppr_normalization,
                          normalize=self.ppr_cache_params["normalize"],
                          make_undirected=self.ppr_cache_params["make_undirected"],
                          make_unweighted=self.ppr_cache_params["make_unweighted"])

            stored_topk_ppr = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                         params, find_first=True)

            self.ppr_matrix, _ = stored_topk_ppr[0] if len(stored_topk_ppr) == 1 else (None, None)

        if self.ppr_matrix is None:

            sp_adj = self.adj.to_scipy(layout="csr")
            self.ppr_matrix = ppr.topk_ppr_matrix(sp_adj, self.model.alpha, self.model.eps, ppr_nodes,
                                                  self.model.topk, normalization=self.model.ppr_normalization)
            # save topk_ppr to disk
            if self.ppr_cache_params is not None:
                storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"], params,
                                           self.ppr_matrix, ignore_duplicate=True)

        if self.attack_labeled_nodes_only:
            relabeled_row = torch.from_numpy(ppr_nodes)[self.ppr_matrix.storage.row()]
            self.ppr_matrix = SparseTensor(row=relabeled_row, col=self.ppr_matrix.storage.col(),
                                           value=self.ppr_matrix.storage.value(), sparse_sizes=(self.n, self.n))

        self.ppr_recalc_at_end = ppr_recalc_at_end

        logging.info(f'self.ppr_matrix is of shape {self.ppr_matrix.shape}')
        logging.info(f'Memory after loading ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    def get_logits(self, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        if perturbed_graph is None:
            perturbed_graph = SparseTensor.from_scipy(self.ppr_matrix[node_idx])
        return F.log_softmax(self.model.forward(self.X, None, ppr_scores=perturbed_graph), dim=-1)

    def get_final_logits(self, node_idx: int, n_perturbations: int):
        if self.ppr_recalc_at_end:
            adj = self.get_updated_vector_or_graph(node_idx, only_update_adj=True)
            # Handle disconnected nodes
            disconnected_nodes = (adj.sum(0) == 0).nonzero().flatten()
            if disconnected_nodes.nelement():
                adj = SparseTensor(row=torch.cat((adj.storage.row(), disconnected_nodes)),
                                   col=torch.cat((adj.storage.col(), disconnected_nodes)),
                                   value=torch.cat((adj.storage.col(), torch.full_like(disconnected_nodes, 1e-9))))

            self.model.topk = self.model.topk + n_perturbations
            logits = F.log_softmax(self.model.forward(self.X, adj, ppr_idx=np.array([node_idx])), dim=-1)
            self.model.topk = self.model.topk - n_perturbations
        else:
            perturbed_graph = self.sample_edges(node_idx, n_perturbations)
            logits = self.get_logits(node_idx, perturbed_graph)

        return logits

    def get_perturbed_graph(self, node_idx: int, only_update_adj: bool = False) -> SparseTensor:
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
                                                            modified_edge_weight_diff, node_idx, self.model.alpha)
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
