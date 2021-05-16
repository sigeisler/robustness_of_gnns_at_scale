from typing import Dict, Optional, Any

import logging

import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor

from rgnn_at_scale.helper.utils import calc_ppr_update_sparse_result

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.attacks.local_prbcd import LocalPRBCD
from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr
from rgnn_at_scale.helper.utils import to_symmetric
from rgnn_at_scale.data import CachedPPRMatrix


class LocalBatchedPRBCD(LocalPRBCD):

    def __init__(self, ppr_recalc_at_end: bool = True, ppr_cache_params: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)

        assert type(self.attacked_model) in BATCHED_PPR_MODELS.__args__, "LocalBatchedPRBCD Attack only supports PPRGo models"

        self.ppr_cache_params = ppr_cache_params
        if self.ppr_cache_params is not None:
            self.ppr_cache_params.update(dict(make_undirected=self.make_undirected))

        self.ppr_recalc_at_end = ppr_recalc_at_end
        self.ppr_matrix = CachedPPRMatrix(self.adj,
                                          self.ppr_cache_params,
                                          self.attacked_model.alpha,
                                          self.attacked_model.eps,
                                          self.attacked_model.topk,
                                          self.attacked_model.ppr_normalization)

        logging.info(f'self.ppr_matrix is of shape {self.ppr_matrix.shape}')
        logging.info(f'Memory after initalizing attack: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        if perturbed_graph is None:
            perturbed_graph = SparseTensor.from_scipy(self.ppr_matrix[node_idx])

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return model.forward(self.attr, None, ppr_scores=perturbed_graph)
        else:
            return model(data=self.attr.to(self.device), adj=perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def sample_final_edges(self, node_idx: int, n_perturbations: int):
        perturbed_graph = super().sample_final_edges(node_idx, n_perturbations)

        if self.ppr_recalc_at_end:
            adj = self.perturb_graph(node_idx, only_update_adj=True, n_perturbations=n_perturbations)
            # Handle disconnected nodes
            disconnected_nodes = (adj.sum(0) == 0).nonzero().flatten()
            if disconnected_nodes.nelement():
                adj = SparseTensor(row=torch.cat((adj.storage.row(), disconnected_nodes)),
                                   col=torch.cat((adj.storage.col(), disconnected_nodes)),
                                   value=torch.cat((adj.storage.col(), torch.full_like(disconnected_nodes, 1e-9))))
            sp_adj = adj.to_scipy(layout="csr")
            perturbed_graph = ppr.topk_ppr_matrix(sp_adj,
                                                  self.attacked_model.alpha,
                                                  self.attacked_model.eps,
                                                  np.array([node_idx]),
                                                  self.attacked_model.topk + n_perturbations,
                                                  normalization=self.attacked_model.ppr_normalization)
            perturbed_graph = SparseTensor.from_scipy(perturbed_graph)

        return perturbed_graph

    def perturb_graph(self, node_idx: int, only_update_adj: bool = False, n_perturbations: int = None) -> SparseTensor:
        modified_edge_weight_diff = SparseTensor(row=torch.zeros_like(self.current_search_space),
                                                 col=self.current_search_space,
                                                 value=self.modified_edge_weight_diff,
                                                 sparse_sizes=(1, self.n))
        if not only_update_adj:
            A_row = self.adj[node_idx].to(self.device)
            if not isinstance(A_row, SparseTensor):
                A_row = SparseTensor.from_scipy(A_row)
            perturbed_graph = calc_ppr_update_sparse_result(self.ppr_matrix, A_row,
                                                            modified_edge_weight_diff,
                                                            node_idx, self.attacked_model.alpha)
            return perturbed_graph
        else:
            assert n_perturbations is not None, "n_perturbations must be given when only updating adjacency"
            v_rows, v_cols, v_vals = modified_edge_weight_diff.to(self.adj.device()).coo()
            v_rows += node_idx

            # projection
            pertubations = v_vals.argsort()[-n_perturbations:]
            v_rows = v_rows[pertubations]
            v_cols = v_cols[pertubations]
            v_vals = v_vals[pertubations]
            v_idx = torch.stack([v_rows, v_cols], dim=0)

            A_rows, A_cols, A_vals = self.adj.coo()
            A_idx = torch.stack([A_rows, A_cols], dim=0)

            # sparse addition: row = A[i] + v

            # we know v_rows only has node_idx as value
            assert all(v_rows == node_idx), "In a local attack only outgoing edges of the attacked edge are valid"
            pert_segment_mask = A_rows == node_idx

            A_idx_pert_seg = A_idx[:, pert_segment_mask]
            A_vals_pert_seg = A_vals[pert_segment_mask]

            A_idx_pert_seg = torch.cat((v_idx, A_idx_pert_seg), dim=-1)
            A_vals_pert_seg = torch.cat((v_vals, A_vals_pert_seg))

            # we need to insert the new edges already sorted, otherwise if the index passed to torch_sparse is not
            # fully sorted it will try to sort the complete index which is infeasible for large graphs
            idx = A_idx_pert_seg[1].new_zeros(A_idx_pert_seg[1].numel() + 1)
            idx[1:] = A_idx_pert_seg[0]
            idx[1:] *= self.n
            idx[1:] += A_idx_pert_seg[1]
            perm = idx[1:].argsort()

            A_idx_pert_seg = A_idx_pert_seg[:, perm]
            A_vals_pert_seg = A_vals_pert_seg[perm]

            pert_segment_mask_before = A_rows < node_idx
            pert_segment_mask_after = A_rows > node_idx

            A_idx = torch.cat((A_idx[:, pert_segment_mask_before], A_idx_pert_seg,
                               A_idx[:, pert_segment_mask_after]), dim=-1)
            A_weights = torch.cat((A_vals[pert_segment_mask_before], A_vals_pert_seg,
                                   A_vals[pert_segment_mask_after]), dim=-1)

            A_idx, A_weights = torch_sparse.coalesce(
                A_idx,
                A_weights,
                m=1,
                n=self.n,
                op='sum'
            )

            # Works since the attack will always assign at least a small constant the elements in p
            A_weights[A_weights > 1] = -A_weights[A_weights > 1] + 2

            if self.make_undirected:
                A_idx, A_weights = to_symmetric(A_idx, A_weights, self.n, op='max')

            updated_adj = SparseTensor.from_edge_index(A_idx, A_weights, (self.n, self.n))

            return updated_adj
