from typing import Dict, Any
from typeguard import typechecked

import logging

import numpy as np
import torch
from torch_sparse import SparseTensor

from rgnn_at_scale.helper.utils import calc_ppr_update_sparse_result

from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS
from rgnn_at_scale.attacks.local_prbcd import LocalPRBCD
from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr
from rgnn_at_scale.data import CachedPPRMatrix


class LocalBatchedPRBCD(LocalPRBCD):

    @typechecked
    def __init__(self, model: BATCHED_PPR_MODELS, ppr_recalc_at_end: bool = True,
                 ppr_cache_params: Dict[str, Any] = None, **kwargs):
        super().__init__(model=model, **kwargs)

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

        # For poisoning attack we also need the full adjacency matrix
        self._adj_adversary_for_poisoning = None

    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        if perturbed_graph is None:
            perturbed_graph = SparseTensor.from_scipy(self.ppr_matrix[node_idx])

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return model.forward(self.attr, ppr_scores=perturbed_graph)
        else:
            return model(data=self.attr.to(self.device), adj=perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def sample_final_edges(self, node_idx: int, n_perturbations: int):
        perturbed_graph = super().sample_final_edges(node_idx, n_perturbations)

        if self.ppr_recalc_at_end:
            adj = self.perturb_graph(node_idx, only_update_adj=True, n_perturbations=n_perturbations)

            if self.make_undirected:
                # Handle disconnected nodes
                disconnected_nodes = (adj.sum(0) == 0).nonzero().flatten()
                if disconnected_nodes.nelement():
                    logging.info(
                        f'Adding {disconnected_nodes.nelement()} nodes back into perturbed '
                        'adjacency that would have disconnected the graph.')
                    adj = SparseTensor(row=torch.cat((adj.storage.row(), disconnected_nodes)),
                                       col=torch.cat((adj.storage.col(), disconnected_nodes)),
                                       value=torch.cat((adj.storage.col(), torch.full_like(disconnected_nodes, 1e-9))))

            self._adj_adversary_for_poisoning = adj.cpu()

            sp_adj = adj.to_scipy(layout="csr")
            perturbed_graph = ppr.topk_ppr_matrix(sp_adj,
                                                  self.attacked_model.alpha,
                                                  self.attacked_model.eps,
                                                  np.array([node_idx]),
                                                  self.attacked_model.topk,
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

        assert n_perturbations is not None, "n_perturbations must be given when only updating adjacency"

        device = self.adj.device()
        updated_adj = LocalPRBCD.mod_row(modified_edge_weight_diff.to(device), self.adj, node_idx, self.make_undirected)

        return updated_adj

    def adj_adversary_for_poisoning(self):
        assert self._adj_adversary_for_poisoning is not None, 'For poisoning you must set `ppr_recalc_at_end = True`'
        return self._adj_adversary_for_poisoning
