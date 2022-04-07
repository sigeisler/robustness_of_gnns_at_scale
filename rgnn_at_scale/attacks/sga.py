
from typing import Optional, Tuple, Union
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import logging

from tqdm import tqdm
import numpy as np
import torch
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter_add
from torch_geometric.utils import (k_hop_subgraph,
                                   remove_self_loops,
                                   add_remaining_self_loops)

from rgnn_at_scale.attacks.base_attack import SparseLocalAttack
from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS, SGC

patch_typeguard()


@typechecked
class SGA(SparseLocalAttack):
    """
    Implementation of Simplified Gradient-based Attack from Lie et al

    @article{li2020adversarial,
        title={Adversarial attack on large scale graph},
        author={Li, Jintang and Xie, Tao and Chen, Liang and Xie, Fenfang and He, Xiangnan and Zheng, Zibin},
        journal={arXiv preprint arXiv:2009.03488},
        year={2020}
    }

    Parameters
    ----------
    n_perturbations: int
        The number of perturbations (structure or feature) to perform.

    direct: bool, default: True
        indicates whether to directly modify edges/features of the node attacked or only those of influencers.

    n_influencers: int, default: 0
        Number of influencing nodes -- will be ignored if direct is True

    """

    def __init__(self, direct: bool = True, n_influencers: int = 3, loss_type: str = 'Margin', **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.attacked_model,
                          SGC), "The Simplfied Gradient-based Attack is only implemented for the surrogate model SGC"
        assert self.make_undirected, "SGA is only implemented for undirected graphs"

        self.K = self.attacked_model.K
        self.n_classes = self.attacked_model.n_classes
        self.direct = direct
        self.n_influencers = n_influencers
        self.full_adj_degree = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # prohibit caching in the surrogate model
        self.attacked_model.deactivate_caching()

    def _construct_subgraph(self,
                            k_hop_neighbors,
                            k_hop_edge_index,
                            potential_nodes,
                            influencer_nodes):
        potential_edge_index = torch.hstack([
            torch.row_stack([torch.tile(infl.to(self.device), (1, len(potential_nodes))), potential_nodes])
            for infl in influencer_nodes
        ])
        potential_edge_index, _ = remove_self_loops(potential_edge_index)
        potential_edge_weight = torch.zeros(potential_edge_index.shape[1],
                                            dtype=torch.float32,
                                            device=self.device,
                                            requires_grad=True)

        k_hop_edge_weight = torch.ones(k_hop_edge_index.shape[1],
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True)

        (self_loops_index, self_loops_weight,
         A_sub_edge_index, A_sub_edge_weight
         ) = self._build_adjacency(potential_nodes, potential_edge_index, potential_edge_weight,
                                   k_hop_neighbors, k_hop_edge_index, k_hop_edge_weight)

        return (potential_edge_index, potential_edge_weight,
                self_loops_index, self_loops_weight,
                k_hop_edge_index, k_hop_edge_weight,
                A_sub_edge_index, A_sub_edge_weight
                )

    def _build_adjacency(self, potential_nodes, potential_edge_index, potential_edge_weight,
                         k_hop_neighbors, k_hop_edge_index, k_hop_edge_weight):

        self_loops_index = torch.unique(torch.cat([k_hop_neighbors, potential_nodes])).expand(2, -1)
        self_loops_weight = torch.ones(self_loops_index.shape[1],
                                       dtype=torch.float32,
                                       device=self.device,)

        A_sub_edge_index = torch.cat([k_hop_edge_index,
                                      k_hop_edge_index[[1, 0]],
                                      potential_edge_index,
                                      potential_edge_index[[1, 0]],
                                      self_loops_index], dim=-1)
        A_sub_edge_weight = torch.cat([k_hop_edge_weight,
                                       k_hop_edge_weight,
                                       potential_edge_weight,
                                       potential_edge_weight,
                                       self_loops_weight], dim=-1)

        A_sub_edge_index, A_sub_edge_weight = self.normalize_subgraph(A_sub_edge_index,
                                                                      A_sub_edge_weight)
        return (self_loops_index, self_loops_weight,
                A_sub_edge_index, A_sub_edge_weight)

    def _compute_gradient(self, node_idx, A_sub_edge_index, A_sub_edge_weight, input_tensors, eps=5.0):
        logits = self.get_surrogate_logits(node_idx, (A_sub_edge_index, A_sub_edge_weight))

        # model calibration
        logits = (logits.view(1, -1) / eps)

        loss = self.loss_fn(logits, self.labels[node_idx].view(-1)) - \
            self.loss_fn(logits, self.best_non_target_class)
        gradient = torch.autograd.grad(loss,
                                       input_tensors,
                                       create_graph=False)
        return logits, loss, gradient

    def _attack(self,
                n_perturbations: int, node_idx: int,
                **kwargs):

        # prohibit normalization of the adjacency matrix (SGA handles this)
        self.attacked_model.normalize = False

        neighbors = self.adj[node_idx].coo()[1].to(self.device)
        self.full_adj_degree = None
        self.perturbed_edges = None

        with torch.no_grad():
            # To save memory
            device = self.device
            self.device = self.data_device
            self.attacked_model = self.attacked_model.to(self.device)
            logits = self.get_surrogate_logits(node_idx)
            classification_statistics = SGA.classification_statistics(
                logits, self.labels[node_idx].to(self.device))
            logging.info(f'Initial Statstics: {classification_statistics}\n')
            self.device = device
            self.attacked_model = self.attacked_model.to(self.device)
            logits = logits.to(self.device)

        # 1. predict the next most probable class of target t (node_idx)
        # logits = self.get_surrogate_logits(node_idx)
        sorted_logits = logits.argsort(-1)
        self.best_non_target_class = sorted_logits[sorted_logits !=
                                                   self.labels[node_idx, None]].reshape(logits.size(0), -1)[:, -1]

        # 2. Extract the k-hop subgraph centered at t
        (k_hop_neighbors, k_hop_edge_index, _, _) = k_hop_subgraph(node_idx, self.K, self.edge_index, num_nodes=self.n)
        k_hop_neighbors, k_hop_edge_index = k_hop_neighbors.to(self.device), k_hop_edge_index.to(self.device)

        # we are only interested in the directed version
        directed_mask = k_hop_edge_index[0] >= k_hop_edge_index[1]
        k_hop_edge_index = k_hop_edge_index[:, directed_mask]

        # 3. Initialize the subgraph G_(sub) = (A_(sub), X) via Eq.(13);
        # 3.1. Determine all possible/potential candiate nodes to be added to the k-hop subgraph
        # potential nodes: V_p = { u | c_u = c'_t, u in V}
        # where c'_t = argmax_{c' != c_t} T_{t_c'} is second most likely class label
        potential_nodes_mask = self.labels == self.best_non_target_class
        potential_nodes = torch.where(potential_nodes_mask)[0]

        # potential nodes must be outside the k-hop subgraph
        # this is because the potential nodes are the nodes that should be considered
        # to be added to the k-hop neighborhood of the target node

        # this is different from the referenz impl.
        # potential_nodes = potential_nodes[~(potential_nodes == k_hop_neighbors[:, None]).any(0)]
        potential_nodes = potential_nodes[~(potential_nodes == neighbors[:, None]).any(0)]
        # potential edges: E_p = A x V_p
        # where A are the influencer (attacker) nodes
        # A = {t} is target node for direct attacks
        # A = {N(t)} is the neighborhood of the target node for influence attacks
        if self.direct:
            influencer_nodes = torch.tensor([node_idx], device=self.device)
        else:
            influencer_nodes = neighbors

        (potential_edge_index, potential_edge_weight,
         self_loops_idx, self_loops_weights,
         k_hop_edge_index, k_hop_edge_weight,
         A_sub_edge_index, A_sub_edge_weight) = self._construct_subgraph(k_hop_neighbors,
                                                                         k_hop_edge_index,
                                                                         potential_nodes,
                                                                         influencer_nodes)

        # 3.2. Reduce the potential nodes to be added to the k-hop subgraph
        # to only include the \delta-largest (most promising) nodes based on the gradient
        # where \delta = n_pertubations for direct attacks
        # and \delta = n_influencers for influence attacks

        potential_edge_gradient = self._compute_gradient(node_idx, A_sub_edge_index, A_sub_edge_weight,
                                                         [potential_edge_weight])[2][0]
        n_influencers = self.n_influencers
        if self.direct:
            n_influencers = n_perturbations + 1

        _, topk_index = torch.topk(potential_edge_gradient,
                                   k=min(potential_edge_gradient.shape[0], n_influencers),
                                   sorted=False)

        # top k reduce potential nodes
        potential_nodes = potential_nodes[topk_index]

        (potential_edge_index, potential_edge_weight,
         self_loops_idx, self_loops_weights,
         k_hop_edge_index, k_hop_edge_weight,
         A_sub_edge_index, A_sub_edge_weight) = self._construct_subgraph(k_hop_neighbors,
                                                                         k_hop_edge_index,
                                                                         potential_nodes,
                                                                         influencer_nodes)

        # 4. iterative gradient-based attack

        offset = k_hop_edge_weight.shape[0]

        for it in tqdm(range(n_perturbations)):
            # 4.1 compute gradient w.r.t. to G_{sub_i}
            logits, loss, (k_hop_edge_gradient,
                           potential_edge_gradient) = self._compute_gradient(node_idx,
                                                                             A_sub_edge_index, A_sub_edge_weight,
                                                                             [k_hop_edge_weight, potential_edge_weight])
            if it >= 4:
                print("debug")

            with torch.no_grad():
                # 4.2 compute structure score S with Eq.(17)
                k_hop_edge_gradient *= (-2 * k_hop_edge_weight + 1)
                potential_edge_gradient *= (-2 * potential_edge_weight + 1)

                # 4.3 Select e = (u, v) with largest structure score S_{u,v}
                best_edge_idx = torch.cat([k_hop_edge_gradient, potential_edge_gradient]).argmax()
                # _, sorted_edge_idx = torch.cat([k_hop_edge_gradient, potential_edge_gradient]).sort()

                # # in case a edge has already been added or removed do not try to add/remove them again
                # # instead use the next best edge to be added/removed
                # i = 1
                # best_edge_idx = sorted_edge_idx[-i]

                # while (self.perturbed_edges is not None
                #        and torch.any(torch.all(torch.Tensor([u, v]) == self.perturbed_edges.T, dim=1))):
                #     i += 1
                #     best_edge_idx = sorted_edge_idx[-i]
                #     u, v = A_sub_edge_index[:, best_edge_idx]

                # 4.4 Update the k-hop & potential edges according to

                # this only works because A_sub_edge_index is ordered accordingly when constructed

                if best_edge_idx < offset:  # the best edge to flip is an existing edge in the k hop subgraph
                    # hence it should be set to 0
                    u, v = k_hop_edge_index[:, best_edge_idx]
                    k_hop_edge_weight[best_edge_idx] = 0
                    self.full_adj_degree[u] -= 1
                    self.full_adj_degree[v] -= 1
                else:
                    # the best edge to flip is one of the potentiel edges
                    # hence it should be set to 1
                    u, v = potential_edge_index[:, best_edge_idx - offset]
                    potential_edge_weight[best_edge_idx - offset] = 1.0
                    self.full_adj_degree[u] += 1
                    self.full_adj_degree[v] += 1

            (self_loops_index, self_loops_weight,
             A_sub_edge_index, A_sub_edge_weight
             ) = self._build_adjacency(potential_nodes, potential_edge_index, potential_edge_weight,
                                       k_hop_neighbors, k_hop_edge_index, k_hop_edge_weight)

            if self.perturbed_edges is None:
                self.perturbed_edges = torch.Tensor([[u], [v]])
            else:
                self.perturbed_edges = torch.cat([self.perturbed_edges,
                                                  torch.Tensor([[u], [v]])], dim=-1)

            classification_statistics = SGA.classification_statistics(
                logits, self.labels[node_idx].to(self.device))
            logging.info(f'Initial: Loss: {loss.item()} Statstics: {classification_statistics}\n')

        # 5. build perturbed adjacency

        deleted_edges_mask = k_hop_edge_weight == 0
        added_edges_mask = potential_edge_weight == 1

        deleted_edges_idx = k_hop_edge_index[:, deleted_edges_mask]
        deleted_edges_idx = torch.cat([deleted_edges_idx, deleted_edges_idx[[1, 0]]], dim=-1)
        deleted_edges_weights = -torch.ones(deleted_edges_idx.shape[1], dtype=torch.float32, device=self.device)

        added_edges_idx = potential_edge_index[:, added_edges_mask]
        added_edges_idx = torch.cat([added_edges_idx, added_edges_idx[[1, 0]]], dim=-1)
        added_edges_weights = torch.ones(added_edges_idx.shape[1], dtype=torch.float32, device=self.device)

        deleted_edges_idx = deleted_edges_idx.to(self.data_device)
        added_edges_idx = added_edges_idx.to(self.data_device)
        deleted_edges_weights = deleted_edges_weights.to(self.data_device)
        added_edges_weights = added_edges_weights.to(self.data_device)

        A_idx = torch.cat([self.edge_index, deleted_edges_idx,  added_edges_idx], dim=-1)
        A_weights = torch.cat([self.edge_weight, deleted_edges_weights, added_edges_weights], dim=-1)

        A_idx, A_weights, = coalesce(A_idx, A_weights, m=self.n, n=self.n, op='sum')

        eliminate_zeros_mask = A_weights != 0
        A_idx = A_idx[:, eliminate_zeros_mask]
        A_weights = A_weights[eliminate_zeros_mask]

        # make sure there were only valid additions & deletions of edges
        assert torch.all((A_weights == 1))

        self.adj_adversary = SparseTensor.from_edge_index(A_idx, A_weights, (self.n, self.n))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # reactivate normalization of the adjacency matrix (SGA handles this)
        self.attacked_model.normalize = True

    def get_logits(self,
                   model: MODEL_TYPE,
                   node_idx: int,
                   perturbed_graph: Optional[Union[SparseTensor,
                                                   Tuple[TensorType[2, "nnz"], TensorType["nnz"]]]] = None):
        if perturbed_graph is None:
            perturbed_graph = self.adj

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return model.forward(self.attr, perturbed_graph, ppr_idx=np.array([node_idx]))
        else:
            if isinstance(perturbed_graph, tuple):
                edge_index, edge_weight = perturbed_graph
            else:
                edge_index = torch.stack((perturbed_graph.storage.row(), perturbed_graph.storage.col()))
                edge_weight = perturbed_graph.storage.value()

            (_, adj, _, mask) = k_hop_subgraph(node_idx, self.K, edge_index, num_nodes=self.n)
            if edge_weight is not None:
                edge_weight = edge_weight[mask].to(self.device)
            adj = SparseTensor.from_edge_index(adj.to(self.device), edge_weight)

            return model(data=self.attr.to(self.device), adj=adj)[node_idx:node_idx + 1]

    def normalize_subgraph(self, sub_edge_idx: TensorType[2, "nnz"],
                           sub_edge_weight: TensorType["nnz"],) -> Tuple[TensorType[2, "nnz_after"],
                                                                         TensorType["nnz_after"]]:
        # for normalizing the subgraph adjacency matrix we need to use the exact same degree
        # matrix as for the full adjacency matrix to make sure that the normalized edge_weights
        # in the subgraph match the normalized edge_weights of the full adjaceny matrix

        orig_edge_idx, orig_edge_weight = add_remaining_self_loops(self.edge_index, self.edge_weight, 1., self.n)
        assert orig_edge_weight is not None

        if self.full_adj_degree is None:
            self.full_adj_degree = scatter_add(orig_edge_weight, orig_edge_idx[1], dim=0, dim_size=self.n)  # Column sum

        row, col = sub_edge_idx
        deg_inv_sqrt = torch.pow(self.full_adj_degree, -0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        sub_edge_weight = deg_inv_sqrt[row].to(self.device) * sub_edge_weight * deg_inv_sqrt[col].to(self.device)

        return sub_edge_idx, sub_edge_weight

    def get_perturbed_edges(self) -> torch.Tensor:
        if not hasattr(self, "perturbed_edges"):
            return torch.tensor([])
        return self.perturbed_edges
