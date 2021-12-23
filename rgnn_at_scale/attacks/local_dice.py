import numpy as np
import torch
from torch_sparse import SparseTensor

from rgnn_at_scale.attacks.base_attack import SparseLocalAttack
from rgnn_at_scale.models import MODEL_TYPE, BATCHED_PPR_MODELS


class LocalDICE(SparseLocalAttack):
    """A Local version of the DICE Attack

    Parameters
    ----------
    add_ratio : float
        ratio of the attack budget that is used to add new edges

    """

    def __init__(self, add_ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        assert not self.make_undirected, 'Attack only implemented for directed graphs'

        self.add_ratio = add_ratio

    def _attack(self,
                n_perturbations: int, node_idx: int,
                **kwargs):

        # remove edges connecting to nodes of the same class
        # add edges connecting to nodes of different class

        add_budget = int(round((n_perturbations * self.add_ratio), 0))
        delete_budget = int(n_perturbations - add_budget)

        # 1. get all incoming edge indices of the given node

        adj_i = self.adj[node_idx]
        _, neighbors_idx, _ = adj_i.coo()

        # 2. check which of them are connecting to same class nodes

        same_class_mask = self.labels[neighbors_idx] == self.labels[node_idx]

        # 3. sample edges to nodes of not in 2. and add them
        exlude_from_add_idx = [node_idx] + neighbors_idx.tolist()
        add_neighbors_idx = self._sample_additions(
            node_idx, n_perturbations, min(delete_budget, same_class_mask.sum()), exclude=exlude_from_add_idx)

        # 4. sample edges to nodes of 2. and delete them
        delete_neighbors_mask = torch.full_like(neighbors_idx, False, dtype=bool)
        if delete_budget > 0:
            delete_neighbors_idx = neighbors_idx[same_class_mask][torch.randperm(
                same_class_mask.sum())][: delete_budget]
            delete_neighbors_mask = ((neighbors_idx.repeat(delete_neighbors_idx.shape[0]).view(
                delete_neighbors_idx.shape[0], -1) == delete_neighbors_idx[:, None].repeat(1, neighbors_idx.shape[0]))
                .any(dim=0))

        # 5. build perturbed adjacency
        A_rows, A_cols, A_vals = self.adj.coo()
        A_idx = torch.stack([A_rows, A_cols], dim=0)

        is_before = A_rows < node_idx
        is_after = A_rows > node_idx

        i_col = torch.cat([neighbors_idx[~delete_neighbors_mask], add_neighbors_idx], dim=0).sort().values
        i_row = torch.full_like(i_col, node_idx)
        i_idx = torch.stack([i_row, i_col], dim=0)
        i_val = torch.ones(i_idx.shape[1])

        A_idx = torch.cat((A_idx[:, is_before], i_idx, A_idx[:, is_after]), dim=-1)
        A_weights = torch.cat((A_vals[is_before], i_val, A_vals[is_after]), dim=-1)

        self.perturbed_edges = i_idx
        self.adj_adversary = SparseTensor.from_edge_index(A_idx, A_weights, (self.n, self.n))

    def _sample_additions(self, node_idx, n_perturbations, n_deletions, exclude=[]):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """

        additions_idx = []
        while len(additions_idx) < n_perturbations - n_deletions:
            possible_edge = torch.randint(self.n, (1, 1)).item()
            if possible_edge not in exclude and self.labels[possible_edge] != self.labels[node_idx]:
                additions_idx.append(possible_edge)
                exclude.append(possible_edge)
        return torch.tensor(additions_idx, dtype=torch.long)

    def get_logits(self, model: MODEL_TYPE, node_idx: int, perturbed_graph: SparseTensor = None):
        if perturbed_graph is None:
            perturbed_graph = self.adj

        if type(model) in BATCHED_PPR_MODELS.__args__:
            return model.forward(self.attr, perturbed_graph, ppr_idx=np.array([node_idx]))
        else:
            return model(data=self.attr.to(self.device), adj=perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def get_perturbed_edges(self) -> torch.Tensor:
        if not hasattr(self, "perturbed_edges"):
            return torch.tensor([])
        return self.perturbed_edges
