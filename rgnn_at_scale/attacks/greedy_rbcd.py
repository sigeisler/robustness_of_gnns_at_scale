import warnings

from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
import torch
from torch import nn
import torch_sparse

from rgnn_at_scale import utils
from rgnn_at_scale.attacks.prbcd import PRBCD
from rgnn_at_scale.models import GCN


class GreedyRBCD(PRBCD):
    """Sampled and hence scalable PGD attack for graph data.
    """

    def __init__(self,
                 adj: torch.sparse.FloatTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: GCN,
                 epochs: int = 500,
                 eps: float = 1e-7,
                 **kwargs):

        super().__init__(model=model, X=X, adj=adj,
                         labels=labels, idx_attack=idx_attack, eps=eps, **kwargs)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        self.X = self.X.to(self.device)

        self.epochs = epochs

    def attack(self, n_perturbations: int):
        step_size = n_perturbations // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(n_perturbations % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * n_perturbations

        original_search_spacer_size = self.search_space_size
        for step_size in tqdm(steps):
            self.sample_search_space(step_size)
            edge_index, edge_weight = self.get_modified_adj()

            logits = F.log_softmax(self.model(data=self.X, adj=(edge_index, edge_weight)), dim=1)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            gradient = utils.grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            does_original_edge_exist = self.match_search_space_on_edges(edge_index, edge_weight)[0]
            edge_weight_factor = 2 * (0.5 - does_original_edge_exist.float())
            gradient *= edge_weight_factor

            # FIXME: Consider only edges that have not been previously modified
            _, topk_edge_index = torch.topk(gradient, step_size)

            add_edge_index = self.modified_edge_index[:, topk_edge_index]
            add_edge_weight = edge_weight_factor[topk_edge_index]
            n_newly_added = int(add_edge_weight.sum().item())
            self.search_space_size -= n_newly_added

            add_edge_index, add_edge_weight = utils.to_symmetric(add_edge_index, add_edge_weight, self.n)
            add_edge_index = torch.cat((self.edge_index, add_edge_index), dim=-1)
            add_edge_weight = torch.cat((self.edge_weight, add_edge_weight))
            edge_index, edge_weight = torch_sparse.coalesce(
                add_edge_index, add_edge_weight, m=self.n, n=self.n, op='sum'
            )

            assert torch.isclose(self.edge_weight.sum() + 2 * n_newly_added, edge_weight.sum())
            is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
            self.edge_index = edge_index[:, is_one_mask]
            self.edge_weight = edge_weight[is_one_mask]
            self.edge_weight = torch.ones_like(self.edge_weight)
            assert self.edge_index.size(1) == self.edge_weight.size(0)

            if self.search_space_size == 0:
                warnings.warn("Search space smaller than perturbation budget! Aborting!")
                break

        # Set to original value for next invocation of attack
        self.search_space_size = original_search_spacer_size

        self.adj_adversary = torch.sparse.FloatTensor(
            self.edge_index,
            self.edge_weight,
            (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.X
