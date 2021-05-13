from tqdm import tqdm
import torch
import torch_sparse
from torch_sparse import SparseTensor

from rgnn_at_scale.helper import utils
from rgnn_at_scale.attacks.prbcd import PRBCD


class GreedyRBCD(PRBCD):
    """Sampled and hence scalable PGD attack for graph data.
    """

    def __init__(self, epochs: int = 500, **kwargs):
        super().__init__(**kwargs)

        rows, cols, self.edge_weight = self.adj.coo()
        self.edge_index = torch.stack([rows, cols], dim=0)

        self.edge_index = self.edge_index.to(self.data_device)
        self.edge_weight = self.edge_weight.float().to(self.data_device)
        self.attr = self.attr.to(self.data_device)
        self.epochs = epochs

        self.n_perturbations = 0

    def _greedy_update(self, step_size: int, edge_index: torch.Tensor,
                       edge_weight: torch.Tensor, gradient: torch.Tensor):
        does_original_edge_exist = self.match_search_space_on_edges(edge_index, edge_weight)[0]
        edge_weight_factor = 2 * (0.5 - does_original_edge_exist.float())
        gradient *= edge_weight_factor

        # FIXME: Consider only edges that have not been previously modified
        _, topk_edge_index = torch.topk(gradient, step_size)

        add_edge_index = self.modified_edge_index[:, topk_edge_index]
        add_edge_weight = edge_weight_factor[topk_edge_index]
        n_newly_added = int(add_edge_weight.sum().item())

        if self.make_undirected:
            add_edge_index, add_edge_weight = utils.to_symmetric(add_edge_index, add_edge_weight, self.n)
        add_edge_index = torch.cat((self.edge_index, add_edge_index.to(self.data_device)), dim=-1)
        add_edge_weight = torch.cat((self.edge_weight, add_edge_weight.to(self.data_device)))
        edge_index, edge_weight = torch_sparse.coalesce(
            add_edge_index, add_edge_weight, m=self.n, n=self.n, op='sum'
        )

        if self.make_undirected:
            assert torch.isclose(self.edge_weight.sum() + 2 * n_newly_added, edge_weight.sum())
        else:
            assert torch.isclose(self.edge_weight.sum() + n_newly_added, edge_weight.sum())
        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        self.edge_weight = torch.ones_like(self.edge_weight)
        assert self.edge_index.size(1) == self.edge_weight.size(0)

    def _attack(self, n_perturbations: int):
        """Perform attack

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        assert n_perturbations > self.n_perturbations, (
            f'Number of perturbations must be bigger as this attack is greedy (current {n_perturbations}, '
            f'previous {self.n_perturbations})'
        )
        n_perturbations -= self.n_perturbations
        self.n_perturbations += n_perturbations

        step_size = n_perturbations // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(n_perturbations % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * n_perturbations

        for step_size in tqdm(steps):
            self.sample_search_space(step_size)
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logits = self.attacked_model(data=self.attr.to(self.device), adj=(edge_index, edge_weight))
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])

            gradient = utils.grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                self._greedy_update(step_size, edge_index, edge_weight, gradient)

            del logits
            del loss
            del gradient

        edge_index, edge_weight = self.get_modified_adj(is_final=True)
        edge_weight = edge_weight.round()
        edge_mask = edge_weight == 1
        self.adj_adversary = SparseTensor.from_edge_index(
            edge_index[:, edge_mask], edge_weight[edge_mask], (self.n, self.n)
        ).coalesce().detach()
        self.attr_adversary = self.attr
