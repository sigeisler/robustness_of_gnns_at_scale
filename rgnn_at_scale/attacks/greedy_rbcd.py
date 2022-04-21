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

    def _greedy_update(self, step_size: int, gradient: torch.Tensor):
        _, topk_edge_index = torch.topk(gradient, step_size)

        add_edge_index = self.modified_edge_index[:, topk_edge_index]
        add_edge_weight = torch.ones_like(add_edge_index[0], dtype=torch.float32)

        if self.make_undirected:
            add_edge_index, add_edge_weight = utils.to_symmetric(add_edge_index, add_edge_weight, self.n)
        add_edge_index = torch.cat((self.edge_index, add_edge_index.to(self.data_device)), dim=-1)
        add_edge_weight = torch.cat((self.edge_weight, add_edge_weight.to(self.data_device)))
        edge_index, edge_weight = torch_sparse.coalesce(
            add_edge_index, add_edge_weight, m=self.n, n=self.n, op='sum'
        )

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        # self.edge_weight = torch.ones_like(self.edge_weight)
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

        # To assert the number of perturbations later on
        clean_edges = self.edge_index.shape[1]

        # Determine the number of edges to be flipped in each attach step / epoch
        step_size = n_perturbations // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(n_perturbations % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * n_perturbations

        for step_size in tqdm(steps):
            # Sample initial search space (Algorithm 2, line 3-4)
            self.sample_random_block(step_size)
            # Retreive sparse perturbed adjacency matrix `A \oplus p_{t-1}` (Algorithm 2, line 7)
            edge_index, edge_weight = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Calculate logits for each node (Algorithm 2, line 7)
            logits = self._get_logits(self.attr, edge_index, edge_weight)
            # Calculate loss combining all each node (Algorithm 2, line 8)
            loss = self.calculate_loss(logits[self.idx_attack], self.labels[self.idx_attack])
            # Retreive gradient towards the current block (Algorithm 2, line 8)
            gradient = utils.grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.no_grad():
                # Greedy update of edges (Algorithm 2, line 8)
                self._greedy_update(step_size, gradient)

            del logits
            del loss
            del gradient

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'

        self.adj_adversary = SparseTensor.from_edge_index(
            self.edge_index, self.edge_weight, (self.n, self.n)
        ).coalesce().detach()

        self.attr_adversary = self.attr
