"""TODO: Do better than this
"""
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch
from tqdm import tqdm


class DICE(object):

    def __init__(self,
                 adj: torch.Tensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 device: Union[str, int, torch.device],
                 add_ratio: float = 0.6,  # ratio of the attack budget that is used for edge addition
                 **kwargs):
        self.n = adj.size()[0]
        coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(adj.indices(), num_nodes=self.n)
        self.adj = sp.csr_matrix(coo_adj)

        self.labels = labels.cpu()
        self.device = device

        self.attr_adversary = X.cpu()
        self.adj_adversary = None
        self.add_ratio = add_ratio

    def attack(self,
               n_perturbations: int,
               attack_seed: int = 0,
               **kwargs):

        np.random.seed(attack_seed)

        adj = self.adj
        labels = self.labels
        add_budget = int(n_perturbations * self.add_ratio)
        delete_budget = n_perturbations - add_budget

        has_self_loops = adj.diagonal().sum() > 0
        if has_self_loops:
            adj.setdiag(0)  # set diagonal to 0 so that we cannot remove self loops in the perturbations
        nonzeros_0, nonzeros_1 = adj.nonzero()

        pbar = tqdm(total=delete_budget, desc='removing edges...')

        while delete_budget > 0:
            edge_index = np.random.randint(nonzeros_0.shape[0])
            first_node = nonzeros_0[edge_index]
            second_node = nonzeros_1[edge_index]
            # check if they have the same label and they don't get disconnected from the graph after removal
            if(
                labels[first_node] == labels[second_node]
                and adj[first_node].count_nonzero() > 1
                and adj[second_node].count_nonzero() > 1
            ):
                delete_budget -= 1
                pbar.update(1)
                adj[first_node, second_node] = 0
                adj[second_node, first_node] = 0
                nonzeros_0, nonzeros_1 = adj.nonzero()
            #print(f'removed symetric edge: {first_node} to {second_node}')
        pbar.close()
        adj.eliminate_zeros()

        # add edges till we fill the budget
        pbar = tqdm(total=add_budget, desc='adding edges...')

        to_be_added = []
        while add_budget > 0:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            if (
                source != dest
                and labels[source] != labels[dest]
                and not (source, dest) in to_be_added
                and not adj[source, dest]
            ):
                add_budget -= 1
                pbar.update(1)
                to_be_added.append((source, dest))
            #print(f'added symetric edge: {first_node} to {second_node}')
        pbar.close()
        to_be_added = torch.tensor(to_be_added).T

        edge_indices, edge_attributes = torch_geometric.utils.from_scipy_sparse_matrix(adj)
        edge_indices = torch.cat((edge_indices, to_be_added, torch.flip(to_be_added, dims=[1, 0])), dim=-1)
        edge_attributes = torch.ones_like(edge_indices[0], dtype=torch.float)

        if has_self_loops:
            adj.setdiag(1)  # set diagonal to 0 so that we cannot remove self loops in the perturbations
        self.adj_adversary = torch.sparse.FloatTensor(
            edge_indices, edge_attributes, (self.n, self.n)
        ).to(self.device).coalesce()

        coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(self.adj_adversary.indices(), num_nodes=self.n)
        self.adj = sp.csr_matrix(coo_adj)
