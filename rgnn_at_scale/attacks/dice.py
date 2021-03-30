"""TODO: Do better than this
"""
import random
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric.utils import add_self_loops


class DICE(object):

    def __init__(self,
                 adj: Union[SparseTensor, torch.Tensor],
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 device: Union[str, int, torch.device],
                 add_ratio: float = 0.6,  # ratio of the attack budget that is used for edge addition
                 **kwargs):
        if isinstance(adj, SparseTensor):
            self.n = adj.size(0)
            self.adj = adj.to_scipy(layout="csr")
        else:
            self.n = adj.size()[0]
            coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(adj.indices(), num_nodes=self.n)
            self.adj = sp.csr_matrix(coo_adj)

        self.labels = labels.cpu()
        self.device = device

        self.attr_adversary = X.cpu()
        self.adj_adversary = None
        self.add_ratio = add_ratio

        self.n_perturbations = 0

    def attack(self, n_perturbations: int):
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

        adj = self.adj
        labels = self.labels
        add_budget = int(n_perturbations * self.add_ratio)
        delete_budget = n_perturbations - add_budget

        has_self_loops = adj.diagonal().sum() > 0
        if has_self_loops:
            adj.setdiag(0)  # set diagonal to 0 so that we cannot remove self loops in the perturbations
            adj.eliminate_zeros()

        nonzeros_0, nonzeros_1 = adj.nonzero()

        pbar = tqdm(total=delete_budget, desc='removing edges...')

        # Potential alternative to checking the degree via the adjacency matrix:
        # node_degree = self.n * [0]
        # for nonzero in nonzeros_0:
        #     node_degree[nonzero] += 1
        to_be_deleted_set = set()

        while delete_budget > 0:
            edge_index = np.random.randint(nonzeros_0.shape[0])
            first_node = nonzeros_0[edge_index]
            second_node = nonzeros_1[edge_index]
            # check if they have the same label and they don't get disconnected from the graph after removal
            if(
                labels[first_node] == labels[second_node]
                and edge_index not in to_be_deleted_set
                and adj[first_node].count_nonzero() > 1
                and adj[second_node].count_nonzero() > 1
                # and node_degree[first_node] > 1
                # and node_degree[second_node] > 1
            ):
                delete_budget -= 1
                pbar.update(1)
                to_be_deleted_set.add(edge_index)
                # node_degree[first_node] -= 1
                # node_degree[second_node] -= 1
            #print(f'removed symetric edge: {first_node} to {second_node}')
        pbar.close()

        # add edges till we fill the budget
        pbar = tqdm(total=add_budget, desc='adding edges...')

        to_be_added_set = set()
        while add_budget > 0:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            if (
                source != dest
                and labels[source] != labels[dest]
                and not (source, dest) in to_be_added_set
                and not adj[source, dest]
            ):
                add_budget -= 1
                pbar.update(1)
                to_be_added_set.add((source, dest))
            #print(f'added symetric edge: {first_node} to {second_node}')
        pbar.close()

        to_be_deleted = torch.tensor(list(to_be_deleted_set))
        to_be_kept = torch.ones(nonzeros_0.shape[0], dtype=bool)
        to_be_kept[to_be_deleted] = False
        to_be_added = torch.tensor(list(to_be_added_set)).T

        edge_indices, edge_attributes = torch_geometric.utils.from_scipy_sparse_matrix(adj)
        edge_indices, edge_attributes = edge_indices[:, to_be_kept], edge_attributes[to_be_kept]
        edge_indices = torch.cat((edge_indices, to_be_added, torch.flip(to_be_added, dims=[1, 0])), dim=-1)
        edge_attributes = torch.ones_like(edge_indices[0], dtype=torch.float)

        if has_self_loops:
            adj.setdiag(1)  # set diagonal to 0 so that we cannot remove self loops in the perturbations
            edge_indices, edge_attributes = add_self_loops(edge_indices, edge_attributes, num_nodes=self.n)
        self.adj_adversary = SparseTensor.from_edge_index(
            edge_indices, edge_attributes, (self.n, self.n)
        ).coalesce().detach().to(self.device)

        self.adj = self.adj_adversary.to_scipy(layout="csr")
