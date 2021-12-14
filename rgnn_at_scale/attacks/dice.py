import random

import numpy as np
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

from rgnn_at_scale.attacks.base_attack import SparseAttack
from rgnn_at_scale.helper import utils


class DICE(SparseAttack):
    """DICE Attack

    Parameters
    ----------
    add_ratio : float
        ratio of the attack budget that is used to add new edges

    """

    def __init__(self, add_ratio: float = 0.6, **kwargs):
        super().__init__(**kwargs)

        assert self.make_undirected, 'Attack only implemented for undirected graphs'

        self.edge_weight = self.edge_weight.float()

        # Create Symmetric Adjacency Matrix
        adj_symmetric_index, adj_symmetric_weights = utils.to_symmetric(self.edge_index, self.edge_weight, self.n)
        self.adj_dict = self._to_dict(adj_symmetric_index, adj_symmetric_weights)
        self.add_ratio = add_ratio

    def _is_in_upper_triangle(self, adj_symmetric_index):
        return (adj_symmetric_index[1] > adj_symmetric_index[0])

    def _to_dict(self, adj_symmetric_index, adj_symmetric_weights):
        """Converts 2D Tensor of indices of sparse matrix into a dictionary.
            Function assumes sparse matrix is symmetrical and returns dictionary of elements in the upper triangle

        Args:
            adj_symmetric_index(torch.LongTensor) : indices of sparse symmetrical matrix

        Returns:
            dict: Adjacency matrix described as dictionar.
                        keys are tuples (first_node, second_node)
                        values are 1
        """
        mask = self._is_in_upper_triangle(adj_symmetric_index)
        adj_symmetric_index = adj_symmetric_index[:, mask]
        adj_symmetric_weights = adj_symmetric_weights[mask]
        # * Move tensors to cpu to have faster performance
        adj_symmetric_index = adj_symmetric_index.to("cpu")
        adj_symmetric_weights = adj_symmetric_weights.to("cpu")
        myAdj = {(source.item(), dest.item()): weight.item()
                 for (source, dest), weight in zip(adj_symmetric_index.T, adj_symmetric_weights)}
        return myAdj

    def _collect_edges_to_delete(self, delete_budget, labels, adj_dict):
        """Chooses the Nodes to be deleted

        Args:
            delete_budget (int): number of nodes to be deleted
            labels ([torch.Tensor]): labels of the nodes

        Returns:
            set: set of tuples(first_node, second_node) of all nodes to be deleted
        """
        pbar = tqdm(total=delete_budget, desc='removing edges...')
        to_be_deleted_set = set()
        dict_keys_list = list(adj_dict.keys())
        while delete_budget > 0:
            first_node, second_node = random.choice(dict_keys_list)
            if(
                labels[first_node] == labels[second_node]
                and (first_node, second_node) not in to_be_deleted_set
            ):
                delete_budget -= 1
                pbar.update(1)
                # we make a set of nodes to be deleted instead of instantly deleting them, because otherwise we might
                # add a connection in the same place where we removed a connection from, that's why we remove the
                # nodes at the end
                to_be_deleted_set.add((first_node, second_node))
        pbar.close()
        return to_be_deleted_set

    def _add_edges(self, labels, add_budget, adj_dict):
        """Adds new edges

        Args:
            labels ([torch.Tensor]): labels of nodes
            add_budget (int): number of nodes to be added
        """
        # add edges till we fill the budget
        pbar = tqdm(total=add_budget, desc='adding edges...')
        while add_budget > 0:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            # We only connect two nodes if they do not have the same classification and are not already connected
            # We do not need to check for (dest, source) since we are working in the upper triangle of matrix
            if (
                source != dest
                and labels[source] != labels[dest]
                and not adj_dict.get((source, dest))
            ):
                adj_dict[(source, dest)] = 1
                add_budget -= 1
                pbar.update(1)
        pbar.close()

    def _delete_edges(self, to_be_deleted_set, adj_dict):
        """Performs the deletion of the nodes

        Args:
            to_be_deleted_set (set): set of Nodes to be deleted in shape of tuples (first_node, second_node)
            adj_dict (dictionary) : the dictionary describing the upper triangle nodes of the adjacency matrix
        """
        for pair in to_be_deleted_set:
            first_node = pair[0]
            second_node = pair[1]
            adj_dict.pop((first_node, second_node), None)

    def _from_dict_to_sparse(self, adj_dict):
        """Converts dictionary(of symmetrical adjacency matrix) back to sparse Tensor(pyTorch)

        Returns:
            [torch.sparse.FloarTensor]: sparse adjacency matrix
        """
        indices = list(adj_dict.keys())
        values = [1] * len(indices)

        edge_index = torch.LongTensor(indices).T.to(self.device)
        edge_attr = torch.FloatTensor(values).to(self.device)

        edge_index, edge_attr = utils.to_symmetric(edge_index, edge_attr, self.n)

        return SparseTensor.from_edge_index(edge_index=edge_index,
                                            edge_attr=edge_attr,
                                            sparse_sizes=torch.Size([self.n, self.n]))

    def _attack(self,
                n_perturbations: int,
                **kwargs):
        add_budget = int(n_perturbations * self.add_ratio)
        delete_budget = n_perturbations - add_budget
        adj_dict = self.adj_dict.copy()
        to_be_deleted_set = self._collect_edges_to_delete(delete_budget, self.labels, adj_dict)
        self._add_edges(self.labels, add_budget, adj_dict)
        self._delete_edges(to_be_deleted_set, adj_dict)
        self.adj_adversary = self._from_dict_to_sparse(adj_dict)
