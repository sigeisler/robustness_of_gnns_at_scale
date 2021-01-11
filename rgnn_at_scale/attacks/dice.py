"""TODO: Do better than this
"""
import random
from rgnn_at_scale import utils
from typing import Union
import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch
from tqdm import tqdm
from torch_geometric.utils import add_self_loops
from rgnn_at_scale import utils
from itertools import chain

class DICE(object):
    """DICE Attack

    Parameters
    ----------
    adj : torch.Tensor
        [n, n] adjacency matrix.
    X : torch.Tensor
        [n, d] feature matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    add_ratio : float
        ratio of the attack budget that is used to add new edges

    """

    def __init__(self,
                 adj: torch.sparse.FloatTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 device: Union[str, int, torch.device],
                 add_ratio: float = 0.6,
                 **kwargs):
        # n is the number of nodes
        self.n = adj.size()[0]
        adj_symmetric_index, _ = utils.to_symmetric(adj.indices(), adj.values(), self.n)
        self.adj_dict = self._to_dict_symmetric(adj_symmetric_index)
        self.adj = adj
        self.labels = labels.cpu()
        self.device = device
        self.attr_adversary = X.cpu()
        self.adj_adversary = None
        # add ratio decides how much of the budget goes to adding edges and the rest goes to deleting
        self.add_ratio = add_ratio

    def _extract_upper_triangle_nodes(self, adj_symmetric_index):
        return (adj_symmetric_index[1] > adj_symmetric_index[0])

    def _to_dict_symmetric(self, adj_symmetric_index):
        """Converts 2D Tensor of indices of sparse matrix into a dictionary.
            Function assumes sparse matrix is symmetrical and returns dictionary of elements in the upper triangle

        Args:
            adj_symmetric_index(torch.LongTensor) : indices of sparse symmetrical matrix
                                  

        Returns:
            dict: Adjacency matrix described as dictionar.
                        keys are tuples (first_node, second_node)
                        values are 1
        """
        mask = self._extract_upper_triangle_nodes(adj_symmetric_index)
        adj_symmetric_index = adj_symmetric_index[:, mask]
        # * Move tensor to cpu to have faster performance
        adj_symmetric_index = adj_symmetric_index.to("cpu")
        myAdj = { (source.item(), dest.item()) : 1 for (source, dest) in adj_symmetric_index.T }
        '''
        for source, dest in zip(adj_symmetric_index[0], adj_symmetric_index[1]):
            myAdj[(source.item(), dest.item())] = 1'''
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
            # check if both nodes have the same label
            if(
                labels[first_node] == labels[second_node]
                and (first_node, second_node) not in to_be_deleted_set
            ):
                delete_budget -= 1
                pbar.update(1)
                # * why do we make a set of nodes to be deleted instead of instantly deleting?
                # * Because we might add a connection in the same place where
                # * we removed a connection from, that's why we perform attack at the end
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
        # ? How to account for both [source, dest] and [dest, source] using list comprehension?
        # indices = [ [source, dest], [dest, source] for source, dest in self.adj_dict.keys()  ]

        indices = []
        for source, dest in adj_dict.keys():
            # We make connection both ways, and update the values list that will be used to construct sparse matrix
            indices.append([source, dest])
            indices.append([dest, source])

        values = [1] * len(indices)
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i.t(), v, torch.Size([self.n, self.n]))

    def attack(self,
               n_perturbations: int,
               **kwargs):
        labels = self.labels
        add_budget = int(n_perturbations * self.add_ratio)
        delete_budget = n_perturbations - add_budget
        adj_dict = self.adj_dict.copy()
        # ? The way this is handled is weird. If there is any self connection, all diagonals get set to zero, then after attack is done, all nodes are set to be having self loops, why?

        # Prepare edges(connections) to be deleted
        to_be_deleted_set = self._collect_edges_to_delete(delete_budget, labels, adj_dict)
        self._add_edges(labels, add_budget, adj_dict)
        # Perform the delete
        self._delete_edges(to_be_deleted_set, adj_dict)
        # change dictionary back to sparse matrix
        self.adj_adversary = self._from_dict_to_sparse(adj_dict)
        self.adj = self.adj_adversary
