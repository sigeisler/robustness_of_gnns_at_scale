"""TODO: Do better than this
"""
import random
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch
from tqdm import tqdm

from torch_geometric.utils import add_self_loops


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
                 adj: torch.Tensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 device: Union[str, int, torch.device],
                 add_ratio: float = 0.6, 
                 **kwargs):
        # n is the number of nodes        
        self.n = adj.size()[0]
        # We are changing a torch Tensor(Dense I assume) to a scipy sparse Matrix(why not torch sparse??)
        coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(adj.indices(), num_nodes=self.n)
        #adjacency matrix ix compressed sparse row matrix(why not torch sparse?)
        self.adj = sp.csr_matrix(coo_adj)
        # why cpu?? apparently some operations can not be performed on GPU(what are they and are we using them??)
        # this is equal to labels.to('cpu') -> Change it to this format to have unity in our code
        self.labels = labels.cpu()
        self.device = device
        # why always on cpu?
        self.attr_adversary = X.cpu()
        self.adj_adversary = None
        # add ratio decides how much of the budget goes to adding edges and the rest goes to deleting
        self.add_ratio = add_ratio

    #Private Helper Functions
    def __deletingEdges(self, nonzeros_0, nonzeros_1, delete_budget):

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
            # if adj[node].count_nonzero() == 1 means that this connection is the last connection that this node has with the graph and by its removal this node will be completely disconnected with the graph!
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
                # why do we make a set of nodes to be deleted instead of instantly deleting?
                to_be_deleted_set.add(edge_index)
                # node_degree[first_node] -= 1
                # node_degree[second_node] -= 1
            #print(f'removed symetric edge: {first_node} to {second_node}')
        pbar.close()
        return to_be_deleted_set
    
    def __addingEdges(self, labels, adj, add_budget):
        # add edges till we fill the budget
        pbar = tqdm(total=add_budget, desc='adding edges...')

        to_be_added_set = set()
        while add_budget > 0:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            # We only connect two nodes if they do not have the same classification and they are not already connected aka (adj[source, dest] != 1)
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
        return to_be_added_set
    
    def __performAttack(self, to_be_deleted_set, nonzeros_0, to_be_added_set, adj, has_self_loops):

        #Here we ended up with a list for connections to be deleted and connections to be added.
        #to_be_kept is a boolean array, telling which non_zero connections will remain connected and which will be deleted

        to_be_deleted = torch.tensor(list(to_be_deleted_set))
        to_be_kept = torch.ones(nonzeros_0.shape[0], dtype=bool)
        to_be_kept[to_be_deleted] = False
        to_be_added = torch.tensor(list(to_be_added_set)).T

        edge_indices, edge_attributes = torch_geometric.utils.from_scipy_sparse_matrix(adj)
        edge_indices, edge_attributes = edge_indices[:, to_be_kept], edge_attributes[to_be_kept]
        edge_indices = torch.cat((edge_indices, to_be_added, torch.flip(to_be_added, dims=[1, 0])), dim=-1)
        edge_attributes = torch.ones_like(edge_indices[0], dtype=torch.float)
        if has_self_loops:
            adj.setdiag(1)  # set diagonal to 0(1 I guess?) so that we cannot remove self loops in the perturbations
            edge_indices, edge_attributes = add_self_loops(edge_indices, edge_attributes, num_nodes=self.n)
        self.adj_adversary = torch.sparse.FloatTensor(
                                                     edge_indices, edge_attributes, (self.n, self.n)
                                                     ).to(self.device).coalesce()

    def attack(self,
               n_perturbations: int,
               attack_seed: int = 0,
               **kwargs):

        np.random.seed(attack_seed)

        adj = self.adj
        labels = self.labels
        add_budget = int(n_perturbations * self.add_ratio)
        delete_budget = n_perturbations - add_budget

        # If there are connections on diagonal this means node connected to itself ie: self loop
        has_self_loops = adj.diagonal().sum() > 0
        if has_self_loops:
            adj.setdiag(0)  # set diagonal to 0 so that we cannot remove self loops in the perturbations
            adj.eliminate_zeros()

        '''
        nonzeros_0 = [1, 3, 4, 5, 8]
        nonzeros_1 = [2, 8, 5, 7, 2]
        This means that connection between 1&2 is non-zero, connection between 3&8 is non-zero and so on...'''
        nonzeros_0, nonzeros_1 = adj.nonzero()

        # Prepare edges(connections) to be deleted and others to be added
        to_be_deleted_set = self.__deletingEdges(nonzeros_0, nonzeros_1, delete_budget)
        to_be_added_set = self.__addingEdges(labels, adj, add_budget)

        # Perform the attack and update the self.adj_adversary Matrix
        self.__performAttack(to_be_deleted_set, nonzeros_0, to_be_added_set, adj, has_self_loops)

        #The adjusted adjacency matrix(adversary) becomes the adjacency matrix of this class, this is not like the FGSM attack, where the original adjacency matrix is preserved so next attacks
        #with different budgets will always attack the original!
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(self.adj_adversary.indices(), num_nodes=self.n)
        self.adj = sp.csr_matrix(coo_adj)