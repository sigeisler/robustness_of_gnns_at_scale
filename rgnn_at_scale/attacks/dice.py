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
    #! New Helper Functions
    def _to_dict(self, adj: torch.sparse.FloatTensor):
        ''' Here we get full adjacency matrix not just upper triangle '''
        myAdj = dict()
        indices = adj.indices()
        print(indices.type())
        for index in range(len(indices[0])):
            key = indices[0, index].item()
            if myAdj.get(key) is None:
                myAdj[key] = []
            myAdj[key].append(indices[1, index].item())
        return myAdj

    def _to_dict_symmetric(self, adj: torch.sparse.FloatTensor):
        '''Here We only need the upper triangle, since adjacency matrix is symmetrical'''
        myAdj = dict()
        indices = adj.indices()
        print(indices.type())
        for index in range(len(indices[0])):
            key = indices[0, index].item()
            if myAdj.get(key) is None:
                myAdj[key] = []
            if key < indices[1, index]:
                myAdj[key].append(indices[1, index].item())
        # Remove all empty keys
        # This will happen if a node has all its connections in the lower triangle
        for key in list(myAdj.keys()):
            if myAdj.get(key) is not None and not myAdj.get(key):
                myAdj.pop(key, None)
        return myAdj

    #!---------------------------------------------------------


    def __init__(self,
                 adj: torch.sparse.FloatTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 device: Union[str, int, torch.device],
                 add_ratio: float = 0.6, 
                 **kwargs):
        # n is the number of nodes        
        self.n = adj.size()[0]

        #TODO: Adding dictionary
        #//self.adj_dict = self._to_dict_symmetric(adj)

        #* We are changing a sparse torch Tensor to a scipy sparse Matrix(why not torch sparse??)
        #* We do not continue with Torch sparse because it is missing many functions
        coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(adj.indices(), num_nodes=self.n)
        #*adjacency matrix is compressed sparse row matrix(why not torch sparse?)
        self.adj = sp.csr_matrix(coo_adj)

        # ? why cpu?? apparently some operations can not be performed on GPU(what are they and are we using them??)
        # ? this is equal to labels.to('cpu') -> Change it to this format to have unity in our code
        self.labels = labels.cpu()
        self.device = device
        # why always on cpu?
        self.attr_adversary = X.cpu()
        self.adj_adversary = None
        
        # add ratio decides how much of the budget goes to adding edges and the rest goes to deleting
        self.add_ratio = add_ratio

    #!Private Helper Functions
    def __deletingEdges(self, nonzeros_0, nonzeros_1, delete_budget, adj, labels):

        pbar = tqdm(total=delete_budget, desc='removing edges...')

        to_be_deleted_set = set()
        node_connections = dict()
        adj_dict =self.adj_dict

        while delete_budget > 0:
            edge_index = np.random.randint(nonzeros_0.shape[0])
            first_node = nonzeros_0[edge_index]
            second_node = nonzeros_1[edge_index]

            #//first_node = random.choice(list(adj_dict.keys()))
            #//second_node = random.choice(adj_dict[first_node])

            # check if both nodes have the same label
            if(
                labels[first_node] == labels[second_node]
                and edge_index not in to_be_deleted_set
            ):

                    delete_budget -= 1
                    pbar.update(1)
                    # * why do we make a set of nodes to be deleted instead of instantly deleting?
                    # * Because we might add a connection in the same place where
                    # * we removed a connection from, that's why we perform attack at the end
                    to_be_deleted_set.add(edge_index)
                    #//print(f'deleted edge from {first_node} to {second_node}')
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
            # We only connect two nodes if they do not have the same classification and 
            # they are not already connected aka (adj[source, dest] != 1)
            if (
                source != dest
                and labels[source] != labels[dest]
                and not (source, dest) in to_be_added_set
                and not adj[source, dest]
            ):
                add_budget -= 1
                pbar.update(1)
                to_be_added_set.add((source, dest))
            #//print(f'added symetric edge: {source} to {dest}')
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
        #?Why--------------
        adj = self.adj
        labels = self.labels
        #?--------------------
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
        to_be_deleted_set = self.__deletingEdges(nonzeros_0, nonzeros_1, delete_budget, adj, labels)
        to_be_added_set = self.__addingEdges(labels, adj, add_budget)

        # Perform the attack and update the self.adj_adversary Matrix
        self.__performAttack(to_be_deleted_set, nonzeros_0, to_be_added_set, adj, has_self_loops)

        #The adjusted adjacency matrix(adversary) becomes the adjacency matrix of this class                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        coo_adj = torch_geometric.utils.to_scipy_sparse_matrix(self.adj_adversary.indices(), num_nodes=self.n)
        self.adj = sp.csr_matrix(coo_adj)