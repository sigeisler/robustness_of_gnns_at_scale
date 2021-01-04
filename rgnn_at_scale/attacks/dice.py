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
        
        #? Transform matrix to symmetrical using utils.to_symmetrical, am I calling the function correctly, does it take the size as self.n??
        adj_symmetric_index, adj_symmetric_values = utils.to_symmetric(adj.indices(), adj.values(), self.n, self.n)
        adj_symmetric = torch.sparse.FloatTensor(adj_symmetric_index, adj_symmetric_values, torch.Size([self.n, self.n]))
        #!
        self.adj_dict = self._to_dict_symmetric(adj_symmetric)
        self.is_symmetric = True
        self.adj = adj_symmetric

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
    def _deletingEdges(self, delete_budget, labels):
        pbar = tqdm(total=delete_budget, desc='removing edges...')
        to_be_deleted_set = set()
        adj_dict =self.adj_dict
        while delete_budget > 0:
            first_node = random.choice( list(self.adj_dict.keys())  )
            second_node = random.choice(self.adj_dict[first_node])
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
                    #* If adjacency matrix is not symmetric we add the other node
                    to_be_deleted_set.add( (first_node, second_node) )  
                    if not self.is_symmetric:
                        to_be_deleted_set.add((second_node, first_node))
        pbar.close()
        return to_be_deleted_set
    def _addingEdges(self, labels, add_budget):
        # add edges till we fill the budget
        pbar = tqdm(total=add_budget, desc='adding edges...')
        while add_budget > 0:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            connected_nodes = self.adj_dict.get(source, [])
            # We only connect two nodes if they do not have the same classification and are not already connected
            if (
                source != dest
                and labels[source] != labels[dest]
                and dest not in connected_nodes
            ):
                connected_nodes.append(dest)
                self.adj_dict[source] = connected_nodes
                add_budget -= 1
                pbar.update(1)
        pbar.close()    
    def _performAttack(self, to_be_deleted_set):
        for pair in to_be_deleted_set:
            first_node = pair[0]
            second_node = pair[1]
            self.adj_dict[first_node].remove(second_node)
            #If node become orphan remove it
            if not self.adj_dict[first_node]:
                self.adj_dict.pop(first_node, None)
    def _from_dict_to_sparse(self):
        indices = []
        values = []
        for key, connected_nodes in self.adj_dict.items():
            #We make connection both ways, and update the values list that will be used to construct sparse matrix
            for i, _ in enumerate(connected_nodes, 0):
                indices.append([key, connected_nodes[i]])
                indices.append([connected_nodes[i], key])
                values.append(1)
                values.append(1)
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i.t(), v, torch.Size([self.n, self.n]))
    
    def attack(self,
               n_perturbations: int,
               attack_seed: int = 0,
               **kwargs):
        np.random.seed(attack_seed)
        labels = self.labels
        add_budget = int(n_perturbations * self.add_ratio)
        delete_budget = n_perturbations - add_budget
        # ? The way this is handled is weird. If there is any self connection, all diagonals get set to zero, then after attack is done, all nodes are set to be having self loops, why?

        # Prepare edges(connections) to be deleted
        to_be_deleted_set = self._deletingEdges( delete_budget, labels)
        self._addingEdges(labels, add_budget)
        # Perform the delete
        self._performAttack(to_be_deleted_set)
        #change dictionary back to sparse matrix
        self.adj_adversary = self._from_dict_to_sparse()
        self.adj = self.adj_adversary