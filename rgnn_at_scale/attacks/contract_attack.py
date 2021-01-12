import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union
from rgnn_at_scale.models import DenseGCN
from copy import deepcopy
from rgnn_at_scale import utils
class EXPAND_CONTRACT():

    def __init__(self,
                 adj: torch.sparse.FloatTensor , 
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 device: Union[str, int, torch.device],
                 k: int = 1, 
                 alpha: float = 2.0, 
                 m: int = 1, 
                 step: int = 1, 
                 protected=None,
                 **kwargs
                 ):
        assert alpha >= 1.0
        assert m >= 1
        assert adj.device == X.device
        #? if k == 0:
            #?return torch.empty((2, 0), dtype=torch.long, device=graph.a.device())
        self.alpha = alpha
        #? Is m the n_perturbations??
        self.m = m
        self.k = k
        #self.rng = np.random.default_rng()
        self.n = adj.size()[0]
        self.adj = adj
        adj_symmetric_index, adj_symmetric_weights = utils.to_symmetric(adj.indices(), adj.values(), self.n)
        self.adj_dict, self.adj_symmetric_index, self.adj_symmetric_weights = self._to_dict_symmetric(adj_symmetric_index, adj_symmetric_weights)
        self.X = X
        self.protected = protected
        self.step = step
        self.device = device
        self.model = deepcopy(model).to(self.device)
        self.idx_attack = idx_attack
        self.labels = labels.to(device)
        self.attr_adversary = None
        self.adj_adversary = None
        
    def _from_dict_to_sparse(self, adj_dict):
        """Converts dictionary(of symmetrical adjacency matrix) back to sparse Tensor(pyTorch)

        Returns:
            [torch.sparse.FloarTensor]: sparse adjacency matrix
        """
        indices = []
        for source, dest in adj_dict.keys():
            # We make connection both ways, and update the values list that will be used to construct sparse matrix
            indices.append([source, dest])
            indices.append([dest, source])

        values = [1] * len(indices)
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i.t(), v, torch.Size([self.n, self.n]))

    def _log_likelihood(self, adj_dict, adj_symmetric_index, adj_symmetric_weights):
        #adj = self._from_dict_to_sparse(adj_dict)
        adj = torch.sparse.FloatTensor(adj_symmetric_index, adj_symmetric_weights, torch.Size([self.n, self.n]))
        logits = self.model.to(self.device)(self.X, adj)
        return -F.cross_entropy(logits[self.idx_attack], self.labels[self.idx_attack])

    def _delete_edge_from_tensor(self, source, dest, adj_symmetric_index, adj_symmetric_weights):
        indices = torch.nonzero( (adj_symmetric_index[0] == source), as_tuple=True  )[0]
        dest_index = indices[ torch.nonzero( (adj_symmetric_index[1][indices] == dest), as_tuple=True  )[0] ]
        adj_symmetric_index = torch.cat( (   adj_symmetric_index[:, 0:dest_index  ],  adj_symmetric_index[:, dest_index+1:]   ), 1 )
        adj_symmetric_weights = torch.cat( (   adj_symmetric_weights[0:dest_index],  adj_symmetric_weights[dest_index+1:]   ) )
        return adj_symmetric_index, adj_symmetric_weights

    def _flip_edges(self, adj_dict, edges, adj_symmetric_index, adj_symmetric_weights):
        if type(edges) is not list:
            edges = [edges]
        for source, dest in edges:
            if( not adj_dict.get((source, dest)) ):
                adj_dict[(source, dest)] = 1
                #?keep lists updated to make log_likelihood function more computationally efficient??
                adj_symmetric_index = torch.cat((adj_symmetric_index, torch.LongTensor([[source], [dest] ])), 1)
                adj_symmetric_weights = torch.cat((adj_symmetric_weights, torch.tensor([1])))

            else:
                adj_dict.pop((source, dest), None)
                adj_symmetric_index, adj_symmetric_weights = self._delete_edge_from_tensor(source, dest, adj_symmetric_index, adj_symmetric_weights)
        return adj_symmetric_index, adj_symmetric_weights

    def _contract(self, adj_dict, cohort, adj_symmetric_index, adj_symmetric_weights):
        # Contract the cohort until we are in budget again
        bar = tqdm(total=len(cohort) - self.k, leave=False, desc="Contract")
        while len(cohort) > self.k:
            llhs =  np.empty( len(cohort) )
            for i, edge in enumerate(cohort):
                adj_symmetric_index, adj_symmetric_weights = self._flip_edges(adj_dict, edge, adj_symmetric_index, adj_symmetric_weights)
                llhs[i] = self._log_likelihood(adj_dict, adj_symmetric_index, adj_symmetric_weights)                
                adj_symmetric_index, adj_symmetric_weights = self._flip_edges(adj_dict, edge, adj_symmetric_index, adj_symmetric_weights)
            # Undo the flips that increase the log-likelihood the least when undone
            n_undo = min(self.step, len(cohort) - self.k)
            for edge in [cohort[i] for i in np.argpartition(llhs, n_undo)[:n_undo]]:
                cohort.remove(edge)
                adj_symmetric_index, adj_symmetric_weights = self._flip_edges(adj_dict, edge, adj_symmetric_index, adj_symmetric_weights)
            bar.update(n_undo)
        bar.close()
        return adj_symmetric_index, adj_symmetric_weights
 
    def _add_edges_to_list(self, my_list, number_of_nodes_to_add):
        count = 0
        while count < number_of_nodes_to_add:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            if(source != dest):
                my_list.append((source, dest))
                count+= 1
        return my_list

    def _extract_upper_triangle_nodes(self, adj_symmetric_index):
        return (adj_symmetric_index[1] > adj_symmetric_index[0])
    
    def _to_dict_symmetric(self, adj_symmetric_index, adj_symmetric_weights):
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
        adj_symmetric_weights = adj_symmetric_weights[mask]
        # * Move tensors to cpu to have faster performance
        adj_symmetric_index = adj_symmetric_index.to("cpu")
        adj_symmetric_weights = adj_symmetric_weights.to("cpu")
        myAdj = { (source.item(), dest.item()) : weight.item() for (source, dest), weight in zip(adj_symmetric_index.T, adj_symmetric_weights) }
        return myAdj, adj_symmetric_index, adj_symmetric_weights
    
    def attack(self, 
               n_perturbations: int, 
               **kwargs):
        k = self.k
        adj_dict = self.adj_dict.copy()
        adj_symmetric_index = self.adj_symmetric_index.detach().clone()
        adj_symmetric_weights = self.adj_symmetric_weights.detach().clone()
        #How many possible edges in the upper triangular
        n_edges = (self.n * (self.n - 1)) //2
        assert n_edges >= k
        clean_ll = self._log_likelihood(adj_dict, adj_symmetric_index, adj_symmetric_weights)
        bar = tqdm(total=n_perturbations, desc="Cycles") 
        # Randomly select a cohort of edges to flip
        cohort_size = min(int(self.alpha * self.k), n_edges)
        cohort = []
        cohort = self._add_edges_to_list(cohort, cohort_size)
        adj_symmetric_index, adj_symmetric_weights = self._flip_edges(adj_dict, cohort, adj_symmetric_index, adj_symmetric_weights)
        n_expand = min(round((self.alpha - 1.0) * k), n_edges - k)
        if n_expand > 0:
            for _ in range(n_perturbations - 1): 
                adj_symmetric_index, adj_symmetric_weights = self._contract(adj_dict,cohort, adj_symmetric_index, adj_symmetric_weights)
                ll = self._log_likelihood(adj_dict, adj_symmetric_index, adj_symmetric_weights)
                bar.set_description(f"Cycles (decr {clean_ll - ll:.5f})")
                bar.update()
                new_edges = []
                new_edges = self._add_edges_to_list(new_edges, n_expand)
                adj_symmetric_index, adj_symmetric_weights = self._flip_edges(adj_dict, new_edges, adj_symmetric_index, adj_symmetric_weights)
                cohort.extend(list(new_edges))

        self._contract(adj_dict,cohort, adj_symmetric_index, adj_symmetric_weights)
        bar.update()
        self.adj_adversary = self._from_dict_to_sparse(adj_dict)
        self.attr_adversary = self.X