import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union
from rgnn_at_scale.models import DenseGCN
from copy import deepcopy

#! For beauty of tqdm bar
import time
#!---------------
#//from .utils import flip_edges_

class EXPAND_CONTRACT():
    #! Helper functions
    def log_likelihood(self):
        #? graph is the adjacency mattrix and attributes, while graph.y is labels
        #? model(graph) should produce predictions based on the type of Model GDC, etc...

        logits = self.model.to(self.device)(self.X, self.adj)
        return -F.cross_entropy(logits[self.idx_attack], self.labels[self.idx_attack])

    def flip_edges(self, adj, row, col, edge):
        new_edge_value = -self.adj[ row[edge], col[edge] ] + 1
        self.adj[row[edge], col[edge]] = new_edge_value
        self.adj[col[edge], row[edge]] = new_edge_value


    def contract(self, adj, cohort, row, col, s):
        # Contract the cohort until we are in budget again
        bar = tqdm(total=len(cohort) - self.k, leave=False, desc="Contract")
        while len(cohort) > self.k:
            #? What does this .new_empty do?? 
            llhs =  np.empty( len(cohort) ) #//X.new_empty(len(cohort))
            for i, edge in enumerate(cohort):
                #* flip the edge in the adjacency matrix, it is a dense matrix!
                #//flip_edges_(graph.a, row[edge], col[edge])
                self.flip_edges(adj, row, col, edge)
                llhs[i] = self.log_likelihood()                
                #//flip_edges_(graph.a, row[edge], col[edge])
                self.flip_edges(adj, row, col, edge)

            #? What is the type of llhs? I assume numpy array from the call of np.argpartition
            #? but from .cpu() it means it was a pyTorch Tensor and then we turn it into numpy. 
            #? In this sense since we so far do not calculate any gradients we should just make it
            #? numpy in the first place!
            #//llhs = llhs.cpu().numpy()

            # Undo the flips that increase the log-likelihood the least when undone
            n_undo = min(self.step, len(cohort) - self.k)
            for edge in [cohort[i] for i in np.argpartition(llhs, n_undo)[:n_undo]]:
                cohort.remove(edge)
                #//flip_edges_(graph.a, row[edge], col[edge])
                self.flip_edges(adj, row, col, edge)
                s[edge] = True
            bar.update(n_undo)
            #time.sleep(0.5)
        bar.close()

    def __init__(self,   
                 adj: torch.sparse.FloatTensor , 
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 device: Union[str, int, torch.device],
                 #//graph,
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
        self.rng = np.random.default_rng()
        self.n = adj.size()[0]

        #? should it the adjacency be dense like their code?
        self.original_adj = adj.to_dense().to(device)
        self.adj = self.original_adj.clone()

        self.X = X.to(device)

        self.protected = protected
        self.step = step
        self.device = device
        self.model = deepcopy(model).to(self.device)
        self.idx_attack = idx_attack
        self.labels = labels.to(device)
        self.attr_adversary = None
        self.adj_adversary = None

    
    def attack(self, 
               n_perturbations: int, 
               **kwargs):

        #? I think m is just n_perturbations and we do not need it in our code!
        m = self.m

        alpha = self.alpha
        k = self.k
        rng = self.rng
        n = self.n
        step = self.step
        protected = self.protected
        adj = self.adj
        X = self.X
        
        #Store for each edge if it is eligible for flipping
        s = np.ones(self.n * (self.n - 1) // 2)

        # Lower-triangular indices for conversion between linear indices and matrix positions
        row, col = torch.tril_indices(self.n, self.n, offset=-1)
        row, col = row.numpy(), col.numpy()

        # Exclude any protected edges
        if self.protected is not None:
            for p in self.protected.cpu().numpy():
                s[(row == p) | (col == p)] = False
        
        n_edges = np.count_nonzero(s)
        assert n_edges >= k

        clean_ll = self.log_likelihood()
        bar = tqdm(total=n_perturbations, desc="Cycles") #//total = m
        # Randomly select a cohort of edges to flip
        cohort_size = min(int(self.alpha * self.k), n_edges)
        cohort = list(self.rng.choice(s.size, size=cohort_size, replace=False, p=s / s.sum()))
        #//flip_edges_(graph.a, row[cohort], col[cohort])
        self.flip_edges(adj, row, col, cohort)
        s[cohort] = False

        n_expand = min(round((self.alpha - 1.0) * k), n_edges - k)
        if n_expand > 0:
            for _ in range(n_perturbations - 1): #// in range(m-1)
                self.contract(adj,cohort,row,col,s)

                ll = self.log_likelihood()
                bar.set_description(f"Cycles (decr {clean_ll - ll:.5f})")
                bar.update()

                # Expand the cohort again
                new_edges = rng.choice(s.size, size=n_expand, replace=False, p=s / s.sum())
                #//flip_edges_(graph.a, row[new_edges], col[new_edges])
                self.flip_edges(adj, row, col, new_edges)
                s[new_edges] = False
                cohort.extend(list(new_edges))

        self.contract(adj,cohort,row,col, s)
        bar.update()

        #//graph.a = orig_a
        #? Does self.adj or just adj makes a difference here?? 
        #? If self.adj is the same as adj, are we saving the original adj for new attacks or destroying it?
        self.adj_adversary = self.adj.to_sparse().detach()
        self.attr_adversary = self.X

        #? What was this line supposed to do?
        #//return torch.from_numpy(np.vstack([row[cohort], col[cohort]])).to(graph.a.device())




