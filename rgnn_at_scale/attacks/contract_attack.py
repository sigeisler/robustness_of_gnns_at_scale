import numpy as np
from numpy.core.function_base import _add_docstring
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union
from rgnn_at_scale.models import DenseGCN
from copy import deepcopy
from rgnn_at_scale import utils


class EXPAND_CONTRACT():

    def __init__(self,
                 adj: torch.sparse.FloatTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 device: Union[str, int, torch.device],
                 k: int = 1,
                 alpha: float = 2.0,
                 m: int = 10,
                 step: int = 1,
                 protected=None,
                 **kwargs
                 ):
        assert alpha >= 1.0
        assert m >= 1
        assert adj.device == X.device
        self.alpha = alpha
        # * m is the number of times we filter our selected nodes
        self.m = m
        self.k = k
        self.n = adj.size()[0]
        self.adj = adj
        adj_symmetric_index, adj_symmetric_weights = utils.to_symmetric(adj.indices(), adj.values(), self.n)
        self.adj_dict, self.adj_symmetric_index, self.adj_symmetric_weights = self._to_dict(
            adj_symmetric_index, adj_symmetric_weights)
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
        """Calculates the log likelihood error of the model prediction for self.idx_attack nodes

        Args:
            adj_dict (Dict): dictionary describing the adjacency matrix
            adj_symmetric_index (sparse.Tensor): sparse tensor describing the connected indices of adjacency matrix
            adj_symmetric_weights (sparse.Tensor): sparse tensor containing the values of the connected indices of adjacency matrix

        Returns:
            [float]: [the log likelihood error for the nodes' prediction]
        """
        # ? Would it speed up the calculation of self.X and the two tensors get moved to GPU, so model calculation become faster??
        adj_symmetric_index = adj_symmetric_index.to(self.device)
        adj_symmetric_weights = adj_symmetric_weights.to(self.device)
        adj = torch.sparse.FloatTensor(adj_symmetric_index, adj_symmetric_weights, torch.Size([self.n, self.n]))
        logits = self.model.to(self.device)(self.X, adj)
        # ? And then back to CPU??
        adj_symmetric_index = adj_symmetric_index.to("cpu")
        adj_symmetric_weights = adj_symmetric_weights.to("cpu")

        return -F.cross_entropy(logits[self.idx_attack], self.labels[self.idx_attack])

    def _delete_edge_from_tensor(self, source, dest, adj_symmetric_index, adj_symmetric_weights):
        """Deletes edges from sparse Tensor representation of the adjacency Matrix

        Args:
            source (int): source Node
            dest (int): destination Node
            adj_symmetric_index (sparse.Tensor): sparse tensor describing the connected indices of adjacency matrix
            adj_symmetric_weights ([type]): sparse tensor containing the values of the connected indices of adjacency matrix

        Returns:
            sparse.Tensor, sparse.Tensor : the two input sparse Tensors are returned after removing the edge
        """
        # Get all occurences of the source Node
        indices = torch.nonzero((adj_symmetric_index[0] == source), as_tuple=True)[0]
        # Get the occurences of dest Node where it is connected with source Node
        dest_index = indices[torch.nonzero((adj_symmetric_index[1][indices] == dest), as_tuple=True)[0]]
        # Remove that connection
        adj_symmetric_index = torch.cat(
            (adj_symmetric_index[:, 0:dest_index],  adj_symmetric_index[:, dest_index + 1:]), 1)
        adj_symmetric_weights = torch.cat(
            (adj_symmetric_weights[0:dest_index],  adj_symmetric_weights[dest_index + 1:]))
        return adj_symmetric_index, adj_symmetric_weights

    def _flip_edges(self, adj_dict, edges, adj_symmetric_index, adj_symmetric_weights):
        """Flip edges in teh adjacency Matrix(both in Dictionary and sparse Tensor representation)

        Args:
            adj_dict (Dict): Dict representation of Adjacency Matrix
            edges (list): list of edges to be flipped
            adj_symmetric_index (sparse.Tensor): sparse tensor describing the connected indices of adjacency matrix
            adj_symmetric_weights (sparse.Tensor): sparse tensor containing the values of the connected indices of adjacency matrix

        Returns:
            Dict, sparse.Tensor, sparse.Tensor: The updated representation of adjacency matrix both in Dictionary and sparse Tensor Format
        """
        if type(edges) is not list:
            edges = [edges]
        for source, dest in edges:
            if(not adj_dict.get((source, dest))):
                adj_dict[(source, dest)] = 1
                # ?keep lists updated to make log_likelihood function more computationally efficient??
                adj_symmetric_index = torch.cat((adj_symmetric_index, torch.LongTensor([[source], [dest]])), 1)
                adj_symmetric_weights = torch.cat((adj_symmetric_weights, torch.tensor([1])))
            else:
                adj_dict.pop((source, dest), None)
                adj_symmetric_index, adj_symmetric_weights = self._delete_edge_from_tensor(
                    source, dest, adj_symmetric_index, adj_symmetric_weights)
        return adj_dict, adj_symmetric_index, adj_symmetric_weights

    def _contract(self, adj_dict, cohort, adj_symmetric_index, adj_symmetric_weights, n_perturbations):
        """Performing the attack

        Args:
            adj_dict (Dict): Dict representation of Adjacency Matrix
            cohort (list): list of edges to choose the best n_perturbations edges from
            adj_symmetric_index (sparse.Tensor): sparse tensor describing the connected indices of adjacency matrix
            adj_symmetric_weights (sparse.Tensor): sparse tensor containing the values of the connected indices of adjacency matrix
            n_perturbations (int): number of flipped nodes to be chosen

        Returns:
            Dict, sparse.Tensor, sparse.Tensor: The updated representation of adjacency matrix both in Dictionary and sparse Tensor Format
        """
        # Contract the cohort until we are in budget again
        bar = tqdm(total=len(cohort) - n_perturbations, leave=False, desc="Contract")
        # we retain n_perturbation nodes
        while len(cohort) > n_perturbations:  # //self.k:
            llhs = np.empty(len(cohort))
            bar2 = tqdm(total=len(cohort), leave=False, desc="Flipping Edges")
            for i, edge in enumerate(cohort):
                adj_dict, adj_symmetric_index, adj_symmetric_weights = self._flip_edges(
                    adj_dict, edge, adj_symmetric_index, adj_symmetric_weights)
                llhs[i] = self._log_likelihood(adj_dict, adj_symmetric_index, adj_symmetric_weights)
                adj_dict, adj_symmetric_index, adj_symmetric_weights = self._flip_edges(
                    adj_dict, edge, adj_symmetric_index, adj_symmetric_weights)
                bar2.update(1)
            # Undo the flips that increase the log-likelihood the least when undone
            # //n_undo = min(self.step, len(cohort) - self.k)
            n_undo = min(self.step, len(cohort) - n_perturbations)
            for edge in [cohort[i] for i in np.argpartition(llhs, n_undo)[:n_undo]]:
                cohort.remove(edge)
                adj_dict, adj_symmetric_index, adj_symmetric_weights = self._flip_edges(
                    adj_dict, edge, adj_symmetric_index, adj_symmetric_weights)
            bar.update(n_undo)
            bar2.close()
        bar.close()

        return adj_dict, adj_symmetric_index, adj_symmetric_weights

    def _add_edges_to_list(self, my_list, number_of_nodes_to_add):
        """Add new random edges to a list

        Args:
            my_list (list): list of tuples to which new edges should be added
            number_of_nodes_to_add (int): how many new nodes are to be added

        Returns:
            list: list of tuples containing new edges
        """
        count = 0
        while count < number_of_nodes_to_add:
            source = np.random.randint(self.n)
            dest = np.random.randint(self.n)
            source, dest = (source, dest) if source < dest else (dest, source)
            if(source != dest):
                my_list.append((source, dest))
                count += 1
        return my_list

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
        return myAdj, adj_symmetric_index, adj_symmetric_weights

    def attack(self,
               n_perturbations: int,
               **kwargs):
        k = self.k
        adj_dict = self.adj_dict.copy()
        adj_symmetric_index = self.adj_symmetric_index.detach().clone()
        adj_symmetric_weights = self.adj_symmetric_weights.detach().clone()
        # How many possible edges in the upper triangular
        n_edges = (self.n * (self.n - 1)) // 2
        assert n_edges >= k
        bar = tqdm(total=self.m, desc="Cycles")
        # Randomly select a cohort of edges to flip
        # * Cohort size is k*n_perturbation, we keep just n_perturbation nodes at the end
        # //cohort_size = min(int(self.alpha * self.k), n_edges)
        cohort_size = min(int(n_perturbations * self.k), n_edges)
        cohort = []
        cohort = self._add_edges_to_list(cohort, cohort_size)
        adj_dict, adj_symmetric_index, adj_symmetric_weights = self._flip_edges(
            adj_dict, cohort, adj_symmetric_index, adj_symmetric_weights)
        # *How many new nodes to add to cohort after each filteration step
        # //n_expand = min(round((self.alpha - 1.0) * k), n_edges - k)
        n_expand = min(round((k - 1) * n_perturbations), n_edges - k)
        if n_expand > 0:
            # refine the choice of nodes m times
            for _ in range(self.m - 1):  # //range(n_perturbations - 1):
                adj_dict, adj_symmetric_index, adj_symmetric_weights = self._contract(
                    adj_dict, cohort, adj_symmetric_index, adj_symmetric_weights, n_perturbations)
                bar.update()
                new_edges = self._add_edges_to_list([], n_expand)
                adj_dict, adj_symmetric_index, adj_symmetric_weights = self._flip_edges(
                    adj_dict, new_edges, adj_symmetric_index, adj_symmetric_weights)
                cohort.extend(list(new_edges))

        adj_dict, adj_symmetric_index, adj_symmetric_weights = self._contract(
            adj_dict, cohort, adj_symmetric_index, adj_symmetric_weights, n_perturbations)
        bar.update()
        adj_symmetric_index, adj_symmetric_weights = utils.to_symmetric(
            adj_symmetric_index, adj_symmetric_weights, self.n)
        self.adj_adversary = torch.sparse.FloatTensor(
            adj_symmetric_index, adj_symmetric_weights, torch.Size([self.n, self.n]))
        self.attr_adversary = self.X
