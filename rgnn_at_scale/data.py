"""Utils to retrieve/split/... the data.
"""
import logging

from pathlib import Path
from typing import Any, Dict, Iterable, List, Union, Tuple, Optional

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import warnings

import numpy as np
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

import torch
import torch_sparse
from torch_sparse import SparseTensor

from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr


patch_typeguard()

sparse_graph_properties = [
    'adj_matrix', 'attr_matrix', 'labels', 'node_names', 'attr_names', 'class_names', 'metadata'
]


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    All properties are immutable so users don't mess up the
    data format's assumptions.
    Be careful when circumventing this and changing the internal matrices
    regardless (e.g. by exchanging the data array of a sparse matrix).

    Parameters
    ----------
    adj_matrix
        Adjacency matrix in CSR format. Shape [num_nodes, num_nodes]
    attr_matrix
        Attribute matrix in CSR or numpy format. Shape [num_nodes, num_attr]
    labels
        Array, where each entry represents respective node's label(s). Shape [num_nodes]
        Alternatively, CSR matrix with labels in one-hot format. Shape [num_nodes, num_classes]
    node_names
        Names of nodes (as strings). Shape [num_nodes]
    attr_names
        Names of the attributes (as strings). Shape [num_attr]
    class_names
        Names of the class labels (as strings). Shape [num_classes]
    metadata
        Additional metadata such as text.

    """

    def __init__(
            self, adj_matrix: sp.spmatrix,
            attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            labels: Union[np.ndarray, sp.spmatrix] = None,
            node_names: np.ndarray = None,
            attr_names: np.ndarray = None,
            class_names: np.ndarray = None,
            metadata: Any = None):
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)."
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree.")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree.")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree.")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree.")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree.")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph.
        """
        return self.adj_matrix.shape[0]

    def num_edges(self, warn: bool = True) -> int:
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as _two_ edges.

        """
        if warn and not self.is_directed():
            warnings.warn("num_edges always returns the number of directed edges now.", FutureWarning)
        return self.adj_matrix.nnz

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def get_edgeid_to_idx_array(self) -> np.ndarray:
        """Return a Numpy Array that maps edgeids to the indices in the adjacency matrix.

        Returns
        -------
        np.ndarray
            The i'th entry contains the x- and y-coordinates of edge i in the adjacency matrix.
            Shape [num_edges, 2]

        """
        return np.transpose(self.adj_matrix.nonzero())

    def get_idx_to_edgeid_matrix(self) -> sp.csr_matrix:
        """Return a sparse matrix that maps indices in the adjacency matrix to edgeids.

        Caution: This contains one explicit 0 (zero stored as a nonzero),
        which is the index of the first edge.

        Returns
        -------
        sp.csr_matrix
            The entry [x, y] contains the edgeid of the corresponding edge (or 0 for non-edges).
            Shape [num_nodes, num_nodes]

        """
        return sp.csr_matrix(
            (np.arange(self.adj_matrix.nnz), self.adj_matrix.indices, self.adj_matrix.indptr),
            shape=self.adj_matrix.shape)

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric).
        """
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self) -> 'SparseGraph':
        """Convert to an undirected graph (make adjacency matrix symmetric).
        """
        idx = self.get_edgeid_to_idx_array().T
        ridx = np.ravel_multi_index(idx, self.adj_matrix.shape)
        ridx_rev = np.ravel_multi_index(idx[::-1], self.adj_matrix.shape)

        # Get duplicate edges (self-loops and opposing edges)
        dup_ridx = ridx[np.isin(ridx, ridx_rev)]
        dup_idx = np.unravel_index(dup_ridx, self.adj_matrix.shape)

        # Check if the adjacency matrix weights are symmetric (if nonzero)
        if len(dup_ridx) > 0 and not np.allclose(self.adj_matrix[dup_idx], self.adj_matrix[dup_idx[::-1]]):
            raise ValueError("Adjacency matrix weights of opposing edges differ.")

        # Create symmetric matrix
        new_adj_matrix = self.adj_matrix + self.adj_matrix.T
        if len(dup_ridx) > 0:
            new_adj_matrix[dup_idx] = (new_adj_matrix[dup_idx] - self.adj_matrix[dup_idx]).A1

        self.adj_matrix = new_adj_matrix
        return self

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1).
        """
        return np.any(np.unique(self.adj_matrix[self.adj_matrix.nonzero()].A1) != 1)

    def to_unweighted(self) -> 'SparseGraph':
        """Convert to an unweighted graph (set all edge weights to 1).
        """
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def is_connected(self) -> bool:
        """Check if the graph is connected.
        """
        return sp.csgraph.connected_components(self.adj_matrix, return_labels=False) == 1

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops.
        """
        return not np.allclose(self.adj_matrix.diagonal(), 0)

    def __repr__(self) -> str:

        dir_string = 'Directed' if self.is_directed() else 'Undirected'
        weight_string = 'weighted' if self.is_weighted() else 'unweighted'
        conn_string = 'connected' if self.is_connected() else 'disconnected'
        loop_string = 'has self-loops' if self.has_self_loops() else 'no self-loops'
        return ("<{}, {} and {} SparseGraph with {} edges ({})>"
                .format(dir_string, weight_string, conn_string,
                        self.num_edges(warn=False), loop_string))

    def _adopt_graph(self, graph: 'SparseGraph'):
        """Copy all properties from the given graph to this graph.
        """
        for prop in sparse_graph_properties:
            setattr(self, '_{}'.format(prop), getattr(graph, prop))

    # Quality of life (shortcuts)
    def standardize(
            self, make_unweighted: bool = True,
            make_undirected: bool = True,
            no_self_loops: bool = True,
            select_lcc: bool = True
    ) -> 'SparseGraph':
        """Perform common preprocessing steps: remove self-loops, make unweighted/undirected, select LCC.

        All changes are done inplace.

        Parameters
        ----------
        make_unweighted
            Whether to set all edge weights to 1.
        make_undirected
            Whether to make the adjacency matrix symmetric. Can only be used if make_unweighted is True.
        no_self_loops
            Whether to remove self loops.
        select_lcc
            Whether to select the largest connected component of the graph.

        """
        G = self
        if make_unweighted and G.is_weighted():
            G = G.to_unweighted()
        if make_undirected and G.is_directed():
            G = G.to_undirected()
        if no_self_loops and G.has_self_loops():
            G = remove_self_loops(G)
        if select_lcc and not G.is_connected():
            G = largest_connected_components(G, 1, make_undirected)
        self._adopt_graph(G)
        return G

    @staticmethod
    def from_flat_dict(data_dict: Dict[str, Any]) -> 'SparseGraph':
        """Initialize SparseGraph from a flat dictionary.
        """
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith('_data') or key.endswith('.data'):
                if key.endswith('_data'):
                    sep = '_'
                else:
                    sep = '.'
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = '{}{}indices'.format(matrix_name, sep)
                mat_indptr = '{}{}indptr'.format(matrix_name, sep)
                mat_shape = '{}{}shape'.format(matrix_name, sep)
                if matrix_name == 'adj' or matrix_name == 'attr':
                    matrix_name += '_matrix'
                init_dict[matrix_name] = sp.csr_matrix(
                    (data_dict[mat_data],
                     data_dict[mat_indices],
                     data_dict[mat_indptr]),
                    shape=data_dict[mat_shape])
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Delete sparse matrix entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Load everything else
        for key, val in data_dict.items():
            if ((val is not None) and (None not in val)):
                init_dict[key] = val

        return SparseGraph(**init_dict)


@typechecked
def largest_connected_components(sparse_graph: SparseGraph, n_components: int = 1, make_undirected=True) -> SparseGraph:
    """Select the largest connected components in the graph.

    Changes are returned in a partially new SparseGraph.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        Number of largest connected components to keep.

    Returns
    -------
    SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix, directed=make_undirected)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


@typechecked
def remove_self_loops(sparse_graph: SparseGraph) -> SparseGraph:
    """Remove self loops (diagonal entries in the adjacency matrix).

    Changes are returned in a partially new SparseGraph.

    """
    num_self_loops = (~np.isclose(sparse_graph.adj_matrix.diagonal(), 0)).sum()
    if num_self_loops > 0:
        adj_matrix = sparse_graph.adj_matrix.copy().tolil()
        adj_matrix.setdiag(0)
        adj_matrix = adj_matrix.tocsr()
        return SparseGraph(
            adj_matrix, sparse_graph.attr_matrix, sparse_graph.labels, sparse_graph.node_names,
            sparse_graph.attr_names, sparse_graph.class_names, sparse_graph.metadata)
    else:
        return sparse_graph


def create_subgraph(
        sparse_graph: SparseGraph,
        _sentinel: None = None,
        nodes_to_remove: np.ndarray = None,
        nodes_to_keep: np.ndarray = None
) -> SparseGraph:
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    The subgraph partially points to the old graph's data.

    Parameters
    ----------
    sparse_graph
        Input graph.
    _sentinel
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove
        Indices of nodes that have to removed.
    nodes_to_keep
        Indices of nodes that have to be kept.

    Returns
    -------
    SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...).")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is None:
        attr_matrix = None
    else:
        attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is None:
        labels = None
    else:
        labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is None:
        node_names = None
    else:
        node_names = sparse_graph.node_names[nodes_to_keep]
    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, sparse_graph.attr_names,
        sparse_graph.class_names, sparse_graph.metadata)


@typechecked
def load_dataset(name: str,
                 directory: Union[Path, str] = './data'
                 ) -> SparseGraph:
    """Load a dataset.

    Parameters
    ----------
    name
        Name of the dataset to load.
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    SparseGraph
        The requested dataset in sparse format.

    """
    if isinstance(directory, str):
        directory = Path(directory)
    path_to_file = directory / (name + ".npz")
    if path_to_file.exists():
        with np.load(path_to_file, allow_pickle=True) as loader:
            loader = dict(loader)
            del loader['type']
            del loader['edge_attr_matrix']
            del loader['edge_attr_names']
            dataset = SparseGraph.from_flat_dict(loader)
    else:
        raise ValueError("{} doesn't exist.".format(path_to_file))
    return dataset


@typechecked
def train_val_test_split_tabular(
        *arrays: Iterable[Union[np.ndarray, sp.spmatrix]],
        train_size: float = 0.5,
        val_size: float = 0.3,
        test_size: float = 0.2,
        stratify: np.ndarray = None,
        random_state: int = None
) -> List[Union[np.ndarray, sp.spmatrix]]:
    """Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices with the same length / shape[0].
    train_size
        Proportion of the dataset included in the train split.
    val_size
        Proportion of the dataset included in the validation split.
    test_size
        Proportion of the dataset included in the test split.
    stratify
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state
        Random_state is the seed used by the random number generator;

    Returns
    -------
    list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def split(labels, n_per_class=20, seed=None):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [num_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test


@typechecked
def prep_cora_citeseer_pubmed(name: str,
                              dataset_root: str,
                              device: Union[int, str, torch.device] = 0,
                              make_undirected: bool = True) -> Tuple[TensorType["num_nodes", "num_features"],
                                                                     SparseTensor,
                                                                     TensorType["num_nodes"]]:
    """Prepares and normalizes the desired dataset

    Parameters
    ----------
    name : str
        Name of the data set. One of: `cora_ml`, `citeseer`, `pubmed`
    dataset_root : str
        Path where to find/store the dataset.
    device : Union[int, torch.device], optional
        `cpu` or GPU id, by default 0
    make_undirected : bool, optional
        Normalize adjacency matrix with symmetric degree normalization (non-scalable implementation!), by default False

    Returns
    -------
    Tuple[torch.Tensor, SparseTensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor
    """
    graph = load_dataset(name, dataset_root).standardize(
        make_unweighted=True,
        make_undirected=make_undirected,
        no_self_loops=True,
        select_lcc=True
    )

    attr = torch.FloatTensor(graph.attr_matrix.toarray()).to(device)
    adj = utils.sparse_tensor(graph.adj_matrix.tocoo()).to(device)

    labels = torch.LongTensor(graph.labels).to(device)

    return attr, adj, labels


@typechecked
def prep_graph(name: str,
               device: Union[int, str, torch.device] = 0,
               make_undirected: bool = True,
               binary_attr: bool = False,
               feat_norm: bool = False,
               dataset_root: str = 'data',
               return_original_split: bool = False) -> Tuple[TensorType["num_nodes", "num_features"],
                                                             SparseTensor,
                                                             TensorType["num_nodes"],
                                                             Optional[Dict[str, np.ndarray]]]:
    """Prepares and normalizes the desired dataset

    Parameters
    ----------
    name : str
        Name of the data set. One of: `cora_ml`, `citeseer`, `pubmed`
    device : Union[int, torch.device]
        `cpu` or GPU id, by default 0
    binary_attr : bool, optional
        If true the attributes are binarized (!=0), by default False
    dataset_root : str, optional
        Path where to find/store the dataset, by default "datasets"
    return_original_split: bool, optional
        If true (and the split is available for the choice of dataset) additionally the original split is returned.

    Returns
    -------
    Tuple[torch.Tensor, torch_sparse.SparseTensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor.
    """
    split = None

    logging.debug("Memory Usage before loading the dataset:")
    logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    if name in ['cora_ml', 'citeseer', 'pubmed']:
        attr, adj, labels = prep_cora_citeseer_pubmed(name, dataset_root, device, make_undirected)
    elif name.startswith('ogbn'):
        pyg_dataset = PygNodePropPredDataset(root=dataset_root, name=name)

        data = pyg_dataset[0]

        if hasattr(data, '__num_nodes__'):
            num_nodes = data.__num_nodes__
        else:
            num_nodes = data.num_nodes

        if hasattr(pyg_dataset, 'get_idx_split'):
            split = pyg_dataset.get_idx_split()
        else:
            split = dict(
                train=data.train_mask.nonzero().squeeze(),
                valid=data.val_mask.nonzero().squeeze(),
                test=data.test_mask.nonzero().squeeze()
            )

        # converting to numpy arrays, so we don't have to handle different
        # array types (tensor/numpy/list) later on.
        # Also we need numpy arrays because Numba cant determine type of torch.Tensor
        split = {k: v.numpy() for k, v in split.items()}

        edge_index = data.edge_index.cpu()
        if data.edge_attr is None:
            edge_weight = torch.ones(edge_index.size(1))
        else:
            edge_weight = data.edge_attr
        edge_weight = edge_weight.cpu()

        adj = sp.csr_matrix((edge_weight, edge_index), (num_nodes, num_nodes))

        del edge_index
        del edge_weight

        # make unweighted
        adj.data = np.ones_like(adj.data)

        if make_undirected:
            adj = utils.to_symmetric_scipy(adj)

            logging.debug("Memory Usage after making the graph undirected:")
            logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

        logging.debug("Memory Usage after normalizing the graph")
        logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

        adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)

        attr_matrix = data.x.cpu().numpy()

        attr = torch.from_numpy(attr_matrix).to(device)

        logging.debug("Memory Usage after normalizing graph attributes:")
        logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

        labels = data.y.squeeze().to(device)
    else:
        raise NotImplementedError(f"Dataset `with name '{name}' is not supported")

    if binary_attr:
        # NOTE: do not use this for really large datasets.
        # The mask is a **dense** matrix of the same size as the attribute matrix
        attr[attr != 0] = 1
    elif feat_norm:
        attr = utils.row_norm(attr)

    if return_original_split and split is not None:
        return attr, adj, labels, split

    return attr, adj, labels, None


class RobustPPRDataset(torch.utils.data.Dataset):

    @typechecked
    def __init__(self,
                 attr_matrix_all: TensorType["num_nodes", "num_features"],
                 ppr_matrix: sp.csr.csr_matrix,
                 indices: np.ndarray,
                 labels_all: TensorType["num_nodes"] = None,
                 allow_cache: bool = True):
        """
        Parameters:
            attr_matrix_all: torch.Tensor of shape (num_nodes, num_features)
                Node features / attributes of all nodes in the graph
            ppr_matrix: scipy.sparse.csr.csr_matrix of shape (num_training_nodes, num_nodes)
                The personal page rank vectors for all nodes of the training set
            indices: array-like of shape (num_training_nodes)
                The ids of the training nodes
            labels_all: torch.Tensor of shape (num_nodes)
                The class labels for all nodes in the graph
        """
        self.attr_matrix_all = attr_matrix_all
        self.ppr_matrix = ppr_matrix
        self.indices = indices
        self.labels_all = torch.tensor(labels_all, dtype=torch.long) if labels_all is not None else None
        self.allow_cache = allow_cache
        self.cached = {}

    def __len__(self):
        return self.indices.shape[0]

    @typechecked
    def __getitem__(self, idx: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray,
                                                                      Tuple[TensorType["ppr_nnz",
                                                                                       "num_features"], SparseTensor],
                                                                      Optional[TensorType["batch_size"]]]:
        """
        Parameters:
            idx: np.ndarray of shape (batch_size)
                The relative id of nodes in the RobustPPRDataset instance
        Returns:
            A tuple (indices, data, labels), where
                indices:
                    The absolut indices of the nodes in the batch w.r.t the original
                    indexing defined by the original dataset (e.g. ogbn-datsets)
                data: tuple of
                    - attr_matrix: torch.Tensor of shape (ppr_num_nonzeros, num_features)
                        The node features of all neighboring nodes of the training nodes in
                        the graph derived from the Personal Page Rank as specified by idx
                    - ppr_matrix: torch_sparse.SparseTensor of shape (batch_size, ppr_num_nonzeros)
                        The page rank scores of all neighboring nodes of the training nodes in
                        the graph derived from the Personal Page Rank as specified by idx
                label: np.ndarray of shape (batch_size)
                    The labels of the nodes in the batch
        """
        # for performance reasons just checking if first element
        # of the batch is cached. If it is, it is assummed that all other
        # elements of the batch have also been cached. This implicitely
        # assumes that individual batches always contian the same elements
        key = idx[0]
        if key not in self.cached:
            # shape (batch_size, num_nodes)
            ppr_matrix = utils.matrix_to_torch(self.ppr_matrix[idx])

            # shape (ppr_num_nonzeros)
            source_idx, neighbor_idx, ppr_scores = ppr_matrix.coo()

            ppr_matrix = ppr_matrix[:, neighbor_idx.unique()]
            attr_matrix = self.attr_matrix_all[neighbor_idx.unique()]

            if self.labels_all is None:
                labels = None
            else:
                labels = self.labels_all[self.indices[idx]]

            batch = (self.indices[idx], (attr_matrix, ppr_matrix), labels)

            if self.allow_cache:
                self.cached[key] = batch
            else:
                return batch

        return self.cached[key]


INT_TYPES = (int, np.integer)


class CachedPPRMatrix:
    """
    TODO: Add docstring
    """

    def __init__(self,
                 adj: SparseTensor,
                 ppr_cache_params: Dict[str, Any],
                 alpha: float,
                 eps: float,
                 topk: int,
                 ppr_normalization: str,
                 use_train_val_ppr: bool = True,
                 ppr_values_on_demand: bool = False):

        logging.info("Memory Usage before creating CachedPPRMatrix:")
        logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

        self.adj = adj.to_scipy(layout="csr")
        self.ppr_cache_params = ppr_cache_params
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.ppr_normalization = ppr_normalization

        n = self.adj.shape[0]
        self.shape = (n, n)
        self.storage = None
        self.csr_ppr = None
        self.coo_ppr = None
        self.ppr_values_on_demand = ppr_values_on_demand

        logging.info("Memory Usage before loading CachedPPRMatrix from storage:")
        logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

        if self.ppr_cache_params is not None:
            self.storage_params = dict(dataset=self.ppr_cache_params["dataset"],
                                       alpha=self.alpha,
                                       eps=self.eps,
                                       topk=self.topk,
                                       split_desc="attack",
                                       ppr_normalization=self.ppr_normalization,
                                       make_undirected=self.ppr_cache_params["make_undirected"])
            # late import to avoid circular import issue
            from rgnn_at_scale.helper.io import Storage
            self.storage = Storage(self.ppr_cache_params["data_artifact_dir"])
            stored_topk_ppr = self.storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                              self.storage_params, find_first=True)

            self.csr_ppr, _ = stored_topk_ppr[0] if len(stored_topk_ppr) == 1 else (None, None)
            logging.info(
                f"Memory after loading 'Attack' CachedPPRMatrix: {utils.get_max_memory_bytes() / (1024 ** 3)}")

        self.has_missing_ppr_values = True if self.csr_ppr is None else (self.csr_ppr.sum(-1) == 0).any()

        if self.csr_ppr is None and use_train_val_ppr and self.storage is not None:
            stored_pprs = self._load_partial_pprs()
            self.coo_ppr = sp.coo_matrix(self.shape, dtype=self.adj.dtype)
            self._join_partial_pprs_with_base(stored_pprs)
            self.csr_ppr = self.coo_ppr.tocsr()
            logging.info(
                f'Memory after building ppr from train/val/test ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

        if self.csr_ppr is None:
            self.csr_ppr = sp.csr_matrix(self.shape, dtype=self.adj.dtype)

        if self.coo_ppr is None and (self.ppr_values_on_demand or self.has_missing_ppr_values):
            self.coo_ppr = self.csr_ppr.tocoo()
            logging.info(f'Memory after initalizing coo_ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

        if self.has_missing_ppr_values:
            rows, _ = self.csr_ppr.nonzero()
            logging.info("Memory after .nonzero() ")
            logging.info(utils.get_max_memory_bytes() / (1024 ** 3))
            # make this look up table
            self.cached_csr_rows = np.array(np.unique(rows))
            logging.info("Memory after self.cached_csr_rows:")
            logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

            if not self.ppr_values_on_demand:
                # calculate all ppr scores beforehand, instead of calculating them on demand
                # this improves training/attack speed as there's no need to check whether the
                # required ppr scores are already cached or not
                missing_ppr_idx = self._get_uncached(np.arange(self.shape[0]))
                if len(missing_ppr_idx) > 0:
                    self._calc_ppr(missing_ppr_idx)
                    self.save_to_storage()
                    logging.info(
                        f"Memory after computing all missing ppr values:{utils.get_max_memory_bytes() / (1024 ** 3)}")
                    self.has_missing_ppr_values = False

        logging.info("Memory after loading CachedPPRMatrix from storage:")
        logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

    def _load_partial_pprs(self):
        del self.storage_params["split_desc"]
        stored_ppr_documents = self.storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                               self.storage_params, return_documents_only=True)
        stored_pprs = []
        if len(stored_ppr_documents) > 0:
            df_documents = pd.DataFrame(list(map(lambda doc: doc["params"], stored_ppr_documents)))
            df_documents["id"] = df_documents.index
            df_documents["ppr_idx"] = df_documents["ppr_idx"].apply(lambda x: list(x))
            df_cross = df_documents.merge(df_documents, how="cross", suffixes=[
                "_1", "_2"]).merge(df_documents, how="cross")

            df_cross["joint_ppr_idx"] = df_cross["ppr_idx"] + df_cross["ppr_idx_1"] + df_cross["ppr_idx_2"]
            df_cross["joint_ppr_unique"] = df_cross["joint_ppr_idx"].apply(lambda x: np.unique(x))
            df_cross["joint_ppr_unique_len"] = df_cross["joint_ppr_unique"].apply(lambda x: len(x))
            df_cross["joint_ppr_diff"] = df_cross["joint_ppr_idx"].apply(
                lambda x: len(x)) - df_cross["joint_ppr_unique_len"]

            df_cross = df_cross.sort_values(['joint_ppr_unique_len', 'joint_ppr_diff'], ascending=[False, True])

            doc_ids_to_read = list(df_cross.iloc[0][["id", "id_1", "id_2"]])

            for i in doc_ids_to_read:
                doc = stored_ppr_documents[i]
                path = self.storage._build_artifact_path(self.ppr_cache_params["data_storage_type"],
                                                         doc.doc_id).replace(".pt", ".npz")
                sparse_matrix = sp.load_npz(path).tocoo()
                stored_pprs.append((sparse_matrix, doc["params"]["ppr_idx"]))

            logging.info("Memory Usage loading CachedPPRMatrix from storage:")
            logging.info(utils.get_max_memory_bytes() / (1024 ** 3))
        return stored_pprs

    def _join_partial_pprs_with_base(self, stored_pprs: List[Tuple[sp.coo_matrix, np.ndarray]]):
        for stored_ppr, ppr_idx in stored_pprs:

            # don't add duplicate entries
            new_ppr_idx_mask = np.isin(ppr_idx, self.coo_ppr.row, invert=True)
            new_ppr_idx = ppr_idx[new_ppr_idx_mask]
            if len(new_ppr_idx) == 0:
                continue

            _, index = np.unique(stored_ppr.row, return_inverse=True)
            rows = ppr_idx[index]

            new_ppr_values_mask = np.isin(rows, new_ppr_idx)
            rows = rows[new_ppr_values_mask]
            cols = stored_ppr.col[new_ppr_values_mask]
            data = stored_ppr.data[new_ppr_values_mask]

            self.coo_ppr.row = np.concatenate([self.coo_ppr.row, rows])
            self.coo_ppr.col = np.concatenate([self.coo_ppr.col, cols])
            self.coo_ppr.data = np.concatenate([self.coo_ppr.data, data])

            logging.info("Memory Usage loading CachedPPRMatrix from storage:")
            logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

    def _sync_pprs(self):
        if self.has_missing_ppr_values:
            logging.info("Updating csr ppr matrix...")
            if self.csr_ppr is None:
                self.csr_ppr = self.coo_ppr.tocsr()
            elif self.coo_ppr is None:
                self.coo_ppr = self.csr_ppr.tocoo()
            elif len(self.csr_ppr.data) < len(self.coo_ppr.data):
                self.csr_ppr = self.coo_ppr.tocsr()
            elif len(self.csr_ppr.data) > len(self.coo_ppr.data):
                self.coo_ppr = self.csr_ppr.tocoo()
            logging.info(
                f"Memory after syncing csr and coo ppr representation :{utils.get_max_memory_bytes() / (1024 ** 3)}")

    def save_to_storage(self):
        self._sync_pprs()
        if self.storage is not None and self.has_missing_ppr_values:
            logging.info("Save ppr to storage")
            self.storage_params["split_desc"] = "attack"
            rows, _ = self.csr_ppr.nonzero()
            self.storage_params["ppr_idx"] = np.unique(rows)
            self.storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                            self.storage_params,
                                            self.csr_ppr, ignore_duplicate=True)
            logging.info(
                f"Memory after  saving CachedPPRMatrix to storage:{utils.get_max_memory_bytes() / (1024 ** 3)}")

    def _calc_ppr(self, new_ppr_idx: Union[List[int], np.ndarray]):
        if len(new_ppr_idx) > 0:

            logging.info(f"Calculating {len(new_ppr_idx)} ppr scores for CachedPPRMatrix...")
            ppr_scores = ppr.topk_ppr_matrix(self.adj, self.alpha, self.eps, new_ppr_idx.copy(),
                                             self.topk, normalization=self.ppr_normalization)
            ppr_scores = ppr_scores.tocoo()

            logging.info(f"Memory after calculating {len(new_ppr_idx)} ppr scores for CachedPPRMatrix")
            logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

            _, index = np.unique(ppr_scores.row, return_inverse=True)
            rows = new_ppr_idx[index]

            self.coo_ppr.row = np.concatenate([self.coo_ppr.row, rows])
            self.coo_ppr.col = np.concatenate([self.coo_ppr.col, ppr_scores.col])
            self.coo_ppr.data = np.concatenate([self.coo_ppr.data, ppr_scores.data])

            self.csr_ppr = self.coo_ppr.tocsr()

            # make sure cached rows stays unique
            self.cached_csr_rows = np.union1d(self.cached_csr_rows, rows)

    def _get_uncached(self, row):

        if isinstance(row, INT_TYPES):
            rows = np.array([row])
        else:
            rows = np.array(row)

        uncached_csr_rows = np.setdiff1d(np.unique(rows), self.cached_csr_rows, assume_unique=True)

        return uncached_csr_rows

    def __getitem__(self, key):
        if self.has_missing_ppr_values:
            row, col = self.csr_ppr._validate_indices(key)
            uncached_csr_rows = self._get_uncached(row)
            self._calc_ppr(uncached_csr_rows)
        return self.csr_ppr[key]
