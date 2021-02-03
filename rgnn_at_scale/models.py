"""The models: GCN, GDC SVG GCN, Jaccard GCN, ...
"""

import collections
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import logging
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_sparse import coalesce, SparseTensor
from tqdm.auto import tqdm

from rgnn_at_scale.aggregation import ROBUST_MEANS, chunked_message_and_aggregate
from rgnn_at_scale import r_gcn
from rgnn_at_scale.utils import (get_approx_topk_ppr_matrix, get_ppr_matrix, get_truncated_svd, get_jaccard,
                                 sparse_tensor_to_tuple, tuple_to_sparse_tensor)
from rgnn_at_scale.data import RobustPPRDataset
from pprgo.pprgo import RobustPPRGo, PPRGo
from pprgo import ppr
from pprgo import utils as ppr_utils


class ChainableGCNConv(GCNConv):
    """Simple extension to allow the use of `nn.Sequential` with `GCNConv`. The arguments are wrapped as a Tuple/List
    are are expanded for Pytorch Geometric.

    Parameters
    ----------
    See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.gcn
    """

    def __init__(self, do_chunk: bool = False, n_chunks: int = 8, *input, **kwargs):
        super().__init__(*input, **kwargs)
        self.do_chunk = do_chunk
        self.n_chunks = n_chunks

    def forward(self, arguments: Sequence[torch.Tensor] = None) -> torch.Tensor:
        """Predictions based on the input.

        Parameters
        ----------
        arguments : Sequence[torch.Tensor]
            [x, edge indices] or [x, edge indices, edge weights], by default None

        Returns
        -------
        torch.Tensor
            the output of `GCNConv`.

        Raises
        ------
        NotImplementedError
            if the arguments are not of length 2 or 3
        """
        if len(arguments) == 2:
            x, edge_index = arguments
            edge_weight = None
        elif len(arguments) == 3:
            x, edge_index, edge_weight = arguments
        else:
            raise NotImplementedError("This method is just implemented for two or three arguments")
        embedding = super(ChainableGCNConv, self).forward(x, edge_index, edge_weight=edge_weight)
        if int(torch_geometric.__version__.split('.')[1]) < 6:
            embedding = super(ChainableGCNConv, self).update(embedding)
        return embedding

    # TODO: Add docstring
    def message_and_aggregate(self, adj_t: Union[torch.Tensor, SparseTensor], x: torch.Tensor) -> torch.Tensor:
        if not self.do_chunk or not isinstance(adj_t, SparseTensor):
            return super(ChainableGCNConv, self).message_and_aggregate(adj_t, x)
        else:
            return chunked_message_and_aggregate(adj_t, x, n_chunks=self.n_chunks)


ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "Identiy": nn.Identity()
}


class GCN(nn.Module):
    """Two layer GCN implemntation to be extended by the RGNN which supports the adjacency preprocessings:
    - SVD: Negin Entezari, Saba A. Al-Sayouri, Amirali Darvishzadeh, and Evangelos E. Papalexakis. All you need is Low
    (rank):  Defending against adversarial attacks on graphs.
    - GDC: Johannes Klicpera, Stefan Weißenberger, and Stephan Günnemann. Diffusion Improves Graph Learning.
    - Jaccard: Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu, and Liming Zhu.  Adversarial examples
    for graph data: Deep insights into attack and defense.

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    activation : nn.Module, optional
        Arbitrary activation function for the hidden layer, by default nn.ReLU()
    n_filters : int, optional
        number of dimensions for the hidden units, by default 64
    bias (bool, optional): If set to :obj:`False`, the gcn layers will not learn
            an additive bias. (default: :obj:`True`)
    dropout : int, optional
        Dropout rate, by default 0.5
    do_omit_softmax : bool, optional
        If you wanto omit the softmax of the output logits (for efficency), by default False
    gdc_params : Dict[str, float], optional
        Parameters for the GCN preprocessing (`alpha`, `k`, `use_cpu`), by default None
    svd_params : Dict[str, float], optional
        Parameters for the SVD preprocessing (`rank`), by default None
    jaccard_params : Dict[str, float], optional
        Parameters for the Jaccard preprocessing (`threshold`), by default None
    do_cache_adj_prep : bool, optional
        If `True` the preoprocessing of the adjacency matrix is chached for training, by default False
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 activation: Union[str, nn.Module] = nn.ReLU(),
                 n_filters: Union[int, Sequence[int]] = 64,
                 bias: bool = True,
                 dropout: int = 0.5,
                 do_omit_softmax: bool = False,
                 with_batch_norm: bool = False,
                 gdc_params: Optional[Dict[str, float]] = None,
                 svd_params: Optional[Dict[str, float]] = None,
                 jaccard_params: Optional[Dict[str, float]] = None,
                 do_cache_adj_prep: bool = True,
                 do_normalize_adj_once: bool = True,
                 add_self_loops: bool = True,
                 do_use_sparse_tensor: bool = True,
                 do_checkpoint: bool = False,  # TODO: Doc string
                 n_chunks: int = 8,
                 **kwargs):
        super().__init__()
        if not isinstance(n_filters, collections.Sequence):
            self.n_filters = [n_filters]
        else:
            self.n_filters = list(n_filters)
        if isinstance(activation, str):
            if activation in ACTIVATIONS.keys():
                self.activation = ACTIVATIONS[activation]
            else:
                raise AttributeError(f"Activation {activation} is not defined.")
        else:
            self.activation = activation

        self.n_features = n_features
        self.bias = bias
        self.n_classes = n_classes
        self.dropout = dropout
        self.do_omit_softmax = do_omit_softmax
        self.with_batch_norm = with_batch_norm
        self.gdc_params = gdc_params
        self.svd_params = svd_params
        self.jaccard_params = jaccard_params
        self.do_cache_adj_prep = do_cache_adj_prep
        self.do_normalize_adj_once = do_normalize_adj_once
        self.add_self_loops = add_self_loops
        self.do_use_sparse_tensor = do_use_sparse_tensor
        self.do_checkpoint = do_checkpoint
        self.n_chunks = n_chunks
        self.adj_preped = None
        self.layers = self._build_layers()

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return ChainableGCNConv(in_channels=in_channels, out_channels=out_channels,
                                do_chunk=self.do_checkpoint, n_chunks=self.n_chunks, bias=self.bias)

    def _build_layers(self):
        modules = nn.ModuleList([
            nn.Sequential(collections.OrderedDict(
                [(f'gcn_{idx}', self._build_conv_layer(in_channels=in_channels, out_channels=out_channels))]
                + ([(f'bn_{idx}', torch.nn.BatchNorm1d(out_channels))] if self.with_batch_norm else [])
                + [(f'activation_{idx}', self.activation),
                   (f'dropout_{idx}', nn.Dropout(p=self.dropout))]
            ))
            for idx, (in_channels, out_channels)
            in enumerate(zip([self.n_features] + self.n_filters[:-1], self.n_filters))
        ])
        idx = len(modules)
        modules.append(nn.Sequential(collections.OrderedDict([
            (f'gcn_{idx}', self._build_conv_layer(in_channels=self.n_filters[-1], out_channels=self.n_classes)),
            (f'softmax_{idx}', nn.Identity() if self.do_omit_softmax else nn.LogSoftmax(dim=1))
        ])))
        return modules

    def forward(self,
                data: Optional[Union[Data, torch.Tensor]] = None,
                adj: Optional[Union[torch.sparse.FloatTensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                attr_idx: Optional[torch.Tensor] = None,
                edge_idx: Optional[torch.Tensor] = None,
                n: Optional[int] = None,
                d: Optional[int] = None) -> torch.Tensor:
        x, edge_idx, edge_weight = GCN.parse_forward_input(data, adj, attr_idx, edge_idx, n, d)

        # Perform preprocessing such as SVD, GDC or Jaccard
        edge_idx, edge_weight = self._cache_if_option_is_set(self._preprocess_adjacency_matrix,
                                                             x, edge_idx, edge_weight)

        # Enforce that the input is contiguous
        x, edge_idx, edge_weight = self._ensure_contiguousness(x, edge_idx, edge_weight)

        for layer in self.layers:
            x = layer((x, edge_idx, edge_weight))

        return x

    @ staticmethod
    def parse_forward_input(data: Optional[Union[Data, torch.Tensor]] = None,
                            adj: Optional[Union[SparseTensor, torch.sparse.FloatTensor,
                                                Tuple[torch.Tensor, torch.Tensor]]] = None,
                            attr_idx: Optional[torch.Tensor] = None,
                            edge_idx: Optional[torch.Tensor] = None,
                            edge_weight: Optional[torch.Tensor] = None,
                            n: Optional[int] = None,
                            d: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_weight = None
        # PyTorch Geometric support
        if isinstance(data, Data):
            x, edge_idx = data.x, data.edge_index
        # Randomized smoothing support
        elif attr_idx is not None and edge_idx is not None and n is not None and d is not None:
            x = coalesce(attr_idx, torch.ones_like(attr_idx[0], dtype=torch.float32), m=n, n=d)
            x = torch.sparse.FloatTensor(x[0], x[1], torch.Size([n, d])).to_dense()
            edge_idx = edge_idx
        # Empirical robustness support
        elif isinstance(adj, tuple):
            # Necessary since `torch.sparse.FloatTensor` eliminates the gradient...
            x, edge_idx, edge_weight = data, adj[0], adj[1]
        elif isinstance(adj, SparseTensor):
            x = data
            edge_idx_rows, edge_idx_cols, edge_weight = adj.coo()
            edge_idx = torch.stack([edge_idx_rows, edge_idx_cols], dim=0)
        else:
            x, edge_idx, edge_weight = data, adj._indices(), adj._values()
        return x, edge_idx, edge_weight

    def release_cache(self):
        self.adj_preped = None

    def _ensure_contiguousness(self,
                               x: torch.Tensor,
                               edge_idx: Union[torch.Tensor, SparseTensor],
                               edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

    def _preprocess_adjacency_matrix(self,
                                     x: torch.Tensor,
                                     edge_idx: torch.Tensor,
                                     edge_weight: Optional[torch.Tensor] = None
                                     ) -> Tuple[Union[torch.Tensor, SparseTensor], Optional[torch.Tensor]]:
        if self.gdc_params is not None:
            if 'use_cpu' in self.gdc_params and self.gdc_params['use_cpu']:
                edge_idx, edge_weight = get_approx_topk_ppr_matrix(
                    edge_idx,
                    x.shape[0],
                    **self.gdc_params
                )
            else:
                adj = get_ppr_matrix(
                    torch.sparse.FloatTensor(edge_idx, torch.ones_like(edge_idx[0], dtype=torch.float32)),
                    **self.gdc_params,
                    normalize_adjacency_matrix=True
                )
                edge_idx, edge_weight = adj.indices(), adj.values()
                del adj
        elif self.svd_params is not None:
            adj = get_truncated_svd(
                torch.sparse.FloatTensor(
                    edge_idx,
                    torch.ones_like(edge_idx[0], dtype=torch.float32)
                ),
                **self.svd_params
            )
            self._deactivate_normalization()
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj
        elif self.jaccard_params is not None:
            adj = get_jaccard(
                torch.sparse.FloatTensor(
                    edge_idx,
                    torch.ones_like(edge_idx[0], dtype=torch.float32)
                ),
                x,
                **self.jaccard_params
            ).coalesce()
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj
        if self.do_checkpoint and (x.requires_grad or edge_weight.requires_grad):
            if not self.do_use_sparse_tensor:
                raise NotImplementedError('Checkpointing is only implemented in combination with sparse tensor input')
            # Currently (1.6.0) PyTorch does not support return arguments of `checkpoint` that do not require gradient.
            # For this reason we need to execute the code twice (due to checkpointing in fact three times...)
            adj = [checkpoint(
                lambda edge_weight: sparse_tensor_to_tuple(self._convert_and_normalize(x, edge_idx, edge_weight)[0])[0],
                edge_weight
            )]
            with torch.no_grad():
                adj.extend(sparse_tensor_to_tuple(self._convert_and_normalize(x, edge_idx, edge_weight)[0])[1:])
            return tuple_to_sparse_tensor(*adj), None
        else:
            return self._convert_and_normalize(x, edge_idx, edge_weight)

    def _cache_if_option_is_set(self,
                                callable: Callable[[Any], Any],
                                *inputs) -> Any:
        if self.training and self.adj_preped is not None:
            return self.adj_preped
        else:
            adj_preped = callable(*inputs)

        if (
            self.training
            and self.do_cache_adj_prep
            and (self.gdc_params is not None or self.svd_params is not None or self.jaccard_params is not None
                 or self.do_normalize_adj_once or self.do_use_sparse_tensor)
        ):
            self.adj_preped = adj_preped

        return adj_preped

    def _convert_and_normalize(self,
                               x: torch.Tensor,
                               edge_idx: torch.Tensor,
                               edge_weight: Optional[torch.Tensor] = None
                               ) -> Tuple[Union[torch.Tensor, SparseTensor], Optional[torch.Tensor]]:
        if self.do_normalize_adj_once:
            self._deactivate_normalization()

            num_nodes = x.shape[0]
            if edge_weight is None:
                edge_weight = torch.ones((edge_idx.size(1), ), dtype=torch.float32,
                                         device=edge_idx.device)

            if self.add_self_loops:
                edge_idx, tmp_edge_weight = add_remaining_self_loops(
                    edge_idx, edge_weight, 1., num_nodes)
                assert tmp_edge_weight is not None
                edge_weight = tmp_edge_weight

            row, col = edge_idx
            deg = scatter_add(edge_weight, col, dim=0, dim_size=x.shape[0])
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        if self.do_use_sparse_tensor:
            if hasattr(SparseTensor, 'from_edge_index'):
                adj = SparseTensor.from_edge_index(edge_idx, edge_weight, sparse_sizes=2 * x.shape[:1])
            else:
                adj = SparseTensor(row=edge_idx[0], col=edge_idx[1], value=edge_weight, sparse_sizes=2 * x.shape[:1])
            edge_idx = adj
            edge_weight = None
        return edge_idx, edge_weight

    def _deactivate_normalization(self):
        for layer in self.layers:
            layer[0].normalize = False


class RGNNConv(ChainableGCNConv):
    """Extension of Pytorch Geometric's `GCNConv` to execute a robust aggregation function:
    - soft_k_medoid
    - soft_medoid (not scalable)
    - k_medoid
    - medoid (not scalable)
    - dimmedian

    Parameters
    ----------
    mean : str, optional
        The desired mean (see above for the options), by default 'soft_k_medoid'
    mean_kwargs : Dict[str, Any], optional
        Arguments for the mean, by default dict(k=64, temperature=1.0, with_weight_correction=True)
    """

    def __init__(self, mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=64, temperature=1.0, with_weight_correction=True),
                 **kwargs):
        super().__init__(**kwargs)
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs

    def message_and_aggregate(self, adj_t) -> torch.Tensor:
        raise NotImplementedError

    def propagate(self, edge_index: Union[torch.Tensor, SparseTensor], size=None, **kwargs) -> torch.Tensor:
        x = kwargs['x']
        if not isinstance(edge_index, SparseTensor):
            edge_weights = kwargs['norm'] if 'norm' in kwargs else kwargs['edge_weight']
            A = SparseTensor.from_edge_index(edge_index, edge_weights, (x.size(0), x.size(0)))
            return self._mean(A, x, **self._mean_kwargs)

        def aggregate(edge_index: SparseTensor, x: torch.Tensor):
            return self._mean(edge_index, x, **self._mean_kwargs)
        if self.do_chunk:
            return chunked_message_and_aggregate(edge_index, x, n_chunks=self.n_chunks, aggregation_function=aggregate)
        else:
            return aggregate(edge_index, x)


class RGNN(GCN):
    """Generic Reliable Graph Neural Network (RGNN) implementation which currently supports a GCN architecture with the
    aggregation functions:
    - soft_k_medoid
    - soft_medoid (not scalable)
    - k_medoid
    - medoid (not scalable)
    - dimmedian

    and with the adjacency preprocessings:
    - SVD: Negin Entezari, Saba A. Al-Sayouri, Amirali Darvishzadeh, and Evangelos E. Papalexakis. All you need is Low
    (rank):  Defending against adversarial attacks on graphs.
    - GDC: Johannes Klicpera, Stefan Weißenberger, and Stephan Günnemann. Diffusion Improves Graph Learning.
    - Jaccard: Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu, and Liming Zhu.  Adversarial examples
    for graph data: Deep insights into attack and defense.

    Parameters
    ----------
    mean : str, optional
        The desired mean (see above for the options), by default 'soft_k_medoid'
    mean_kwargs : Dict[str, Any], optional
        Arguments for the mean, by default dict(k=64, temperature=1.0, with_weight_correction=True)
    """

    def __init__(self,
                 mean: str = 'soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=64, temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        self._mean_kwargs = dict(mean_kwargs)
        self._mean = mean
        super().__init__(**kwargs)

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return RGNNConv(mean=self._mean, mean_kwargs=self._mean_kwargs, in_channels=in_channels,
                        out_channels=out_channels, do_chunk=self.do_checkpoint, n_chunks=self.n_chunks)


class DenseGraphConvolution(nn.Module):
    """Dense GCN convolution layer for the FGSM attack that requires a gradient towards the adjacency matrix.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels of the input
        out_channels : int
            Desired number of channels for the output (for trainable linear transform)
        """
        super().__init__()
        self._linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, arguments: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Prediction based on input.

        Parameters
        ----------
        arguments : Tuple[torch.Tensor, torch.Tensor]
            Tuple with two elements of the attributes and dense adjacency matrix

        Returns
        -------
        torch.Tensor
            The new embeddings
        """
        x, adj_matrix = arguments

        x_trans = self._linear(x)
        return adj_matrix @ x_trans


class DenseGCN(nn.Module):
    """Dense two layer GCN for the FGSM attack that requires a gradient towards the adjacency matrix.
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_filters: int = 64,
                 activation: nn.Module = nn.ReLU(),
                 dropout: int = 0.5,
                 ** kwargs):
        """
        Parameters
        ----------
        n_features : int
            Number of attributes for each node
        n_classes : int
            Number of classes for prediction
        n_filters : int, optional
            number of dimensions for the hidden units, by default 80
        activation : nn.Module, optional
            Arbitrary activation function for the hidden layer, by default nn.ReLU()
        dropout : int, optional
            Dropout rate, by default 0.5
        """
        super().__init__()
        self.n_features = n_features
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = dropout
        self.layers = nn.ModuleList([
            nn.Sequential(collections.OrderedDict([
                ('gcn_0', DenseGraphConvolution(in_channels=n_features,
                                                out_channels=n_filters)),
                ('activation_0', self.activation),
                ('dropout_0', nn.Dropout(p=dropout))
            ])),
            nn.Sequential(collections.OrderedDict([
                ('gcn_1', DenseGraphConvolution(in_channels=n_filters,
                                                out_channels=n_classes)),
                ('softmax_1', nn.LogSoftmax(dim=1))
            ]))
        ])

    @ staticmethod
    def normalize_dense_adjacency_matrix(adj: torch.Tensor) -> torch.Tensor:
        """Normalizes the adjacency matrix as proposed for a GCN by Kipf et al. Moreover, it only uses the upper triangular
        matrix of the input to obtain the right gradient towards the undirected adjacency matrix.

        Parameters
        ----------
        adj: torch.Tensor
            The weighted undirected [n x n] adjacency matrix.

        Returns
        -------
        torch.Tensor
            Normalized [n x n] adjacency matrix.
        """
        adj_norm = torch.triu(adj, diagonal=1) + torch.triu(adj, diagonal=1).T
        adj_norm.data[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 1
        deg = torch.diag(torch.pow(adj_norm.sum(axis=1), - 1 / 2))
        adj_norm = deg @ adj_norm @ deg
        return adj_norm

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Prediction based on input.

        Parameters
        ----------
        x : torch.Tensor
            Dense [n, d] tensor holding the attributes
        adjacency_matrix : torch.Tensor
            Dense [n, n] tensor for the adjacency matrix

        Returns
        -------
        torch.Tensor
            The predictions (after applying the softmax)
        """
        adjacency_matrix = DenseGCN.normalize_dense_adjacency_matrix(adjacency_matrix)
        for layer in self.layers:
            x = layer((x, adjacency_matrix))
        return x


class RGCN(r_gcn.RGCN):
    """Wrapper around the RGCN implementation of https: // github.com / DSE - MSU / DeepRobust
    """

    def __init__(self, n_classes: int, n_features: int, n_filters: int = 64, **kwargs):
        super().__init__(nfeat=n_features, nhid=n_filters, nclass=n_classes)

    def forward(self,
                data: Optional[Union[Data, torch.Tensor]] = None,
                adj: Optional[torch.Tensor] = None,
                attr_idx: Optional[torch.Tensor] = None,
                edge_idx: Optional[torch.Tensor] = None,
                n: Optional[int] = None,
                d: Optional[int] = None):
        x, edge_idx, _ = GCN.parse_forward_input(data, adj, attr_idx, edge_idx, n, d)
        self.device = x.device

        if adj is None:
            n = x.shape[0]
            adj = torch.sparse.FloatTensor(
                edge_idx,
                torch.ones_like(edge_idx[0], dtype=torch.float32),
                torch.Size([n, n])
            )
        adj = adj.to_dense()

        self.features = x
        self.adj_norm1 = self._normalize_adj(adj, power=-1 / 2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)

        return super()._forward()

    def fit(self,
            adj: Union[SparseTensor, torch.sparse.FloatTensor],
            attr: torch.Tensor,
            labels: torch.Tensor,
            idx_train: np.ndarray,
            idx_val: np.ndarray,
            max_epochs: int = 200,
            **kwargs):

        if isinstance(adj, SparseTensor):
            self.device = adj.device()
        else:  # torch.sparse.FloatTensor
            self.device = adj.device

        super().fit(
            features=attr,
            adj=adj.to_dense(),
            labels=labels,
            idx_train=idx_train,
            idx_val=idx_val,
            train_iters=max_epochs)


class PPRGoWrapperBase():
    def model_forward(self, *args, **kwargs):
        pass

    def forward_wrapper(self,
                        attr: torch.Tensor,
                        adj: Union[SparseTensor, sp.csr_matrix],
                        ppr_scores: SparseTensor = None,
                        ppr_idx=None):

        device = next(self.parameters()).device
        if ppr_scores is not None:

            source_idx, neighbor_idx, ppr_vals = ppr_scores.coo()
            ppr_matrix = ppr_scores[:, neighbor_idx.unique()]
            attr_matrix = attr[neighbor_idx.unique()]

            return self.model_forward(attr_matrix.to(device), ppr_matrix.to(device))
        else:
            # we need to precompute the ppr_score first

            # TODO: Calculate topk ppr with pytorch so autograd can backprop through adjacency

            if isinstance(adj, SparseTensor):
                num_nodes = adj.size(0)
                adj = adj.to_scipy(layout="csr")
            else:
                # for scipy sparse matrix
                num_nodes = adj.size

            if ppr_idx is None:
                ppr_idx = np.arange(num_nodes)

            topk_ppr = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, ppr_idx,
                                           self.topk,  normalization=self.ppr_normalization)

            # there are to many node for a single forward pass, we need to do batched prediction
            data_set = RobustPPRDataset(
                attr_matrix_all=attr,
                ppr_matrix=topk_ppr,
                indices=ppr_idx,
                allow_cache=False)
            data_loader = torch.utils.data.DataLoader(
                dataset=data_set,
                sampler=torch.utils.data.BatchSampler(
                    torch.utils.data.SequentialSampler(data_set),
                    batch_size=self.forward_batch_size, drop_last=False
                ),
                batch_size=None,
                num_workers=0,
            )
            num_predictions = topk_ppr.shape[0]

            logits = torch.zeros(num_predictions, self.n_classes, device="cpu", dtype=torch.float32)

            num_batches = len(data_loader)
            for batch_id, (idx, xbs, _) in enumerate(data_loader):

                logging.info(f"inference batch {batch_id}/{num_batches}")
                logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))
                if device.type == "cuda":
                    logging.info(torch.cuda.max_memory_allocated() / (1024 ** 3))

                xbs = [xb.to(device) for xb in xbs]
                start = batch_id * self.forward_batch_size
                end = start + xbs[1].size(0)  # batch_id * batch_size
                logits[start:end] = self.model_forward(*xbs).cpu()

            return logits

    def fit(self,
            adj: Union[SparseTensor, sp.csr_matrix],
            attr: torch.Tensor,
            labels: torch.Tensor,
            idx_train: np.ndarray,
            idx_val: np.ndarray,
            lr,
            weight_decay: int,
            patience: int,
            max_epochs: int = 200,
            batch_size=512,
            batch_mult_val=4,
            eval_step=1,
            display_step=50,
            **kwargs):
        device = next(self.parameters()).device

        logging.info("fit start")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        if isinstance(adj, SparseTensor):
            adj = adj.to_scipy(layout="csr")

        topk_train = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_train,
                                         self.topk,  normalization=self.ppr_normalization)

        topk_val = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_val,
                                       self.topk,  normalization=self.ppr_normalization)

        train_set = RobustPPRDataset(attr_matrix_all=attr,
                                     ppr_matrix=topk_train,
                                     indices=idx_train,
                                     labels_all=labels,
                                     allow_cache=False)
        val_set = RobustPPRDataset(attr_matrix_all=attr,
                                   ppr_matrix=topk_val,
                                   indices=idx_val,
                                   labels_all=labels,
                                   allow_cache=False)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(train_set),
                batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            num_workers=0,
        )
        trace_train_loss = []
        trace_val_loss = []
        trace_train_acc = []
        trace_val_acc = []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = np.inf

        step = 0
        epoch_pbar = tqdm(range(max_epochs), desc='Training Epoch...')
        for it in epoch_pbar:
            batch_pbar = tqdm(train_loader, desc="Training Batch...")
            for batch_train_idx, xbs, yb in batch_pbar:
                xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

                logging.info("Train batch")
                logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))
                if device.type == "cuda":
                    logging.info(torch.cuda.max_memory_allocated() / (1024 ** 3))

                loss_train, ncorrect_train = self.__run_batch(xbs, yb, optimizer, train=True)

                train_acc = ncorrect_train / float(yb.shape[0])

                # validation on batch of val_set
                val_batch_size = batch_mult_val * batch_size
                rnd_idx = np.random.choice(len(val_set), size=len(val_set), replace=False)[:val_batch_size]
                batch_val_idx, xbs, yb = val_set[rnd_idx]
                xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

                loss_val, ncorrect_val = self.__run_batch(xbs, yb, None, train=False)
                val_acc = ncorrect_val / float(yb.shape[0])

                trace_train_loss.append(loss_train)
                trace_val_loss.append(loss_val)
                trace_train_acc.append(train_acc)
                trace_val_acc.append(val_acc)

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = it
                    best_state = {key: value.cpu() for key, value in self.state_dict().items()}
                else:
                    if it >= best_epoch + patience:
                        break

                batch_pbar.set_description(f"Epoch: {it:}, loss_train: {loss_train: .5f}, loss_val: {loss_val: .5f}",
                                           refresh=False)

                step += 1

            epoch_pbar.set_description(f"Training Epoch... acc_train: {train_acc: .4f}, acc_val: {val_acc: .4f}",
                                       refresh=False)

            # restore the best validation state
        self.load_state_dict(best_state)
        return {"loss": trace_train_loss, "acc": trace_train_acc}, {"loss": trace_val_loss, "acc": trace_val_acc},

    def __run_batch(self, xbs, yb, optimizer, train):
        # Set model to training mode
        if train:
            self.train()
        else:
            self.eval()

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            logits = self.model_forward(*xbs)
            loss = F.cross_entropy(logits, yb)
            top1 = torch.argmax(logits, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item(), ncorrect.detach().cpu().item()


class PPRGoWrapper(PPRGo, PPRGoWrapperBase):
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 alpha,
                 eps,
                 topk,
                 ppr_normalization,
                 forward_batch_size=128,
                 **kwargs):
        super().__init__(n_features, n_classes, hidden_size, nlayers, dropout, **kwargs)
        self.n_classes = n_classes
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.ppr_normalization = ppr_normalization
        self.forward_batch_size = forward_batch_size

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self,
                      attr: torch.Tensor,
                      ppr_matrix: SparseTensor,
                      **kwargs):
        source_idx, neighbor_idx, ppr_scores = ppr_matrix.coo()
        attr = attr[neighbor_idx]
        return super().forward(attr, ppr_scores, source_idx)


class RobustPPRGoWrapper(RobustPPRGo, PPRGoWrapperBase):
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 alpha,
                 eps,
                 topk,
                 ppr_normalization,
                 forward_batch_size=128,
                 mean='soft_k_medoid',
                 **kwargs):
        super().__init__(n_features, n_classes, hidden_size, nlayers, dropout, mean=mean,  **kwargs)
        self.n_classes = n_classes
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.ppr_normalization = ppr_normalization
        self.forward_batch_size = forward_batch_size

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


MODEL_TYPE = Union[GCN, RGNN, RGCN, RobustPPRGoWrapper, PPRGoWrapper]
BATCHED_PPR_MODELS = Union[RobustPPRGoWrapper, PPRGoWrapper]


def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Containing the hyperparameters.

    Returns
    -------
    Union[GCN, RGNN]
        The created instance.
    """
    if 'model' not in hyperparams or hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == 'RGCN':
        return RGCN(**hyperparams)
    elif hyperparams['model'] == "RobustPPRGo":
        return RobustPPRGoWrapper(**hyperparams)
    elif hyperparams['model'] == "PPRGo":
        return PPRGoWrapper(**hyperparams)
    else:
        return RGNN(**hyperparams)
