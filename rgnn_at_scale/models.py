"""The models: GCN, GDC SVG GCN, Jaccard GCN, ...
"""

import collections
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_sparse import coalesce

from rgnn_at_scale.means import ROBUST_MEANS
from rgnn_at_scale import r_gcn
from rgnn_at_scale.utils import get_approx_topk_ppr_matrix, get_ppr_matrix, get_truncated_svd, get_jaccard


class ChainableGCNConv(GCNConv):
    """Simple extension to allow the use of `nn.Sequential` with `GCNConv`. The arguments are wrapped as a Tuple/List
    are are expanded for Pytorch Geometric.

    Parameters
    ----------
    See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.gcn
    """

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
        number of dimensions for the hidden units, by default 80
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
                 activation: nn.Module = nn.ReLU(),
                 n_filters: Union[int, Sequence[int]] = 64,
                 dropout: int = 0.5,
                 do_omit_softmax: bool = False,
                 with_batch_norm: bool = False,
                 gdc_params: Optional[Dict[str, float]] = None,
                 svd_params: Optional[Dict[str, float]] = None,
                 jaccard_params: Optional[Dict[str, float]] = None,
                 do_cache_adj_prep: bool = False,
                 do_normalize_adj_once: bool = False,
                 **kwargs):
        super().__init__()
        if not isinstance(n_filters, collections.Sequence):
            self.n_filters = [n_filters]
        else:
            self.n_filters = list(n_filters)
        self.n_features = n_features
        self.n_classes = n_classes
        self._activation = activation
        self._dropout = dropout
        self._do_omit_softmax = do_omit_softmax
        self._with_batch_norm = with_batch_norm
        self._gdc_params = gdc_params
        self._svd_params = svd_params
        self._jaccard_params = jaccard_params
        self._do_cache_adj_prep = do_cache_adj_prep
        self._do_normalize_adj_once = do_normalize_adj_once
        self._adj_preped = None
        self.layers = self._build_layers()

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return ChainableGCNConv(in_channels=in_channels, out_channels=out_channels)

    def _build_layers(self):
        modules = nn.ModuleList([
            nn.Sequential(collections.OrderedDict(
                [(f'gcn_{idx}', self._build_conv_layer(in_channels=in_channels, out_channels=out_channels))]
                + ([(f'bn_{idx}', torch.nn.BatchNorm1d(out_channels))] if self._with_batch_norm else [])
                + [(f'activation_{idx}', self._activation),
                   (f'dropout_{idx}', nn.Dropout(p=self._dropout))]
            ))
            for idx, (in_channels, out_channels)
            in enumerate(zip([self.n_features] + self.n_filters[:-1], self.n_filters))
        ])
        idx = len(modules)
        modules.append(nn.Sequential(collections.OrderedDict([
            (f'gcn_{idx}', self._build_conv_layer(in_channels=self.n_filters[-1], out_channels=self.n_classes)),
            (f'softmax_{idx}', nn.LogSoftmax(dim=1) if self._do_omit_softmax else nn.Identity())
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
        edge_idx, edge_weight = self._preprocess_adjacency_matrix(x, edge_idx, edge_weight)

        # Enforce that the input is contiguous
        x, edge_idx, edge_weight = self._ensure_contiguousness(x, edge_idx, edge_weight)

        for layer in self.layers:
            x = layer((x, edge_idx, edge_weight))

        return x

    @ staticmethod
    def parse_forward_input(data: Optional[Union[Data, torch.Tensor]] = None,
                            adj: Optional[Union[torch.sparse.FloatTensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
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
        else:
            x, edge_idx, edge_weight = data, adj._indices(), adj._values()
        return x, edge_idx, edge_weight

    def _ensure_contiguousness(self,
                               x: torch.Tensor,
                               edge_idx: torch.Tensor,
                               edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not x.is_sparse:
            x = x.contiguous()
        edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

    def _preprocess_adjacency_matrix(self,
                                     x: torch.Tensor,
                                     edge_idx: torch.Tensor,
                                     edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self._adj_preped is not None:
            return self._adj_preped

        if self._gdc_params is not None:
            if 'use_cpu' in self._gdc_params and self._gdc_params['use_cpu']:
                edge_idx, edge_weight = get_approx_topk_ppr_matrix(
                    edge_idx,
                    x.shape[0],
                    **self._gdc_params
                )
            else:
                adj = get_ppr_matrix(
                    torch.sparse.FloatTensor(edge_idx, torch.ones_like(edge_idx[0], dtype=torch.float32)),
                    **self._gdc_params,
                    normalize_adjacency_matrix=True
                )
                edge_idx, edge_weight = adj.indices(), adj.values()
                del adj
        elif self._svd_params is not None:
            adj = get_truncated_svd(
                torch.sparse.FloatTensor(
                    edge_idx,
                    torch.ones_like(edge_idx[0], dtype=torch.float32)
                ),
                **self._svd_params
            )
            self.deactivate_normalization()
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj
        elif self._jaccard_params is not None:
            adj = get_jaccard(
                torch.sparse.FloatTensor(
                    edge_idx,
                    torch.ones_like(edge_idx[0], dtype=torch.float32)
                ),
                x,
                **self._jaccard_params
            ).coalesce()
            edge_idx, edge_weight = adj.indices(), adj.values()
            del adj

        if (
            self.training
            and self._do_cache_adj_prep
            and (self._gdc_params is not None or self._svd_params is not None or self._jaccard_params is not None)
        ):
            self._adj_preped = (edge_idx, edge_weight)

        if self._do_normalize_adj_once:
            self.deactivate_normalization()

            row, col = edge_idx
            deg = scatter_add(edge_weight, col, dim=0, dim_size=x.shape[0])
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_idx, edge_weight

    def deactivate_normalization(self):
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
        return NotImplemented

    def propagate(self, edge_index: torch.Tensor, size=None, **kwargs) -> torch.Tensor:
        x = kwargs['x']
        edge_weights = kwargs['norm'] if 'norm' in kwargs else kwargs['edge_weight']
        A = torch.sparse.FloatTensor(edge_index, edge_weights).coalesce()
        return self._mean(A, x, **self._mean_kwargs)


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
    do_omit_softmax : bool, optional
        If you wanto omit the softmax of the output logits (for efficency), by default False
    """

    def __init__(self,
                 mean: str = 'soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=64, temperature=1.0,
                                                    with_weight_correction=True),
                 do_omit_softmax=False,
                 **kwargs):
        self._mean_kwargs = dict(mean_kwargs)
        self._mean = mean
        self._do_omit_softmax = do_omit_softmax
        super().__init__(**kwargs)

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return RGNNConv(mean=self._mean, mean_kwargs=self._mean_kwargs,
                        in_channels=in_channels, out_channels=out_channels)


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
        self._activation = activation
        self._dropout = dropout
        self.layers = nn.ModuleList([
            nn.Sequential(collections.OrderedDict([
                ('gcn_0', DenseGraphConvolution(in_channels=n_features,
                                                out_channels=n_filters)),
                ('activation_0', self._activation),
                ('dropout_0', nn.Dropout(p=dropout))
            ])),
            nn.Sequential(collections.OrderedDict([
                ('gcn_1', DenseGraphConvolution(in_channels=n_filters,
                                                out_channels=n_classes)),
                ('softmax_1', nn.LogSoftmax(dim=1))
            ]))
        ])

    @staticmethod
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
        x, edge_idx = GCN.parse_forward_input(data, adj, attr_idx, edge_idx, n, d)
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
            adj: torch.sparse.FloatTensor,
            attr: torch.Tensor,
            labels: torch.Tensor,
            idx_train: np.ndarray,
            idx_val: np.ndarray,
            max_epochs: int = 200,
            **kwargs):

        self.device = adj.device

        super().fit(
            features=attr,
            adj=adj.to_dense(),
            labels=labels,
            idx_train=idx_train,
            idx_val=idx_val,
            train_iters=max_epochs)


MODEL_TYPE = Union[GCN, RGNN, RGCN]


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
    else:
        return RGNN(**hyperparams)
