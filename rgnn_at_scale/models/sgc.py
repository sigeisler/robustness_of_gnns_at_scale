
import collections
from typing import Optional, Tuple, Union, Callable
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


import torch
from torch import Tensor
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_sparse import coalesce, SparseTensor

import torch_geometric
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm


from rgnn_at_scale.aggregation import chunked_message_and_aggregate
from rgnn_at_scale.helper.utils import sparse_tensor_to_tuple, tuple_to_sparse_tensor
from rgnn_at_scale.models.gcn import GCN

patch_typeguard()


class SGConv(torch_geometric.nn.SGConv):

    def __init__(self, dropout=0.5, **kwargs):
        super(SGConv, self).__init__(**kwargs)
        self.normalize = True
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            if self.normalize:
                if isinstance(edge_index, Tensor):
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                elif isinstance(edge_index, SparseTensor):
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache

        return self.lin(self.dropout(x))


@typechecked
class ChainableSGConv(SGConv):
    """Simple extension to allow the use of `nn.Sequential` with `SGConv`. The arguments are wrapped as a Tuple/List
    are are expanded for Pytorch Geometric.

    Parameters
    ----------
    See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv
    """

    def __init__(self, do_chunk: bool = False, n_chunks: int = 8, *input, **kwargs):
        super().__init__(*input, **kwargs)
        self.do_chunk = do_chunk
        self.n_chunks = n_chunks

    def forward(self, arguments: Tuple[TensorType["n_nodes", "n_features"],
                                       Union[TensorType[2, "nnz"], SparseTensor],
                                       Optional[TensorType["nnz"]]]) -> TensorType["n_nodes", "n_classes"]:
        """Predictions based on the input.

        Parameters
        ----------
        arguments : Sequence[torch.Tensor]
            [x, edge indices] or [x, edge indices, edge weights], by default None

        Returns
        -------
        torch.Tensor
            the output of `SGConv`.

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

        if self.training:
            self.cached = True
        else:
            self.cached = False
            self._cached_x = None

        embedding = super(ChainableSGConv, self).forward(x, edge_index, edge_weight=edge_weight)
        if int(torch_geometric.__version__.split('.')[1]) < 6:
            embedding = super(ChainableSGConv, self).update(embedding)
        return embedding

    def message_and_aggregate(self, adj_t: Union[torch.Tensor, SparseTensor], x: torch.Tensor) -> torch.Tensor:
        if not self.do_chunk or not isinstance(adj_t, SparseTensor):
            return super(ChainableSGConv, self).message_and_aggregate(adj_t, x)
        else:
            return chunked_message_and_aggregate(adj_t, x, n_chunks=self.n_chunks)


ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "ELU": nn.ELU(),
    "Identiy": nn.Identity()
}


@typechecked
class SGC(nn.Module):
    """
    Implementation of Simplifying Graph Convolutional Networks (SGC).
    `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
    Pytorch implementation: <https://github.com/Tiiiger/SGC>

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    activation : nn.Module, optional
        Arbitrary activation function for the hidden layer, by default nn.ReLU()
    K : int, optional
        Number of hops
    bias (bool, optional): If set to :obj:`False`, the gcn layers will not learn
            an additive bias. (default: :obj:`True`)
    dropout : int, optional
        Dropout rate, by default 0.5
    do_cache_adj_prep : bool, optional
        If `True` the preoprocessing of the adjacency matrix is chached for training, by default True
    do_normalize_adj_once : bool, optional
        If true the adjacency matrix is normalized only once, by default True
    do_use_sparse_tensor : bool, optional
        If true use SparseTensor internally, by default True
    do_checkpoint : bool, optional
        If true use checkpointing in message passing, by default False
    n_chunks : int, optional
        Number of chunks for checkpointing, by default 8
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 K: int = 2,
                 bias: bool = True,
                 dropout: float = 0,
                 with_batch_norm: bool = False,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 do_cache_adj_prep: bool = True,
                 do_normalize_adj_once: bool = True,
                 do_use_sparse_tensor: bool = True,
                 do_checkpoint: bool = False,
                 n_chunks: int = 8,
                 **kwargs):
        super().__init__()

        assert K > 0, "K must be positiv"

        self.n_features = n_features
        self.n_classes = n_classes
        self.K = K
        self.bias = bias
        self.cached = cached
        self.dropout = dropout
        self.with_batch_norm = with_batch_norm
        self.add_self_loops = add_self_loops
        self.do_cache_adj_prep = do_cache_adj_prep
        self.do_normalize_adj_once = do_normalize_adj_once
        self.do_use_sparse_tensor = do_use_sparse_tensor
        self.do_checkpoint = do_checkpoint
        self.n_chunks = n_chunks
        self.adj_preped = None
        self.normalize = True
        self.layers = self._build_layers()

    def _build_conv_layer(self, in_channels: int, out_channels: int, K: int):
        return ChainableSGConv(in_channels=in_channels, out_channels=out_channels, K=K, cached=self.cached,
                               do_chunk=self.do_checkpoint, n_chunks=self.n_chunks, bias=self.bias,
                               dropout=self.dropout)

    def _build_layers(self):
        modules = nn.ModuleList([
            nn.Sequential(collections.OrderedDict(
                [
                    ('sgc', self._build_conv_layer(in_channels=self.n_features,
                                                   out_channels=self.n_classes, K=self.K))]
            ))
        ])

        return modules

    def forward(self,
                data: Optional[Union[Data, TensorType["n_nodes", "n_features"]]] = None,
                adj: Optional[Union[SparseTensor,
                                    torch.sparse.FloatTensor,
                                    Tuple[TensorType[2, "nnz"], TensorType["nnz"]]]] = None,
                attr_idx: Optional[TensorType["n_nodes", "n_features"]] = None,
                edge_idx: Optional[TensorType[2, "nnz"]] = None,
                edge_weight: Optional[TensorType["nnz"]] = None,
                n: Optional[int] = None,
                d: Optional[int] = None) -> TensorType["n_nodes", "n_classes"]:
        x, edge_idx, edge_weight = SGC.parse_forward_input(data, adj, attr_idx, edge_idx, edge_weight, n, d)

        # Perform preprocessing
        if self.normalize:
            edge_idx, edge_weight = self._cache_if_option_is_set(self._preprocess_adjacency_matrix,
                                                                 x, edge_idx, edge_weight)
        else:
            self._deactivate_normalization()

        # Enforce that the input is contiguous
        x, edge_idx, edge_weight = self._ensure_contiguousness(x, edge_idx, edge_weight)

        for layer in self.layers:
            x = layer((x, edge_idx, edge_weight))

        return x

    @ staticmethod
    def parse_forward_input(data: Optional[Union[Data, TensorType["n_nodes", "n_features"]]] = None,
                            adj: Optional[Union[SparseTensor,
                                                torch.sparse.FloatTensor,
                                                Tuple[TensorType[2, "nnz"], TensorType["nnz"]]]] = None,
                            attr_idx: Optional[TensorType["n_nodes", "n_features"]] = None,
                            edge_idx: Optional[TensorType[2, "nnz"]] = None,
                            edge_weight: Optional[TensorType["nnz"]] = None,
                            n: Optional[int] = None,
                            d: Optional[int] = None) -> Tuple[TensorType["n_nodes", "n_features"],
                                                              TensorType[2, "nnz"],
                                                              TensorType["nnz"]]:
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

        if edge_weight is None:
            edge_weight = torch.ones_like(edge_idx[0], dtype=torch.float32)

        if edge_weight.dtype != torch.float32:
            edge_weight = edge_weight.float()

        return x, edge_idx, edge_weight

    def release_cache(self):
        self.adj_preped = None

    def _ensure_contiguousness(self,
                               x: torch.Tensor,
                               edge_idx: Union[torch.Tensor, SparseTensor],
                               edge_weight: Optional[torch.Tensor]) -> Tuple[TensorType["n_nodes", "n_features"],
                                                                             Union[TensorType[2, "nnz"], SparseTensor],
                                                                             Optional[TensorType["nnz"]]]:

        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

    def _preprocess_adjacency_matrix(self,
                                     x: TensorType["n_nodes", "n_features"],
                                     edge_idx: TensorType[2, "nnz"],
                                     edge_weight: Optional[TensorType["nnz"]] = None
                                     ) -> Tuple[Union[TensorType[2, "nnz_after"], SparseTensor],
                                                Optional[TensorType["nnz_after"]]]:
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
                                callable: Callable,
                                *inputs):
        if self.training and self.adj_preped is not None:
            return self.adj_preped
        else:
            adj_preped = callable(*inputs)

        if (
            self.training
            and self.do_cache_adj_prep
            and (self.do_normalize_adj_once or self.do_use_sparse_tensor)
        ):
            self.adj_preped = adj_preped

        return adj_preped

    def _convert_and_normalize(self,
                               x: TensorType["n_nodes", "n_features"],
                               edge_idx: TensorType[2, "nnz"],
                               edge_weight: Optional[TensorType["nnz"]] = None,
                               ) -> Tuple[Union[TensorType[2, "nnz_after"], SparseTensor],
                                          Optional[TensorType["nnz_after"]]]:
        if self.do_normalize_adj_once:
            self._deactivate_normalization()
            n = x.shape[0]
            edge_idx, edge_weight = GCN.normalize(edge_idx, n, edge_weight, self.add_self_loops, row_norm=False)

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

    def deactivate_caching(self):
        for layer in self.layers:
            layer[0].cached = False
            layer[0]._cached_x = None
