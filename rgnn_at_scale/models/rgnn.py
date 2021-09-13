from typing import Any,  Dict,  Union


import torch

from torch_sparse import SparseTensor

from rgnn_at_scale.aggregation import ROBUST_MEANS, chunked_message_and_aggregate
from rgnn_at_scale.models.gcn import ChainableGCNConv
from rgnn_at_scale.models.gcn import GCN


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
