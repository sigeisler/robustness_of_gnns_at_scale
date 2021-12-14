from typing import Any, Dict, Tuple

import torch

from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor, set_diag

from rgnn_at_scale.aggregation import ROBUST_MEANS
from rgnn_at_scale.models.gcn import GCN


class RGATConv(GATConv):
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
        kwargs['in_channels'] = 2 * [kwargs['in_channels']]
        super().__init__(**kwargs)
        self._mean = ROBUST_MEANS[mean] if mean is not None else None
        self._mean_kwargs = mean_kwargs

    def forward(self, arguments: Tuple[torch.Tensor, SparseTensor] = None) -> torch.Tensor:
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
        assert isinstance(edge_index, SparseTensor), 'GAT requires a SparseTensor as input'
        assert edge_weight is None, 'The weights must be passed via a SparseTensor'

        H, C = self.heads, self.out_channels

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_l = x_r = self.lin_l(x).view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        if self.add_self_loops:
            edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r))

        alpha = self._alpha * edge_index.storage.value()[:, None]
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        attention_matrix = edge_index.set_value(alpha, layout='coo')
        attention_matrix.storage._value = attention_matrix.storage._value.squeeze()

        x = self.lin_l(x)

        if self._mean is not None:
            x = self._mean(attention_matrix, x, **self._mean_kwargs)
        else:
            x = attention_matrix @ x
        x += self.bias
        return x


class RGAT(GCN):
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
                 mean_kwargs: Dict[str, Any] = dict(k=64, temperature=1.0, with_weight_correction=True),
                 **kwargs):
        self._mean_kwargs = dict(mean_kwargs)
        self._mean = mean
        super().__init__(**kwargs)

        assert not self.do_checkpoint, 'Checkpointing is not supported'

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return RGATConv(mean=self._mean, mean_kwargs=self._mean_kwargs,
                        in_channels=in_channels, out_channels=out_channels)

    def _cache_if_option_is_set(self, callback, x, edge_idx, edge_weight):
        return SparseTensor.from_edge_index(edge_idx, edge_weight, (x.shape[0], x.shape[0])), None
