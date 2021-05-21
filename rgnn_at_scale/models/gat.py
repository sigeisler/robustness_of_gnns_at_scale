from typing import Any, Dict, Tuple

import torch

from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

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
        kwargs['in_channels'] = 2*[kwargs['in_channels']]
        super().__init__(**kwargs)
        self._mean = ROBUST_MEANS[mean] if mean is not None else None
        self._mean_kwargs = mean_kwargs

        #assert isinstance(kwargs['in_channels'], int), 'Only identical encoders for left and right are supported'

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
        _, attention_matrix = super().forward(x, edge_index, return_attention_weights=True)
        attention_matrix.storage._value = attention_matrix.storage._value.squeeze()

        #assert (attention_matrix.storage._row == attention_matrix.storage._row).all()
        #assert (attention_matrix.storage._col == attention_matrix.storage._col).all()

        #attention_matrix.storage._value *= edge_index.storage._value

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
