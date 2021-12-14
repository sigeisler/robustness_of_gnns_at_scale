from typing import Union, Optional


import torch
import numpy as np

from torch_geometric.data import Data
from torch_sparse import SparseTensor

from rgnn_at_scale.models.gcn import GCN
from rgnn_at_scale.models.deeprobust_rgcn import RGCN


class RGCN(RGCN):
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
        if isinstance(adj, SparseTensor):
            adj = adj.to_torch_sparse_coo_tensor()
        if adj.is_sparse:
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
