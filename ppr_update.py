
import logging
from pathlib import Path

import os.path as osp

import numpy as np
import math
import torch
import torch_sparse
import scipy.sparse as sp

from pprgo import utils as ppr_utils
from pprgo.pytorch_utils import matrix_to_torch
from pprgo import ppr


from ogb.nodeproppred import PygNodePropPredDataset
from rgnn_at_scale.local import setup_logging
from rgnn_at_scale import utils
from torch_geometric.utils import add_remaining_self_loops

setup_logging()

device = 0 if torch.cuda.is_available() else 'cpu'
A_dense = torch.tensor([[0, 1, 0, 1],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 1, 0]],
                       dtype=torch.float32,
                       device=device)

A_pert = A_dense + torch.tensor([[0, -1, 1, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                dtype=torch.float32,
                                device=device)

num_nodes = A_dense.shape[0]

alpha = 0.1
eps = 1e-12
ppr_idx = np.arange(num_nodes)
topk = 4

#ppr_topk = matrix_to_torch(ppr.topk_ppr_matrix(A_sp, alpha, eps, ppr_idx, topk, normalization='row')).to_dense()


def calc_A_row(adj: torch.Tensor) -> sp.spmatrix:
    """
    From https://github.com/klicperajo/ppnp
    """
    nnodes = adj.shape[0]
    A = adj + torch.eye(nnodes)
    D_vec = A.sum(-1)
    return A / D_vec[:, None]


def calc_ppr_exact_row(adj_matrix: torch.Tensor, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_row(adj_matrix)
    A_inner = torch.eye(nnodes) - (1 - alpha) * M
    return alpha * torch.inverse(A_inner)


ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)
ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)


D_vec = (torch.eye(num_nodes) + A_dense).sum(-1)
P_inv = (1 / alpha) * ppr_exact

norm_const = 1 / D_vec[:, None] * (alpha - 1)
u = norm_const * torch.tensor([[1],
                               [0],
                               [0],
                               [0]],
                              dtype=torch.float32,
                              device=device)
u_hat = u / (alpha - 1)


v = torch.tensor([[0, -1, 1, 0]],
                 dtype=torch.float32,
                 device=device)

# Sherman Morrison Formular for (P + uv)^-1
P_uv_inv = P_inv - (P_inv @ u @ v @ P_inv) / (1 + v @ P_inv @ u)

# to assert P_uv_inv is calculated correctly:
P = torch.inverse(P_inv)
P_uv_inv_2 = torch.inverse(P + u@v)

assert torch.allclose(P_uv_inv, P_uv_inv_2, atol=1e-05)

# check that P + u@v == I + (alpha -1) * (A_row + u_hat @ v)
A_row = calc_A_row(A_dense)
P_uv = P + u@v
P_uv_2 = torch.eye(num_nodes) + (alpha - 1) * (A_row + u_hat @ v)

assert torch.allclose(P_uv, P_uv_2, atol=1e-05)

# check that (A_row + u_hat @ v) == A_row_pert
A_row_pert = calc_A_row(A_pert)
assert torch.allclose((A_row + u_hat @ v), A_row_pert, atol=1e-05)

ppr_pert_update = alpha * (P_uv_inv)

assert torch.allclose(ppr_pert_update, ppr_pert_exact, atol=1e-05)
