
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
                        [1, 0, 1, 0],
                        [1, 1, 0, 1],
                        [1, 0, 1, 0]],
                       dtype=torch.float32,
                       device=device)

u = torch.tensor([[0],
                  [0],
                  [-1],
                  [0]],
                 dtype=torch.float32,
                 device=device)
v = torch.tensor([[-1, 0, 0, 1]],
                 dtype=torch.float32,
                 device=device)

A_pert = A_dense + u@v


num_nodes = A_dense.shape[0]

alpha = 0.1
eps = 1e-12
ppr_idx = np.arange(num_nodes)
topk = 2


def calc_A_row(adj):
    A = torch.eye(adj.shape[0]) + adj
    return A / A.sum(-1)[:, None]


def calc_ppr_exact_row(A, alpha):
    A_norm = calc_A_row(A)
    return alpha * torch.inverse(torch.eye(4) + (alpha - 1) * A_norm)


ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)
ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

A_sp = torch_sparse.SparseTensor.from_dense(calc_A_row(A_dense)).to_scipy(layout="csr")
A_pert_sp = torch_sparse.SparseTensor.from_dense(calc_A_row(A_pert)).to_scipy(layout="csr")

ppr_topk = matrix_to_torch(ppr.topk_ppr_matrix(A_sp,
                                               alpha,
                                               eps,
                                               ppr_idx,
                                               topk,
                                               normalization='row')).to_dense()
ppr_pert_topk = matrix_to_torch(ppr.topk_ppr_matrix(A_pert_sp,
                                                    alpha,
                                                    eps,
                                                    ppr_idx,
                                                    topk,
                                                    normalization='row')).to_dense()

D_vec = (torch.eye(num_nodes) + A_dense).sum(-1)
norm_const = (alpha - 1) / D_vec[:, None]
u = u * norm_const
u_hat = u / (alpha - 1)


# Sherman Morrison Formular for (P + uv)^-1
P_inv = (1 / alpha) * ppr_exact
P_uv_inv = P_inv - (P_inv @ u @ v @ P_inv) / (1 + v @ P_inv @ u)

P_inv_topk = (1 / alpha) * ppr_topk
P_uv_inv_topk = P_inv_topk - (P_inv_topk @ u @ v @ P_inv_topk) / (1 + v @ P_inv_topk @ u)

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
ppr_pert_update_topk = alpha * (P_uv_inv_topk)

assert torch.allclose(ppr_pert_update, ppr_pert_exact, atol=1e-05)

# this will fail because ppr_topk is not row normalized.
# if the matrix would be bigger it would work again, because it's "almost" row normalized
assert torch.allclose(ppr_pert_update_topk, ppr_pert_exact, atol=1e-03)
# ppr_pert_topk
