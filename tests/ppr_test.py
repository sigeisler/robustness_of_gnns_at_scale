import os

import numpy as np
import torch
from torch_sparse import SparseTensor

from rgnn_at_scale import utils

from pprgo.pytorch_utils import matrix_to_torch
from pprgo.ppr import topk_ppr_matrix

from rgnn_at_scale.utils import calc_ppr_update_dense, calc_ppr_exact_row, calc_A_row

device = 0 if torch.cuda.is_available() else 'cpu'


class TestPPRUpdate():

    def test_simple_example_cpu(self):
        alpha = 0.1
        i = 2
        A_dense = torch.tensor([[0, 1, 0, 1],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 1, 1, 0]],
                               dtype=torch.float32)
        p = torch.tensor([[0.3, 0.1, 0, 0.3]],
                         dtype=torch.float32,
                         requires_grad=True)

        ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)

        ppr_pert_update = calc_ppr_update_dense(ppr=ppr_exact,
                                                A=A_dense,
                                                p=p,
                                                i=i,
                                                alpha=alpha)

        num_nodes = A_dense.shape[0]
        u = torch.zeros((num_nodes, 1),
                        dtype=torch.float32)
        u[i] = 1
        v = torch.where(A_dense[i] > 0, -p, p)
        A_pert = A_dense + u@v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update, ppr_pert_exact, atol=1e-05)

        ppr_pert_update.sum().backward()
        assert p.grad is not None

    if torch.cuda.is_available():
        def test_simple_example_cuda(self):
            pass
