import os

import numpy as np
import torch
from torch_sparse import SparseTensor

from rgnn_at_scale import utils

from pprgo.pytorch_utils import matrix_to_torch
from pprgo.ppr import topk_ppr_matrix

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.utils import calc_ppr_update, calc_ppr_update_dense, calc_ppr_update_topk_dense,\
    calc_ppr_exact_row, calc_A_row
from rgnn_at_scale.utils import calc_ppr_update_sparse_result

device = 0 if torch.cuda.is_available() else 'cpu'


class TestPPRUpdate():

    def test_simple_example_dense(self):
        alpha = 0.1
        i = 2
        A_dense = torch.tensor([[0, 1, 1, 1],
                                [1, 0, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 0]],
                               dtype=torch.float32)

        p = torch.tensor([[1.0, 1.0, 0, 0.3]],
                         dtype=torch.float32,
                         requires_grad=True)
        # A_dense = torch.tensor([[0, 1, 0, 1],
        #                         [1, 0, 1, 0],
        #                         [0, 0, 0, 1],
        #                         [1, 1, 1, 0]],
        #                        dtype=torch.float32)
        # p = torch.tensor([[0.3, 0.1, 0, 0.3]],
        #                  dtype=torch.float32,
        #                  requires_grad=True)

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
        A_pert = A_dense + u @ v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update, ppr_pert_exact, atol=1e-05)

        ppr_pert_update.sum().backward()
        assert p.grad is not None

    def test_simple_example_topk_dense(self):
        alpha = 0.9
        i = 2

        eps = 1e-16
        topk = 4

        A_dense = torch.tensor([[0, 1, 0, 1],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 1, 1, 0]],
                               dtype=torch.float32)
        p = torch.tensor([[0.3, 0.1, 0, 0.3]],
                         dtype=torch.float32,
                         requires_grad=True)

        num_nodes = A_dense.shape[0]
        ppr_idx = np.arange(num_nodes)

        A_sp = SparseTensor.from_dense(calc_A_row(A_dense)).to_scipy(layout="csr")

        ppr_topk = matrix_to_torch(topk_ppr_matrix(A_sp, alpha, eps, ppr_idx, topk, normalization='row')).to_dense()

        ppr_pert_update = calc_ppr_update_dense(ppr=ppr_topk,
                                                A=A_dense,
                                                p=p,
                                                i=i,
                                                alpha=alpha)

        u = torch.zeros((num_nodes, 1),
                        dtype=torch.float32)
        u[i] = 1
        v = torch.where(A_dense[i] > 0, -p, p)
        A_pert = A_dense + u @ v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update, ppr_pert_exact, atol=1e-05)

        ppr_pert_update.sum().backward()
        assert p.grad is not None

    def test_on_cora_topk_dense(self):
        alpha = 0.9
        i = 2

        eps = 1e-8
        topk = 64

        graph = prep_graph("cora_ml", device,  # dataset_root=data_dir,
                           make_undirected=True,
                           make_unweighted=True,
                           normalize=False,
                           binary_attr=False,
                           return_original_split=False)

        _, adj, labels = graph[:3]
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())

        A_dense = adj.to_dense()
        num_nodes = A_dense.shape[0]
        A_dense -= torch.eye(num_nodes)

        p = torch.rand((1, num_nodes),
                       requires_grad=True)
        p[0, i] = 0

        ppr_idx = np.arange(num_nodes)

        A_sp = SparseTensor.from_dense(calc_A_row(A_dense)).to_scipy(layout="csr")

        ppr_topk = matrix_to_torch(topk_ppr_matrix(A_sp, alpha, eps, ppr_idx, topk, normalization='row')).to_dense()

        ppr_pert_update_topk = calc_ppr_update_topk_dense(ppr=ppr_topk,
                                                          A=A_dense,
                                                          p=p,
                                                          i=i,
                                                          alpha=alpha,
                                                          topk=topk)
        u = torch.zeros((num_nodes, 1),
                        dtype=torch.float32)
        u[i] = 1
        v = torch.where(A_dense[i] > 0, -p, p)
        A_pert = A_dense + u @ v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update_topk, ppr_pert_exact, atol=1e-02)

    def test_simple_example_sparse(self):
        alpha = 0.1
        i = 2
        A_dense = torch.tensor([[0, 1, 0, 1],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [1, 1, 1, 0]],
                               dtype=torch.float32)
        ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)

        ppr_exact_sparse = SparseTensor.from_dense(ppr_exact)
        A_sparse = SparseTensor.from_dense(A_dense)

        num_nodes = A_dense.shape[0]
        # p_sample_size = 100
        # p_idx = torch.randint(num_nodes, (p_sample_size,)).unique()
        # p_val = torch.rand(p_idx.shape[0])

        p_dense = torch.tensor([[0.5, 0.0, 0, 0.0]],
                               dtype=torch.float32,
                               requires_grad=True)
        p = SparseTensor.from_dense(p_dense)

        ppr_pert_update = calc_ppr_update(ppr=ppr_exact_sparse,
                                          Ai=A_sparse[i],
                                          p=p,
                                          i=i,
                                          alpha=alpha)
        u = torch.zeros((num_nodes, 1),
                        dtype=torch.float32)
        u[i] = 1
        v = torch.where(A_dense[i] > 0, -p_dense, p_dense)
        A_pert = A_dense + u @ v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update.to_dense(), ppr_pert_exact, atol=1e-05)

    def test_simple_example_sparse_result(self):
        alpha = 0.1
        A_dense = torch.tensor([[0, 1, 0, 1],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [1, 1, 1, 0]],
                               dtype=torch.float32)
        ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)

        ppr_exact_sparse = SparseTensor.from_dense(ppr_exact)
        A_sparse = SparseTensor.from_dense(A_dense)

        num_nodes = A_dense.shape[0]

        for i in range(num_nodes):
            p_dense = torch.tensor([[0.5, 0.3, 0, 0.3]],
                                   dtype=torch.float32)
            p_dense[0, i] = 0
            p_dense.requires_grad = True
            p = SparseTensor.from_dense(p_dense)

            ppr_pert_update = calc_ppr_update_sparse_result(ppr=ppr_exact_sparse,
                                                            Ai=A_sparse[i],
                                                            p=p,
                                                            i=i,
                                                            alpha=alpha)
            u = torch.zeros((num_nodes, 1),
                            dtype=torch.float32)
            u[i] = 1
            v = torch.where(A_dense[i] > 0, -p_dense, p_dense)
            A_pert = A_dense + u @ v
            ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

            assert torch.allclose(ppr_pert_update, ppr_pert_exact[i], atol=1e-05)

            ppr_pert_update.sum().backward()
            assert p_dense.grad is not None

    def test_random_sparse_result(self):
        alpha = 0.1
        prob_edge = 0.1
        num_nodes = 100
        vector_size = int(0.1 * num_nodes)

        A_dense = torch.bernoulli(torch.full((num_nodes, num_nodes), prob_edge, dtype=torch.float32))
        ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)

        ppr_exact_sparse = SparseTensor.from_dense(ppr_exact)
        A_sparse = SparseTensor.from_dense(A_dense)

        for i in tqdm(range(num_nodes)):
            p = torch.zeros((1, vector_size)).uniform_()
            p_dense = torch.zeros((1, num_nodes))

            col = torch.randperm(num_nodes)
            col = col[col != i]
            col = col[:vector_size]

            p_dense[0, col] = p
            p_dense.requires_grad = True

            p = SparseTensor.from_dense(p_dense)

            ppr_pert_update = calc_ppr_update_sparse_result(ppr=ppr_exact_sparse,
                                                            Ai=A_sparse[i],
                                                            p=p,
                                                            i=i,
                                                            alpha=alpha)
            u = torch.zeros((num_nodes, 1),
                            dtype=torch.float32)
            u[i] = 1
            v = torch.where(A_dense[i] > 0, -p_dense, p_dense)
            A_pert = A_dense + u @ v
            ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

            assert torch.allclose(ppr_pert_update, ppr_pert_exact[i], atol=1e-05)

            ppr_pert_update.sum().backward()
            assert p_dense.grad is not None
