from typing import List
import numpy as np
import torch
from torch_sparse import SparseTensor, coalesce
from tqdm.auto import tqdm


from pprgo.pytorch_utils import matrix_to_torch
from pprgo.ppr import topk_ppr_matrix

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.utils import calc_ppr_update, calc_ppr_update_dense, calc_ppr_update_topk_dense,\
    calc_ppr_update_topk_sparse, calc_ppr_exact_row, calc_A_row
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
        alpha = 0.01
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
        alpha = 0.085
        i = 0

        eps = 1e-4
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

        p = torch.rand((1, num_nodes),
                       requires_grad=True)

        p[0, i] = 0

        ppr_idx = np.arange(num_nodes)

        A_sp = SparseTensor.from_dense(calc_A_row(A_dense)).to_scipy(layout="csr")

        ppr_topk_i = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, 1e-6, np.array([i]), num_nodes, normalization='row')).to_dense()
        ppr_topk = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, eps, ppr_idx, topk, normalization='row')).to_dense()

        ppr_topk[i] = ppr_topk_i
        ppr_exact = calc_ppr_exact_row(A_dense, alpha=alpha)

        ppr_pert_update_topk = calc_ppr_update_dense(ppr=ppr_topk,
                                                     A=A_dense,
                                                     p=p,
                                                     i=i,
                                                     alpha=alpha)

        ppr_pert_update_exact = calc_ppr_update_dense(ppr=ppr_exact,
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
        A_pert_sp = SparseTensor.from_dense(calc_A_row(A_pert)).to_scipy(layout="csr")
        ppr_pert_topk = matrix_to_torch(topk_ppr_matrix(
            A_pert_sp, alpha, eps, ppr_idx, topk, normalization='row')).to_dense()
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

            ppr_pert_update = calc_ppr_update_sparse_result(ppr=ppr_exact_sparse.to_scipy(layout="csr"),
                                                            Ai=A_sparse[i],
                                                            p=p,
                                                            i=i,
                                                            alpha=alpha).to_dense()
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

            ppr_pert_update = calc_ppr_update_sparse_result(ppr=ppr_exact_sparse.to_scipy(layout="csr"),
                                                            Ai=A_sparse[i],
                                                            p=p,
                                                            i=i,
                                                            alpha=alpha).to_dense()
            u = torch.zeros((num_nodes, 1),
                            dtype=torch.float32)
            u[i] = 1
            v = torch.where(A_dense[i] > 0, -p_dense, p_dense)
            A_pert = A_dense + u @ v
            ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

            assert torch.allclose(ppr_pert_update, ppr_pert_exact[i], atol=1e-05)

            ppr_pert_update.sum().backward()
            assert p_dense.grad is not None

    def test_cora_topk_sparse_result(self):
        alpha = 0.085
        i = 0

        eps = 1e-4
        topk = 128

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

        p_dense = torch.rand((1, num_nodes),
                             requires_grad=True)
        p_dense[0, i] = 0

        p = SparseTensor.from_dense(p_dense)

        ppr_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, num_nodes)])

        A_sp = adj.to_scipy(layout="csr")

        ppr_topk_i = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, 1e-6, np.array([i]), num_nodes, normalization='row'))
        ppr_topk_wo_i = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, eps, ppr_idx, topk, normalization='row'))

        r, c, v = ppr_topk_wo_i.coo()
        r[r >= i] += 1
        idx = torch.stack([r, c], dim=0)
        r_i, c_i, v_i = ppr_topk_i.coo()
        idx_i = torch.stack([r_i, c_i], dim=0)

        ppr_topk_idx = torch.cat((idx, idx_i), dim=-1)
        ppr_topk_weights = torch.cat((v, v_i))
        ppr_topk = SparseTensor.from_edge_index(edge_index=ppr_topk_idx,
                                                edge_attr=ppr_topk_weights,
                                                sparse_sizes=(num_nodes, num_nodes))

        ppr_pert_update_topk = calc_ppr_update_sparse_result(ppr=ppr_topk,
                                                             Ai=adj[i],
                                                             p=p,
                                                             i=i,
                                                             alpha=alpha
                                                             )

        u = torch.zeros((num_nodes, 1),
                        dtype=torch.float32)
        u[i] = 1
        v = torch.where(A_dense[i] > 0, -p_dense, p_dense)
        A_pert = A_dense + u @ v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update_topk, ppr_pert_exact[i], atol=1e-02)

    def test_cora_topk_sparse(self):
        alpha = 0.085
        i = 0

        eps = 1e-4
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

        p_dense = torch.rand((1, num_nodes),
                             requires_grad=True)
        p_dense[0, i] = 0

        p = SparseTensor.from_dense(p_dense)

        A_sp = adj.to_scipy(layout="csr")

        ppr_topk_i = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, 1e-6, np.array([i]), num_nodes, normalization='row'))

        ppr_topk_i_rows, ppr_topk_i_cols, ppr_topk_i_vals = ppr_topk_i.coo()
        # ppr_topk_i_idx, topk
        _, most_relevant_ppr_i_neighbors = torch.topk(ppr_topk_i_vals, k=256)
        _, relevant_ppr_i_neighbors = torch.topk(ppr_topk_i_vals, k=256 + 1024)

        least_relevant_ppr_i_neighbors = difference(torch.arange(num_nodes), relevant_ppr_i_neighbors)
        relevant_ppr_i_neighbors = difference(relevant_ppr_i_neighbors, most_relevant_ppr_i_neighbors)

        # make sure we don't recompute ppr_topk for attacked node i
        most_relevant_ppr_i_neighbors = difference(torch.tensor([i]), most_relevant_ppr_i_neighbors)
        # relevant_ppr_i_neighbors = difference(torch.tensor([i]), relevant_ppr_i_neighbors)
        # least_relevant_ppr_i_neighbors = difference(torch.tensor([i]), least_relevant_ppr_i_neighbors)

        assert most_relevant_ppr_i_neighbors.shape[0] + relevant_ppr_i_neighbors.shape[0] + \
            least_relevant_ppr_i_neighbors.shape[0] == num_nodes - 1

        ppr_topk_most_relevant = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, eps, most_relevant_ppr_i_neighbors.numpy(), 128, normalization='row'))

        ppr_topk_relevant = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, eps, relevant_ppr_i_neighbors.numpy(), 128, normalization='row'))

        ppr_topk_least_relevant = matrix_to_torch(topk_ppr_matrix(
            A_sp, alpha, eps, least_relevant_ppr_i_neighbors.numpy(), 32, normalization='row'))

        ppr_topk = sparse_concat_row([ppr_topk_i,
                                      ppr_topk_most_relevant,
                                      ppr_topk_relevant,
                                      ppr_topk_least_relevant],
                                     [torch.tensor([i]),
                                      most_relevant_ppr_i_neighbors,
                                      relevant_ppr_i_neighbors,
                                      least_relevant_ppr_i_neighbors
                                      ], num_nodes)

        ppr_pert_update_topk = calc_ppr_update_sparse_result(ppr=ppr_topk,
                                                             Ai=adj[i],
                                                             p=p,
                                                             i=i,
                                                             alpha=alpha
                                                             )

        u = torch.zeros((num_nodes, 1),
                        dtype=torch.float32)
        u[i] = 1
        v = torch.where(A_dense[i] > 0, -p_dense, p_dense)
        A_pert = A_dense + u @ v
        ppr_pert_exact = calc_ppr_exact_row(A_pert, alpha=alpha)

        assert torch.allclose(ppr_pert_update_topk, ppr_pert_exact[i], atol=1e-02)


def sparse_concat_row(sp_tensors: List[SparseTensor], row_tensors: List[torch.Tensor], num_nodes: int):
    new_idx = list()
    new_vals = list()

    for sp_tensor, row in zip(sp_tensors, row_tensors):
        r, c, v = sp_tensor.coo()
        # map to correct row indices

        def map_new_row(i):
            return row[i]

        r.apply_(map_new_row)

        new_idx.append(torch.stack([r, c], dim=0))
        new_vals.append(v)

    concatenated_idx = torch.cat(new_idx, dim=-1)
    concatenated_val = torch.cat(new_vals)
    concatenated_idx, concatenated_val = coalesce(
        concatenated_idx,
        concatenated_val,
        num_nodes, num_nodes
    )
    return SparseTensor.from_edge_index(edge_index=concatenated_idx,
                                        edge_attr=concatenated_val,
                                        sparse_sizes=(num_nodes, num_nodes))


def difference(t1: torch.Tensor, t2: torch.Tensor):
    """
    Returns the difference of two sets.
    Only works if t1 and t2 don't contain duplicate values
    """
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]
    return difference
