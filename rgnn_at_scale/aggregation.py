"""The (robust) aggregations of our paper.
"""

import logging
import math
import os
import socket
from typing import Callable, Optional, Tuple

import numba
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.cpp_extension import load
import torch_scatter
import torch_sparse

from rgnn_at_scale.utils import sparse_tensor_to_tuple, tuple_to_sparse_tensor

try:
    try:
        import kernels as custom_cuda_kernels
        if not hasattr(custom_cuda_kernels, 'topk'):
            raise ImportError()
    except ImportError:
        cache_dir = os.path.join('.', 'extension', socket.gethostname(), torch.__version__)
        os.makedirs(cache_dir, exist_ok=True)
        custom_cuda_kernels = load(name="kernels",
                                   sources=["kernels/csrc/custom.cpp", "kernels/csrc/custom_kernel.cu"],
                                   extra_cuda_cflags=['-lcusparse', '-l', 'cusparse'],
                                   build_directory=cache_dir)
except:  # noqa: E722
    logging.warn('Cuda kernels could not loaded -> no CUDA support!')


class Chunker(object):

    def __init__(self, n: int, n_chunks: int, requires_grad: bool, do_synchronize: bool = False):
        self.n = n
        self.n_chunks = n_chunks
        self.requires_grad = requires_grad
        self.do_synchronize = do_synchronize
        self.chunk_size = int(math.ceil(n / n_chunks))
        self.lower = [chunk * self.chunk_size for chunk in range(self.n_chunks)]
        self.upper = [(chunk + 1) * self.chunk_size for chunk in range(self.n_chunks)]
        self.upper[-1] = n

    def chunk(self,
              get_run: Callable[[int, int], Callable],
              *input_tensors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        result = []
        for lower, upper in zip(self.lower, self.upper):
            if self.requires_grad:
                result.append(checkpoint(get_run(lower, upper), *input_tensors))
                if self.do_synchronize:
                    torch.cuda.synchronize()
            else:
                result.append(get_run(lower, upper)(*input_tensors))

        result = torch.cat(result)
        return result


def chunked_message_and_aggregate(
    adj_t: torch_sparse.SparseTensor,
    x: torch.Tensor,
    n_chunks: int = 8,
    aggregation_function: Optional[Callable[[torch_sparse.SparseTensor, torch.Tensor], torch.Tensor]] = None,
    **kwargs
) -> torch.Tensor:
    if aggregation_function is None:
        def aggregation_function(adj: torch_sparse.SparseTensor, x: torch.Tensor) -> torch.Tensor:
            return torch_sparse.matmul(adj, x, reduce='sum')

    n, _ = x.size()
    if not adj_t.coo()[-1].requires_grad:
        return aggregation_function(adj_t, x)

    edge_weight, *rest = sparse_tensor_to_tuple(adj_t)

    def row_chunked_matmul(
            lower: int,
            upper: int,
            *rest: Tuple[torch.Tensor, ...],
            is_sorted: bool = True):

        def row_chunked_matmul_run(edge_weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            adj = tuple_to_sparse_tensor(edge_weight, *rest)
            return aggregation_function(adj[lower:upper, :], x)
        return row_chunked_matmul_run

    chunker = Chunker(n, n_chunks, True)
    new_embeddings = chunker.chunk(
        lambda lower, upper: row_chunked_matmul(lower, upper, *rest),
        edge_weight, x
    )

    return new_embeddings


@numba.jit(nopython=True)
def _select_k_idx_cpu(row_idx: np.ndarray,
                      col_idx: np.ndarray,
                      values: np.ndarray,
                      k_per_row: np.ndarray,
                      n: int,
                      method: str = 'top'):
    new_idx = []
    valiue_idx = []
    unroll_idx = []

    sort_idx = row_idx.argsort()
    row_idx = row_idx[sort_idx]
    col_idx = col_idx[sort_idx]

    row_idx_start = 0
    row_idx_end = 0
    for i in range(n):
        k_of_row = k_per_row[i]
        while row_idx_end < len(row_idx) and row_idx[row_idx_end] < i + 1:
            row_idx_end += 1
        if k_of_row > 0:
            if method == 'top':
                curr_idx = (-values[row_idx_start:row_idx_end]).argsort()[:k_of_row]
            else:
                curr_idx = np.random.choice(row_idx_end - row_idx_start, k_of_row, replace=False)
            new_idx.append(np.stack((
                i * np.ones_like(curr_idx),
                col_idx[row_idx_start + curr_idx]
            )))
            valiue_idx.append(row_idx_start + curr_idx)
            unroll_idx.append(np.arange(len(curr_idx)))
        row_idx_start = row_idx_end

    return new_idx, valiue_idx, unroll_idx


def _sparse_top_k(A: torch_sparse.SparseTensor, k: int, return_sparse: bool = True):
    n = A.size(0)

    A_rows, A_cols, A_values = A.coo()
    A_indices = torch.stack([A_rows, A_cols], dim=0)
    print("_sparse_top_k")
    if A.is_cuda():
        print("is cuda")
        topk_values, topk_idx = custom_cuda_kernels.topk(A_indices, A_values, n, k)
        if not return_sparse:
            return topk_values, topk_idx.long()

        mask = topk_idx != -1
        row_idx = torch.arange(n, device=A.device()).view(-1, 1).expand(n, k)
        return torch.sparse.FloatTensor(torch.stack((row_idx[mask], topk_idx[mask].long())), topk_values[mask])

    n_edges_per_row = torch_scatter.scatter_sum(
        torch.ones_like(A_values),
        A_indices[0],
        dim=0
    )
    k_per_row = torch.clamp(
        n_edges_per_row,
        max=k
    ).long()

    new_idx, value_idx, unroll_idx = _select_k_idx_cpu(
        A_indices[0].cpu().numpy(),
        A_indices[1].cpu().numpy(),
        A_values.cpu().numpy(),
        k_per_row.cpu().numpy(),
        n,
        method='top'
    )

    new_idx = torch.from_numpy(np.hstack(new_idx)).to(A.device())
    value_idx = torch.from_numpy(np.hstack(value_idx)).to(A.device())

    if return_sparse:
        return torch.sparse.FloatTensor(new_idx, A_values[value_idx])
    else:
        unroll_idx = np.hstack(unroll_idx)
        values = torch.zeros((n, k), device=A.device())
        indices = -torch.ones((n, k), device=A.device(), dtype=torch.long)
        values[new_idx[0], unroll_idx] = A_values[value_idx]
        indices[new_idx[0], unroll_idx] = new_idx[1]
        return values, indices


def partial_distance_matrix(x: torch.Tensor, partial_idx: torch.Tensor) -> torch.Tensor:
    """Calculates the partial distance matrix given the indices. For a low memory footprint (small computation graph)
    it is essential to avoid duplicated computation of the distances.

    Parameters
    ----------
    x : torch.Tensor
        Dense [n, d] tensor with attributes to calculate the distance between.
    partial_idx : torch.Tensor
        Dense [n, k] tensor where `-1` stands for no index. Pairs are generated by the row id and the contained ids.

    Returns
    -------
    torch.Tensor
        [n, k, k] distances matrix (zero entries for `-1` indices)
    """
    n, k = partial_idx.shape

    # Permute the indices of partial_idx
    idx_row = partial_idx[:, None, :].expand(n, k, k).flatten()
    idx_column = partial_idx[:, None, :].expand(n, k, k).transpose(1, 2).flatten()
    missing_mask = (idx_row != -1) & (idx_column != -1)
    idx_row, idx_column = idx_row[missing_mask], idx_column[missing_mask]

    # Use symmetry of Euclidean distance to half memory footprint
    symmetry_mask = idx_column < idx_row
    idx_row[symmetry_mask], idx_column[symmetry_mask] = idx_column[symmetry_mask], idx_row[symmetry_mask]
    del symmetry_mask

    # Create linear index (faster deduplication)
    linear_index = idx_row * n + idx_column
    del idx_row
    del idx_column

    # Avoid duplicated distance calculation (helps greatly for space cost of backward)
    distance_matrix_idx, unique_reverse_index = torch.unique(linear_index, return_inverse=True)

    # Calculate Euclidean distances between all pairs
    sparse_distances = torch.norm(x[distance_matrix_idx // n] - x[distance_matrix_idx % n], dim=1)

    # Create dense output
    out = torch.zeros(n * k * k, dtype=torch.float, device=x.device)

    # Map sparse distances to output tensor
    out[missing_mask] = sparse_distances[unique_reverse_index]

    return out.view(n, k, k)


def soft_weighted_medoid_k_neighborhood(
    A: torch_sparse.SparseTensor,
    x: torch.Tensor,
    k: int = 32,
    temperature: float = 1.0,
    with_weight_correction: bool = True,
    threshold_for_dense_if_cpu: int = 5_000,
    **kwargs
) -> torch.Tensor:
    """Soft Weighted Medoid in the top `k` neighborhood (see Eq. 6 and Eq. 7 in our paper). This function can be used
    as a robust aggregation function within a message passing GNN (e.g. see `models#RGNN`).

    Note that if `with_weight_correction` is false, we calculate the Weighted Soft Medoid as in Appendix C.4.

    Parameters
    ----------
    A : torch_sparse.SparseTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    k : int, optional
        Neighborhood size for selecting the top k elements, by default 32.
    temperature : float, optional
        Controlling the steepness of the softmax, by default 1.0.
    with_weight_correction : bool, optional
        For enabling an alternative normalisazion (see above), by default True.
    threshold_for_dense_if_cpu : int, optional
        On cpu, for runtime reasons, we use a dense implementation if feasible, by default 5_000.

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    n, d = x.shape
    if k > n:
        if with_weight_correction:
            raise NotImplementedError('`k` less than `n` and `with_weight_correction` is not implemented.')
        return soft_weighted_medoid(A, x, temperature=temperature)
    if not x.is_cuda and n < threshold_for_dense_if_cpu:
        return dense_cpu_soft_weighted_medoid_k_neighborhood(A, x, k, temperature, with_weight_correction)

    # Custom CUDA extension / Numba JIT code for the top k values of the sparse adjacency matrix
    top_k_weights, top_k_idx = _sparse_top_k(A, k=k, return_sparse=False)

    print(top_k_idx)
    print(top_k_idx.shape)
    print(top_k_idx.max())
    # Partial distance matrix calculation
    distances_top_k = partial_distance_matrix(x, top_k_idx)

    # Multiply distances with weights
    distances_top_k = (top_k_weights[:, None, :].expand(n, k, k) * distances_top_k).sum(-1)
    distances_top_k[top_k_idx == -1] = torch.finfo(distances_top_k.dtype).max
    distances_top_k[~torch.isfinite(distances_top_k)] = torch.finfo(distances_top_k.dtype).max

    # Softmax over L1 criterium
    reliable_adj_values = F.softmax(-distances_top_k / temperature, dim=-1)
    del distances_top_k

    # To have GCN as a special case (see Eq. 6 in our paper)
    if with_weight_correction:
        reliable_adj_values = reliable_adj_values * top_k_weights
        # Bug: this is where we might introduce nan values if a node has no outgoing edges -> sum = 0
        # TODO fix this
        reliable_adj_values = reliable_adj_values / reliable_adj_values.sum(-1).view(-1, 1)

    # Map the top k results back to the (sparse) [n,n] matrix
    top_k_inv_idx_row = torch.arange(n, device=A.device())[:, None].expand(n, k).flatten()
    top_k_inv_idx_column = top_k_idx.flatten()
    top_k_mask = top_k_inv_idx_column != -1

    # TODO: The adjacency matrix A might have disconnected nodes. In that case applying the top_k_mask will
    # drop the nodes completely from the adj matrix making, changing its shape and therefore result in an
    # error later on when trying to multiply it with the unchanged attribute matrix x
    reliable_adj_index = torch.stack([top_k_inv_idx_row[top_k_mask], top_k_inv_idx_column[top_k_mask]])
    reliable_adj_values = reliable_adj_values[top_k_mask.view(n, k)]

    A_rows, _, A_values = A.coo()

    # Normalization and calculation of new embeddings
    a_row_sum = torch_scatter.scatter_sum(A_values, A_rows, dim=-1)
    new_embeddings = a_row_sum.view(-1, 1) * torch_sparse.spmm(reliable_adj_index, reliable_adj_values, n, n, x)
    return new_embeddings


def dense_cpu_soft_weighted_medoid_k_neighborhood(
    A: torch.sparse.FloatTensor,
    x: torch.Tensor,
    k: int = 32,
    temperature: float = 1.0,
    with_weight_correction: bool = False,
    **kwargs
) -> torch.Tensor:
    """Dense cpu implementation (for details see `soft_weighted_medoid_k_neighborhood`).
    """
    n, d = x.size()
    A_dense = A.to_dense()

    l2 = _distance_matrix(x)

    topk_a, topk_a_idx = torch.topk(A_dense, k=k, dim=1)
    topk_l2_idx = topk_a_idx[:, None, :].expand(n, k, k)
    distances_k = (
        topk_a[:, None, :].expand(n, k, k)
        * l2[topk_l2_idx, topk_l2_idx.transpose(1, 2)]
    ).sum(-1)
    distances_k[topk_a == 0] = torch.finfo(distances_k.dtype).max
    distances_k[~torch.isfinite(distances_k)] = torch.finfo(distances_k.dtype).max

    row_sum = A_dense.sum(-1)[:, None]

    topk_weights = torch.zeros(A_dense.shape, device=A_dense.device)

    topk_weights[torch.arange(n)[:, None].expand(n, k), topk_a_idx] = F.softmax(- distances_k / temperature, dim=-1)
    if with_weight_correction:
        topk_weights[torch.arange(n)[:, None].expand(n, k), topk_a_idx] *= topk_a
        topk_weights /= topk_weights.sum(-1)[:, None]
    return row_sum * (topk_weights @ x)


def weighted_dimwise_median(A: torch.sparse.FloatTensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """A weighted dimension-wise Median aggregation.

    Parameters
    ----------
    A : torch.sparse.FloatTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d]
    """
    if not A.is_cuda:
        return weighted_dimwise_median_cpu(A, x, **kwargs)

    assert A.is_sparse
    N, D = x.shape

    median_idx = custom_cuda_kernels.dimmedian_idx(x, A)
    col_idx = torch.arange(D, device=A.device).view(1, -1).expand(N, D)
    x_selected = x[median_idx, col_idx]

    a_row_sum = torch_scatter.scatter_sum(A._values(), A._indices()[0], dim=-1).view(-1, 1).expand(N, D)
    return a_row_sum * x_selected


def weighted_dimwise_median_cpu(A: torch.sparse.FloatTensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """A weighted dimension-wise Median aggregation (cpu implementation).

    Parameters
    ----------
    A : torch.sparse.FloatTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    N, D = x.shape
    x_sorted, index_x = torch.sort(x, dim=0)
    matrix_index_for_each_node = torch.arange(N, dtype=torch.long)[:, None, None].expand(N, N, D)
    A_cpu_dense = A.cpu()
    if A.is_sparse:
        A_cpu_dense = A_cpu_dense.to_dense()
    cum_sorted_weights = A_cpu_dense[matrix_index_for_each_node, index_x].cumsum(1)
    weight_sum_per_node = cum_sorted_weights.max(1)[0]
    median_element = (cum_sorted_weights < (weight_sum_per_node / 2)[:, None].expand(N, N, D)).sum(1).to(A.device)

    matrix_reverse_index = torch.arange(D, dtype=torch.long)[None, :].expand(N, D).to(A.device)
    x_selected = x[
        index_x[median_element, matrix_reverse_index],
        matrix_reverse_index
    ]
    return weight_sum_per_node.to(A.device) * x_selected


def _distance_matrix(x: torch.Tensor, eps_factor=1e2) -> torch.Tensor:
    """Naive dense distance matrix calculation.

    Parameters
    ----------
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    eps_factor : [type], optional
        Factor to be multiplied by `torch.finfo(x.dtype).eps` for "safe" sqrt, by default 1e2.

    Returns
    -------
    torch.Tensor
        n by n distance matrix.
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    x_norm_t = x_norm.transpose(0, 1)
    squared = x_norm + x_norm_t - (2 * (x @ x.transpose(0, 1)))
    # For "save" sqrt
    eps = eps_factor * torch.finfo(x.dtype).eps
    return torch.sqrt(torch.abs(squared) + eps)


def weighted_medoid(A: torch.sparse.FloatTensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """A weighted Medoid aggregation.

    Parameters
    ----------
    A : torch.sparse.FloatTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    N, D = x.shape
    l2 = _distance_matrix(x)
    A_cpu_dense = A.cpu()
    l2_cpu = l2.cpu()
    if A.is_sparse:
        A_cpu_dense = A_cpu_dense.to_dense()
    distances = A_cpu_dense[:, None, :].expand(N, N, N) * l2_cpu
    distances[A_cpu_dense == 0] = torch.finfo(distances.dtype).max
    distances = distances.sum(-1).to(x.device)
    distances[~torch.isfinite(distances)] = torch.finfo(distances.dtype).max
    row_sum = A_cpu_dense.sum(-1)[:, None].to(x.device)
    return row_sum * x[distances.argmin(-1)]


def weighted_medoid_k_neighborhood(A: torch.sparse.FloatTensor, x: torch.Tensor, k: int = 32, **kwargs) -> torch.Tensor:
    """A weighted Medoid aggregation.

    Parameters
    ----------
    A : torch.sparse.FloatTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    N, D = x.shape
    if k > N:
        return weighted_medoid(A, x)
    l2 = _distance_matrix(x)
    if A.is_sparse:
        A_dense = A.to_dense()
    else:
        A_dense = A
    topk_a, topk_a_idx = torch.topk(A_dense, k=k, dim=1)
    topk_l2_idx = topk_a_idx[:, None, :].expand(N, k, k)
    distances_k = (
        topk_a[:, None, :].expand(N, k, k)
        * l2[topk_l2_idx, topk_l2_idx.transpose(1, 2)]
    ).sum(-1)
    distances_k[topk_a == 0] = torch.finfo(distances_k.dtype).max
    distances_k[~torch.isfinite(distances_k)] = torch.finfo(distances_k.dtype).max
    row_sum = A_dense.sum(-1)[:, None]
    return row_sum * x[topk_a_idx[torch.arange(N), distances_k.argmin(-1)]]


def soft_weighted_medoid(
    A: torch.sparse.FloatTensor,
    x: torch.Tensor,
    temperature: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """A weighted Medoid aggregation.

    Parameters
    ----------
    A : torch.sparse.FloatTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    temperature : float, optional
        Temperature for the argmin approximation by softmax, by default 1.0

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    N, D = x.shape
    l2 = _distance_matrix(x)
    A_cpu_dense = A.cpu()
    l2_cpu = l2.cpu()
    if A.is_sparse:
        A_cpu_dense = A_cpu_dense.to_dense()
    distances = A_cpu_dense[:, None, :].expand(N, N, N) * l2_cpu
    distances[A_cpu_dense == 0] = torch.finfo(distances.dtype).max
    distances = distances.sum(-1).to(x.device)
    distances[~torch.isfinite(distances)] = torch.finfo(distances.dtype).max
    row_sum = A_cpu_dense.sum(-1)[:, None].to(x.device)
    return row_sum * (F.softmax(-distances / temperature, dim=-1) @ x)


def soft_median(
    A: torch.sparse.FloatTensor,
    x: torch.Tensor,
    p=2,
    temperature=1.0,
    eps=1e-12,
    do_synchronize: bool = False,
    **kwargs
) -> torch.Tensor:
    """Soft Weighted Median.

    Parameters
    ----------
    A : torch.sparse.FloatTensor
        Sparse [n, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    p : int, optional
        Norm for distance calculation
    temperature : float, optional
        Controlling the steepness of the softmax, by default 1.0.
    eps : float, optional
        Precision for softmax calculation.

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    n, d = x.size()
    edge_index, edge_weights = A._indices(), A._values()
    row_index, col_index = edge_index
    weight_sums = torch_scatter.scatter_add(edge_weights, row_index)

    with torch.no_grad():
        median_idx = custom_cuda_kernels.dimmedian_idx(x, torch.sparse.FloatTensor(edge_index, edge_weights, (n, n)))
        median_col_idx = torch.arange(d, device=x.device).view(1, -1).expand(n, d)
    x_median = x[median_idx, median_col_idx]

    distances = torch.norm(x_median[row_index] - x[col_index], dim=1, p=p) / pow(d, 1 / p)

    soft_weights = torch_scatter.composite.scatter_softmax(-distances / temperature, row_index, dim=-1, eps=eps)
    weighted_values = soft_weights * edge_weights
    row_sum_weighted_values = torch_scatter.scatter_add(weighted_values, row_index)
    final_adj_weights = weighted_values / row_sum_weighted_values[row_index] * weight_sums[row_index]

    new_embeddings = torch_sparse.spmm(edge_index, final_adj_weights, n, n, x)

    return new_embeddings


ROBUST_MEANS = {
    'dimmedian': weighted_dimwise_median,
    'medoid': weighted_medoid,
    'k_medoid': weighted_medoid_k_neighborhood,
    'soft_medoid': soft_weighted_medoid,
    'soft_k_medoid': soft_weighted_medoid_k_neighborhood,
    'soft_median': soft_median
}
