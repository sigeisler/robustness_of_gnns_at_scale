"""For the util methods such as conversions or adjacency preprocessings.
"""
from typing import Sequence, Tuple, Union

import numba
import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops
import torch_scatter
import torch_sparse


# TODO: Move to base attack
def grad_with_checkpoint(outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)

    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()

    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def sparse_tensor_to_tuple(adj: torch_sparse.SparseTensor) -> Tuple[torch.Tensor, ...]:
    s = adj.storage
    return (s.value(), s.row(), s.rowptr(), s.col(), s.colptr(),
            s.csr2csc(), s.csc2csr(), torch.tensor(s.sparse_sizes()))


def tuple_to_sparse_tensor(edge_weight: torch.Tensor, row: torch.Tensor, rowptr: torch.Tensor,
                           col: torch.Tensor, colptr: torch.Tensor, csr2csc: torch.Tensor,
                           csc2csr: torch.Tensor, sparse_size: torch.Tensor) -> torch_sparse.SparseTensor:
    sp_st = torch_sparse.SparseStorage(row=row, rowptr=rowptr, col=col, colptr=colptr, csr2csc=csr2csc, csc2csr=csc2csr,
                                       value=edge_weight, sparse_sizes=sparse_size.tolist(), is_sorted=True)
    sparse_tensor = torch_sparse.SparseTensor.from_storage(sp_st)
    return sparse_tensor


def get_ppr_matrix(adjacency_matrix: torch.Tensor,
                   alpha: float = 0.15,
                   k: int = 32,
                   normalize_adjacency_matrix: bool = False,
                   use_cpu: bool = False,
                   **kwargs) -> torch.Tensor:
    """Calculates the personalized page rank diffusion of the adjacency matrix as proposed in Johannes Klicpera,
    Stefan Weißenberger, and Stephan Günnemann. Diffusion Improves Graph Learning.

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Sparse adjacency matrix.
    alpha : float, optional
        Teleport probability, by default 0.15.
    k : int, optional
        Neighborhood for sparsification, by default 32.
    normalize_adjacency_matrix : bool, optional
        Should be true if the adjacency matrix is not normalized via two-sided degree normalization, by default False.
    use_cpu : bool, optional
        If True the matrix inverion will be performed on the CPU, by default False.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    dim = -1

    assert alpha > 0 and alpha < 1
    assert k >= 1
    if use_cpu:
        device = adjacency_matrix.device
        adjacency_matrix = adjacency_matrix.cpu()

    dtype = adjacency_matrix.dtype

    if normalize_adjacency_matrix:
        if adjacency_matrix.is_sparse:
            adjacency_matrix = adjacency_matrix.to_dense()
        adjacency_matrix += torch.eye(*adjacency_matrix.shape, device=adjacency_matrix.device, dtype=dtype)
        D_tilde = torch.diag(1 / torch.sqrt(adjacency_matrix.sum(axis=1)))
        adjacency_matrix = D_tilde @ adjacency_matrix @ D_tilde
        del D_tilde

    adjacency_matrix = alpha * torch.inverse(
        torch.eye(*adjacency_matrix.shape, device=adjacency_matrix.device, dtype=dtype)
        - (1 - alpha) * adjacency_matrix
    )

    if use_cpu:
        adjacency_matrix = adjacency_matrix.to(device)

    selected_vals, selected_idx = torch.topk(adjacency_matrix, int(k), dim=dim)
    norm = selected_vals.sum(dim)
    norm[norm <= 0] = 1
    selected_vals /= norm[:, None]

    row_idx = torch.arange(adjacency_matrix.size(0), device=adjacency_matrix.device)[:, None]\
        .expand(adjacency_matrix.size(0), int(k))
    return torch.sparse.FloatTensor(
        torch.stack((row_idx.flatten(), selected_idx.flatten())),
        selected_vals.flatten()
    ).coalesce()


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()
        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val
            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
    return list(p.keys()), list(p.values())


@numba.njit(cache=True, parallel=True)
def _calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


def _ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""
    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]
    neighbors, weights = _calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                 numba.float32(alpha), numba.float32(epsilon), nodes, topk)
    return _construct_sparse(neighbors, weights, (len(nodes), nnodes))


def _construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


# TODO
def get_approx_topk_ppr_matrix(edge_idx: torch.Tensor,
                               n: int,
                               alpha: float = 0.15,
                               k: float = 64,
                               ppr_err: float = 1e-4,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_weight = torch.ones_like(edge_idx[0], dtype=torch.float32)
    edge_idx, edge_weight = torch_sparse.coalesce(edge_idx, edge_weight, n, n)

    row, col = edge_idx

    deg = torch_scatter.scatter_add(edge_weight, col, dim=0, dim_size=n)

    edge_weight = edge_weight.cpu()
    row, col = row.cpu(), col.cpu()

    ppr = _ppr_topk(
        adj_matrix=sp.csr_matrix((edge_weight, (row, col)), (n, n)),
        alpha=alpha,
        epsilon=ppr_err,
        nodes=np.arange(n),
        topk=k
    ).tocsr()

    edge_idx, edge_weight = [tensor.to(edge_idx.device) for tensor in from_scipy_sparse_matrix(ppr)]

    # Uncommented normalization
    row, col = edge_idx
    deg_inv = deg.sqrt()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]

    # Row norm
    norm = torch_scatter.scatter_sum(edge_weight, edge_idx[0], dim=-1, dim_size=n)
    norm[norm <= 0] = 1
    edge_weight /= norm[edge_idx[0]]

    return edge_idx, edge_weight


def get_truncated_svd(adjacency_matrix: torch.Tensor, rank: int = 50):
    """Truncated SVD preprocessing as proposed in Negin Entezari, Saba A. Al - Sayouri, Amirali Darvishzadeh, and
    Evangelos E. Papalexakis. All you need is Low(rank):  Defending against adversarial attacks on graphs.

    Attention: the result will not be sparse!

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Sparse [n,n] adjacency matrix.
    rank : int, optional
        Rank of the truncated SVD, by default 50.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    row, col = adjacency_matrix._indices().cpu()
    values = adjacency_matrix._values().cpu()
    N = adjacency_matrix.shape[0]

    low_rank_adj = sp.coo_matrix((values, (row, col)), (N, N))
    low_rank_adj = truncatedSVD(low_rank_adj, rank)
    low_rank_adj = torch.from_numpy(low_rank_adj).to(adjacency_matrix.device, adjacency_matrix.dtype)

    return svd_norm_adj(low_rank_adj).to_sparse()


def get_jaccard(adjacency_matrix: torch.Tensor, features: torch.Tensor, threshold: int = 0.01):
    """Jaccard similarity edge filtering as proposed in Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu,
    and Liming Zhu.  Adversarial examples for graph data: Deep insights into attack and defense.

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Sparse [n,n] adjacency matrix.
    features : torch.Tensor
        Dense [n,d] feature matrix.
    threshold : int, optional
        Similarity threshold for filtering, by default 0.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    row, col = adjacency_matrix._indices().cpu()
    values = adjacency_matrix._values().cpu()
    N = adjacency_matrix.shape[0]

    if features.is_sparse:
        features = features.to_dense()

    modified_adj = sp.coo_matrix((values, (row, col)), (N, N))
    modified_adj = drop_dissimilar_edges(features.cpu().numpy(), modified_adj, threshold=threshold)
    modified_adj = torch.sparse.FloatTensor(*from_scipy_sparse_matrix(modified_adj)).to(adjacency_matrix.device)
    return modified_adj


# TODO: This name is confusing as it only adds remaining self loops and converts the adjacency matrix zu symmetric
def normalize_adjacency_matrix(edge_index: torch.Tensor, edge_weight: torch.Tensor, n: int, op='mean') -> torch.tensor:
    """
    For calculating $\hat{A} = 𝐷^{−\frac{1}{2}} 𝐴 𝐷^{−\frac{1}{2}}$.

    Parameters
    ----------
    A: torch.sparse.FloatTensor
        Sparse adjacency matrix (potentially) without added self-loops.

    Returns
    -------
    A_hat: torch.sparse.FloatTensor
        Normalized message passing matrix
    """
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, num_nodes=n)
    edge_index, edge_weight = to_symmetric(edge_index, edge_weight, n, op=op)
    return edge_index, edge_weight


def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )
    symmetric_edge_weight = edge_weight.repeat(2)
    symmetric_edge_index, symmetric_edge_weight = torch_sparse.coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def sparse_tensor(spmat: sp.spmatrix, grad: bool = False):
    """

    Convert a scipy.sparse matrix to a torch_sparse.SparseTensor.
    Parameters
    ----------
    spmat: sp.spmatrix
        The input (sparse) matrix.
    grad: bool
        Whether the resulting tensor should have "requires_grad".
    Returns
    -------
    sparse_tensor: torch_sparse.SparseTensor
        The output sparse tensor.
    """
    if str(spmat.dtype) == "float32":
        dtype = torch.float32
    elif str(spmat.dtype) == "float64":
        dtype = torch.float64
    elif str(spmat.dtype) == "int32":
        dtype = torch.int32
    elif str(spmat.dtype) == "int64":
        dtype = torch.int64
    elif str(spmat.dtype) == "bool":
        dtype = torch.uint8
    else:
        dtype = torch.float32
    return torch_sparse.SparseTensor.from_scipy(spmat).to(dtype).coalesce()


def accuracy(logits: torch.Tensor, labels: torch.Tensor, split_idx: np.ndarray) -> float:
    """Returns the accuracy for a tensor of logits, a list of lables and and a split indices.

    Parameters
    ----------
    prediction : torch.Tensor
        [n x c] tensor of logits (`.argmax(1)` should return most probable class).
    labels : torch.Tensor
        [n x 1] target label.
    split_idx : np.ndarray
        [?] array with indices for current split.

    Returns
    -------
    float
        the Accuracy
    """
    return (logits.argmax(1)[split_idx] == labels[split_idx]).float().mean().item()


# For the next four methods, credits to https://github.com/DSE-MSU/DeepRobust


def drop_dissimilar_edges(features, adj, threshold: int = 0):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    modified_adj = adj.copy().tolil()

    edges = np.array(modified_adj.nonzero()).T
    removed_cnt = 0
    features = sp.csr_matrix(features)
    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue

        J = _jaccard_similarity(features[n1], features[n2])

        if J <= threshold:
            modified_adj[n1, n2] = 0
            modified_adj[n2, n1] = 0
            removed_cnt += 1
    return modified_adj


def _jaccard_similarity(a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return J


def svd_norm_adj(adj: torch.Tensor):
    mx = adj + torch.eye(adj.shape[0]).to(adj.device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def truncatedSVD(data, k=50):
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        diag_S = np.diag(S)

    return U @ diag_S @ V
