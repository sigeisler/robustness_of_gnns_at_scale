"""For the util methods such as conversions or adjacency preprocessings.
"""
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch_scatter
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor, SparseStorage, coalesce

from rgnn_at_scale.helper.ppr_utils import ppr_topk

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

try:
    import resource
    _resource_module_available = True
except ModuleNotFoundError:
    _resource_module_available = False


patch_typeguard()


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


def sparse_tensor_to_tuple(adj: SparseTensor) -> Tuple[torch.Tensor, ...]:
    s = adj.storage
    return (s.value(), s.row(), s.rowptr(), s.col(), s.colptr(),
            s.csr2csc(), s.csc2csr(), torch.tensor(s.sparse_sizes()))


def tuple_to_sparse_tensor(edge_weight: torch.Tensor, row: torch.Tensor, rowptr: torch.Tensor,
                           col: torch.Tensor, colptr: torch.Tensor, csr2csc: torch.Tensor,
                           csc2csr: torch.Tensor, sparse_size: torch.Tensor) -> SparseTensor:
    sp_st = SparseStorage(row=row, rowptr=rowptr, col=col, colptr=colptr, csr2csc=csr2csc, csc2csr=csc2csr,
                          value=edge_weight, sparse_sizes=sparse_size.tolist(), is_sorted=True)
    sparse_tensor = SparseTensor.from_storage(sp_st)
    return sparse_tensor


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    """
    From https://github.com/klicperajo/ppnp
    """
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)

    D_vec_invsqrt_corr = 1 / np.sqrt(np.sum(A, axis=1).A1)

    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)

    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact_sym(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    """
    From https://github.com/klicperajo/ppnp
    """
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


@typechecked
def row_norm(A: TensorType["a", "b"]):
    rowsum = A.sum(-1)
    norm_mask = rowsum != 0
    A[norm_mask] = A[norm_mask] / rowsum[norm_mask][:, None]
    return A / A.sum(-1)[:, None]


def calc_ppr_exact_row(A, alpha):
    num_nodes = A.shape[0]
    A_norm = row_norm(A)
    return alpha * torch.inverse(torch.eye(num_nodes, device=A.device) + (alpha - 1) * A_norm)


def calc_ppr_update(ppr: SparseTensor,
                    Ai: SparseTensor,
                    p: SparseTensor,
                    i: int,
                    alpha: float):
    num_nodes = ppr.size(0)
    assert ppr.size(1) == Ai.size(1), "shapes of ppr and adjacency must be the same"
    assert Ai[0, i].nnz() == 0, "The adjacency's must not have self loops"
    assert (torch.logical_or(Ai.storage.value() == 1, Ai.storage.value() == 0)).all().item(), \
        "The adjacency must be unweighted"
    assert p[0, i].sum() == 0, "Self loops must not be perturbed"
    assert torch.all(p.storage._value > 0), "For technical reasons all values in p must be greater than 0"

    v_rows, v_cols, v_vals = p.coo()
    v_idx = torch.stack([v_rows, v_cols], dim=0)

    Ai_rows, Ai_cols, Ai_vals = Ai.coo()
    Ai_idx = torch.stack([Ai_rows, Ai_cols], dim=0)

    ppr_rows, ppr_cols, ppr_vals = ppr.coo()
    ppr_idx = torch.stack([ppr_rows, ppr_cols], dim=0)

    # A_norm = A[i] / A[i].sum()
    Ai_norm_val = Ai_vals / Ai_vals.sum()

    u = SparseTensor.from_edge_index(edge_index=torch.tensor([[i], [0]]),
                                     edge_attr=torch.tensor([1], dtype=torch.float32),
                                     sparse_sizes=(num_nodes, 1))

    # sparse addition: row = A[i] + v
    row_sum = Ai.sum() + v_vals.sum()
    row_idx = torch.cat((v_idx, Ai_idx), dim=-1)
    row_weights = torch.cat((v_vals, Ai_vals))
    row_idx, row_weights = coalesce(
        row_idx,
        row_weights,
        m=1,
        n=num_nodes,
        op='sum'
    )
    # Works since the attack will always assign at least a small constant the elements in p
    # row_weights[row_weights > 1] = -row_weights[row_weights > 1] + 2

    # sparse normalization: row_diff = row / row.sum()
    row_weights /= row_sum
    v_vals /= row_sum

    # sparse subtraction: row_diff = row - A_norm
    row_idx = torch.cat((row_idx, Ai_idx), dim=-1)
    row_weights = torch.cat((row_weights, -Ai_norm_val))
    row_idx, row_weights = coalesce(
        row_idx,
        row_weights,
        m=1,
        n=num_nodes,
        op='sum'
    )

    # sparse normalization with const: (alpha - 1) * row_diff
    row_weights *= (alpha - 1)

    row = SparseTensor.from_edge_index(edge_index=row_idx,
                                       edge_attr=row_weights,
                                       sparse_sizes=(1, num_nodes))

    P_inv = SparseTensor.from_edge_index(edge_index=ppr_idx,
                                         edge_attr=(1 / alpha) * ppr_vals,
                                         sparse_sizes=(num_nodes, num_nodes))
    # P_inv @ u
    P_inv_at_u = P_inv @ u

    # (1 + row_diff_norm @ P_inv @ u)
    P_uv_inv_norm_const = row @ P_inv_at_u
    P_uv_inv_norm_const_vals = P_uv_inv_norm_const.storage.value()
    P_uv_inv_norm_const_vals += 1

    P_uv_inv_diff = P_inv_at_u @ (row @ P_inv)
    P_uv_inv_diff_rows, P_uv_inv_diff_cols, P_uv_inv_diff_vals = P_uv_inv_diff.coo()
    P_uv_inv_diff_idx = torch.stack([P_uv_inv_diff_rows, P_uv_inv_diff_cols], dim=0)

    P_uv_inv_diff_vals /= P_uv_inv_norm_const_vals

    # sparse subtraction: P_uv_inv = P_inv - P_uv_inv_diff
    P_inv_rows, P_inv_cols, P_inv_vals = P_inv.coo()
    P_inv_idx = torch.stack([P_inv_rows, P_inv_cols], dim=0)

    P_uv_inv_idx = torch.cat((P_inv_idx, P_uv_inv_diff_idx), dim=-1)
    P_uv_inv_weights = torch.cat((P_inv_vals, P_uv_inv_diff_vals * -1))

    P_uv_inv_idx, P_uv_inv_weights = coalesce(
        P_uv_inv_idx,
        P_uv_inv_weights,
        m=num_nodes,
        n=num_nodes,
        op='sum'
    )

    # ppr_pert_update = alpha * (P_uv_inv)
    P_uv_inv_weights *= alpha

    return SparseTensor.from_edge_index(edge_index=P_uv_inv_idx,
                                        edge_attr=P_uv_inv_weights,
                                        sparse_sizes=(num_nodes, num_nodes))


def mul(a: SparseTensor, v: float) -> SparseTensor:
    a = a.copy()
    a.storage._value = a.storage.value() * v
    if a.nnz() == 0:
        return SparseTensor(
            row=torch.tensor([0], dtype=torch.long, device=a.device()),
            col=torch.tensor([0], dtype=torch.long, device=a.device()),
            value=torch.tensor([0.], device=a.device()),
            sparse_sizes=a.sizes()
        )
    return a


def calc_ppr_update_sparse_result(ppr: sp.csr_matrix,
                                  Ai: SparseTensor,
                                  p: SparseTensor,
                                  i: int,
                                  alpha: float):
    """
        Returns only the i-th row of the updated ppr
    """
    num_nodes = ppr.shape[0]
    assert ppr.shape[1] == Ai.size(1), "shapes of ppr and adjacency must be the same"

    assert (torch.logical_or(Ai.storage.value() == 1, Ai.storage.value() == 0)).all().item(), \
        "The adjacency must be unweighted"
    assert torch.all(p.storage.value() > 0), "For technical reasons all values in p must be greater than 0"
    assert torch.all(p.storage.value() <= 1), "All values in p must be less than 1"

    v_rows, v_cols, v_vals = p.coo()
    # Avoid perturbing self-loops
    v_vals = torch.where(v_cols == i, torch.tensor(0., device=v_vals.device), v_vals)
    v_idx = torch.stack([v_rows, v_cols], dim=0)

    Ai_rows, Ai_cols, Ai_vals = Ai.coo()
    Ai_idx = torch.stack([Ai_rows, Ai_cols], dim=0)

    # A_norm = A[i] / A[i].sum()
    Ai_norm_val = Ai_vals / Ai_vals.sum()

    # sparse addition: row = A[i] + v
    row_idx = torch.cat((v_idx, Ai_idx), dim=-1)
    row_weights = torch.cat((v_vals, Ai_vals))
    row_idx, row_weights = coalesce(
        row_idx,
        row_weights,
        m=1,
        n=num_nodes,
        op='sum'
    )
    # Works since the attack will always assign at least a small constant the elements in p
    row_weights[row_weights > 1] = -row_weights[row_weights > 1] + 2

    # sparse normalization: row_diff = row / row.sum()
    row_sum = row_weights.sum()
    row_weights /= row_sum

    # sparse subtraction: row_diff = row - A_norm
    row_idx = torch.cat((row_idx, Ai_idx), dim=-1)
    row_weights = torch.cat((row_weights, -Ai_norm_val))
    row_idx, row_weights = coalesce(
        row_idx,
        row_weights,
        m=1,
        n=num_nodes,
        op='sum'
    )

    # sparse normalization with const: (alpha - 1) * row_diff
    row_weights *= (alpha - 1)

    # row can be dense
    row = SparseTensor.from_edge_index(edge_index=row_idx,
                                       edge_attr=row_weights,
                                       sparse_sizes=(1, num_nodes))

    # (1 + row_diff_norm @ P_inv @ u)
    with torch.no_grad():
        P_uv_inv_norm_const = (  # Shape [1, 1]
            1
            + (
                # Shape [1, |p|]
                mul(SparseTensor.from_scipy(ppr[row.storage.col().cpu(), i]).to(row.device()), 1 / alpha).t()
                @ row.storage.value()[:, None]  # Shape [|p|, 1]
            )
        )

    # (P_inv @ u @ row_diff_norm @ P_inv)
    ppr_slice = ppr[row.storage.col().cpu()]  # Shape [n, |p|]
    col_mask = ppr_slice.getnnz(0) > 0
    col_mask[ppr[i].indices] = True
    ppr_slice = ppr_slice[:, col_mask]  # Shape [l, |p|] - l depends on p (in expectation)

    P_uv_inv_diff = (  # Shape [l, 1]
        ppr[i, i] / alpha * (
            mul(SparseTensor.from_scipy(ppr_slice).to(row.device()).t(), 1 / alpha)  # Shape [l, |p|]
            @ row.storage.value()[:, None]  # Shape [|p|, 1]
        ).T
    )

    P_uv_inv_diff /= P_uv_inv_norm_const

    # sparse subtraction: P_uv_inv = P_inv[:,i] - P_uv_inv_diff

    P_uv_inv = torch.clamp(
        mul(SparseTensor.from_scipy(ppr[i, col_mask]).to(row.device()), 1 / alpha).to_dense() - P_uv_inv_diff,
        0
    )

    ppr_pert_update = alpha * P_uv_inv

    return SparseTensor(
        row=torch.zeros(col_mask.sum(), device=row.device(), dtype=torch.long),
        col=torch.arange(num_nodes, device=row.device())[col_mask],
        value=ppr_pert_update.squeeze(),
        sparse_sizes=(1, num_nodes)
    )


def calc_ppr_update_topk_sparse(ppr: SparseTensor,
                                Ai: SparseTensor,
                                p: torch.Tensor,
                                i: int,
                                alpha: float,
                                topk: int):

    num_nodes = Ai.size(1)
    ppr_pert_update = calc_ppr_update_sparse_result(ppr=ppr,
                                                    Ai=Ai,
                                                    p=p,
                                                    i=i,
                                                    alpha=alpha,)
    values, indices = torch.topk(ppr_pert_update, topk, dim=-1)
    col_ind = indices.flatten()
    row_idx = torch.zeros(topk, dtype=torch.long)
    return torch.sparse.FloatTensor(torch.stack([row_idx, col_ind]), values.flatten(), (1, num_nodes)).to_dense()


@typechecked
def calc_ppr_update_dense(ppr: TensorType["n_nodes", "n_nodes"],
                          A: TensorType["n_nodes", "n_nodes"],
                          p: TensorType[1, "n_nodes"],
                          i: int,
                          alpha: float):
    num_nodes = A.shape[0]
    assert ppr.shape == A.shape, "shapes of ppr and adjacency must be the same"
    assert (torch.diag(A) == torch.zeros(num_nodes, device=A.device)).all().item(), \
        "The adjacency's must not have self loops"
    assert (torch.logical_or(A == 1, A == 0)).all().item(), "The adjacency must be unweighted"

    u = torch.zeros((num_nodes, 1), dtype=torch.float32, device=ppr.device)
    u[i] = 1
    v = torch.where(A[i] > 0, -p, p)

    row = A[i] + v
    row = row / row.sum()
    A_norm = A[i] / A[i].sum()
    row_diff = row - A_norm
    row_diff_norm = (alpha - 1) * row_diff

    # Sherman Morrison Formular for (P + uv)^-1
    P_inv = (1 / alpha) * ppr
    P_uv_inv = P_inv - (P_inv @ u @ row_diff_norm @ P_inv) / (1 + row_diff_norm @ P_inv @ u)

    ppr_pert_update = alpha * (P_uv_inv)
    return ppr_pert_update


@typechecked
def calc_ppr_update_topk_dense(ppr: TensorType["n_nodes", "n_nodes"],
                               A: TensorType["n_nodes", "n_nodes"],
                               p: TensorType[1, "n_nodes"],
                               i: int,
                               alpha: float,
                               topk: int):
    num_nodes = A.shape[0]
    ppr_pert_update = calc_ppr_update_dense(ppr, A, p, i, alpha)
    values, indices = torch.topk(ppr_pert_update, topk, dim=-1)
    col_ind = indices.flatten()
    row_idx = torch.arange(num_nodes)[:, None].expand(num_nodes, topk).flatten()
    return torch.sparse.FloatTensor(torch.stack([row_idx, col_ind]), values.flatten()).coalesce().to_dense()


@typechecked
def get_ppr_matrix(adjacency_matrix: TensorType["n_nodes", "n_nodes"],
                   alpha: float = 0.15,
                   k: int = 32,
                   use_cpu: bool = False,
                   **kwargs) -> TensorType["n_nodes", "n_nodes"]:
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
    use_cpu : bool, optional
        If True the matrix inverion will be performed on the CPU, by default False.

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.
    """
    dim = -1

    if k < 1:
        k = adjacency_matrix.shape[0]

    assert alpha > 0 and alpha < 1
    if use_cpu:
        device = adjacency_matrix.device
        adjacency_matrix = adjacency_matrix.cpu()

    if adjacency_matrix.is_sparse:
        adjacency_matrix = adjacency_matrix.to_dense()

    dtype = adjacency_matrix.dtype

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


def _construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def get_approx_topk_ppr_matrix(edge_idx: torch.Tensor,
                               n: int,
                               alpha: float = 0.15,
                               k: float = 64,
                               ppr_err: float = 1e-4,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the personalized page rank diffusion of the adjacency matrix as proposed in Johannes Klicpera,
    Stefan Weißenberger, and Stephan Günnemann. Diffusion Improves Graph Learning.

    Parameters
    ----------
    edge_idx : torch.Tensor
        Sparse (unweighted) adjacency matrix.
    alpha : float, optional
        Teleport probability, by default 0.15.
    k : int, optional
        Neighborhood for sparsification, by default 32.
    ppr_err : bool, optional
        Admissible error, by default 1e-4

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.

    """
    edge_weight = torch.ones_like(edge_idx[0], dtype=torch.float32)
    edge_idx, edge_weight = coalesce(edge_idx, edge_weight, n, n)

    row, col = edge_idx

    deg = torch_scatter.scatter_add(edge_weight, col, dim=0, dim_size=n)

    edge_weight = edge_weight.cpu()
    row, col = row.cpu(), col.cpu()

    ppr = ppr_topk(
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


def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def to_symmetric_scipy(adjacency: sp.csr_matrix):
    sym_adjacency = (adjacency + adjacency.T).astype(bool).astype(float)

    sym_adjacency.tocsr().sort_indices()

    return sym_adjacency


def normalize_row(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    return normalize(adj_matrix, norm='l1', axis=1)


def normalize_symmetric(adj_matrix: sp.spmatrix) -> sp.spmatrix:

    D_vec_invsqrt_corr = 1 / np.sqrt(np.sum(adj_matrix, axis=1).A1)

    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)

    return D_invsqrt_corr @ adj_matrix @ D_invsqrt_corr


def sparse_tensor(spmat: sp.spmatrix, grad: bool = False):
    """

    Convert a scipy.sparse matrix to a SparseTensor.
    Parameters
    ----------
    spmat: sp.spmatrix
        The input (sparse) matrix.
    grad: bool
        Whether the resulting tensor should have "requires_grad".
    Returns
    -------
    sparse_tensor: SparseTensor
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
    return SparseTensor.from_scipy(spmat).to(dtype).coalesce()


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


def get_max_memory_bytes():
    if _resource_module_available:
        return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return np.nan


def matrix_to_torch(X):
    if sp.issparse(X):
        return SparseTensor.from_scipy(X)
    else:
        return torch.FloatTensor(X)
