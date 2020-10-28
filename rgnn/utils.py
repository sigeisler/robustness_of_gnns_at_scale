"""For the util methods such as conversions or adjacency preprocessings.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix


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


def get_truncated_svd(adjacency_matrix: torch.Tensor, rank: int = 50):
    """Truncated SVD preprocessing as proposed in Negin Entezari, Saba A. Al - Sayouri, Amirali Darvishzadeh, and Evangelos E. Papalexakis. All you need is Low(rank):  Defending against adversarial attacks on graphs. 

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


def sparse_tensor(spmat: sp.spmatrix, grad: bool = False):
    """

    Convert a scipy.sparse matrix to a torch.SparseTensor.
    Parameters
    ----------
    spmat: sp.spmatrix
        The input (sparse) matrix.
    grad: bool
        Whether the resulting tensor should have "requires_grad".
    Returns
    -------
    sparse_tensor: torch.SparseTensor
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
    return torch.sparse_coo_tensor(spmat.nonzero(), spmat.data, size=spmat.shape,
                                   dtype=dtype, requires_grad=grad).coalesce()


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
