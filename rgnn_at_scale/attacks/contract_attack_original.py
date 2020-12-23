import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import flip_edges_


@torch.no_grad()
def expand_contract(model, graph, k=1, alpha=2.0, m=1, step=1, protected=None):
    """Attack the graph structure with the expand-contract algorithm.

    Parameters
    ----------
    alpha
        Expansion factor
    m
        Number of expansion-contraction cycles
    step
        Contract this many edges per step
    """

    assert alpha >= 1.0
    assert m >= 1
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=graph.a.device())

    rng = np.random.default_rng()

    orig_a = graph.a
    graph.a = orig_a.to_dense()
    n = graph.num_nodes

    # Store for each edge if it is eligible for flipping
    s = np.ones(n * (n - 1) // 2)

    # Lower-triangular indices for conversion between linear indices and matrix positions
    row, col = torch.tril_indices(n, n, offset=-1)
    row, col = row.numpy(), col.numpy()

    # Exclude any protected edges
    if protected is not None:
        for p in protected.cpu().numpy():
            s[(row == p) | (col == p)] = False

    n_edges = np.count_nonzero(s)
    assert n_edges >= k

    def log_likelihood():
        return -F.cross_entropy(model(graph), graph.y)

    def contract():
        # Contract the cohort until we are in budget again
        bar = tqdm(total=len(cohort) - k, leave=False, desc="Contract")
        while len(cohort) > k:
            llhs = graph.x.new_empty(len(cohort))
            for i, edge in enumerate(cohort):
                flip_edges_(graph.a, row[edge], col[edge])
                llhs[i] = log_likelihood()
                flip_edges_(graph.a, row[edge], col[edge])
            llhs = llhs.cpu().numpy()

            # Undo the flips that increase the log-likelihood the least when undone
            n_undo = min(step, len(cohort) - k)
            for edge in [cohort[i] for i in np.argpartition(llhs, n_undo)[:n_undo]]:
                cohort.remove(edge)
                flip_edges_(graph.a, row[edge], col[edge])
                s[edge] = True

            bar.update(n_undo)

    clean_ll = log_likelihood()
    bar = tqdm(total=m, desc="Cycles")

    # Randomly select a cohort of edges to flip
    cohort_size = min(int(alpha * k), n_edges)
    cohort = list(rng.choice(s.size, size=cohort_size, replace=False, p=s / s.sum()))
    flip_edges_(graph.a, row[cohort], col[cohort])
    s[cohort] = False

    n_expand = min(round((alpha - 1.0) * k), n_edges - k)
    if n_expand > 0:
        for _ in range(m - 1):
            contract()

            ll = log_likelihood()
            bar.set_description(f"Cycles (decr {clean_ll - ll:.5f})")
            bar.update()

            # Expand the cohort again
            new_edges = rng.choice(s.size, size=n_expand, replace=False, p=s / s.sum())
            flip_edges_(graph.a, row[new_edges], col[new_edges])
            s[new_edges] = False
            cohort.extend(list(new_edges))

    contract()
    bar.update()

    graph.a = orig_a

    return torch.from_numpy(np.vstack([row[cohort], col[cohort]])).to(graph.a.device())