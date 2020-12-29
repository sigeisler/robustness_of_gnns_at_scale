
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import torch_sparse
import numpy as np
from rgnn_at_scale.aggregation import (_sparse_top_k, soft_weighted_medoid, soft_weighted_medoid_k_neighborhood,
                                       weighted_dimwise_median, weighted_medoid, weighted_medoid_k_neighborhood)


# %%
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# %%
device = 0 if torch.cuda.is_available() else 'cpu'
temperature = 1e-3
device


# %%
A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                  [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to(device)
x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).to(device)

A_sparse_tensor = torch_sparse.SparseTensor.from_dense(A).to(device)


# %%
A_sparse_tensor = torch_sparse.SparseTensor.from_dense(torch.load("datasets/test_adj_matrix.pt"))
#x = torch_sparse.SparseTensor.from_dense(torch.load("datasets/test_attr_matrix.pt"))
x = torch.load("datasets/test_attr_matrix.pt")
A_sparse_tensor, x


# %%
medoids = soft_weighted_medoid_k_neighborhood(A_sparse_tensor,
                                              x,
                                              k=2,
                                              temperature=temperature,
                                              threshold_for_dense_if_cpu=0)


# %%
medoids


# %%
row_sum = A.sum(-1)
layer_idx = 0
assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

layer_idx = 1
assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

layer_idx = 2
assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

layer_idx = 3
assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[2]) / 2)
        or torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[3]) / 2)
        or torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[2] + x[3]) / 2))

# %% [markdown]
# For debugging _sparse_top_k:

# %%
unroll_idx = np.array([0, 1, 0, 1, 0, 1, 0, 1])
new_idx = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                        [0, 3, 0, 1, 2, 3, 0, 2]])
value_idx = torch.tensor([0, 2, 3, 4, 5, 6, 7, 8])
values = torch.tensor([[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]])


# %%


# %%


# %%


# %%


# %%
