from rgnn_at_scale.data import prep_graph
import kernels_test
import kernels
import torch
torch.manual_seed(0)


X = torch.rand(4, 5, device=0)
A = torch.tensor([[1, 0, 1, 0], [0.5, 0.5, 0, 0], [1, 2, 1, 0], [1, 0, 0, 0]], device=0)
medians = kernels.dimmedian_idx(X, A.to_sparse().indices(), A.to_sparse().values(), 8, 4)
medians_test = kernels_test.dimmedian_idx(X, A.to_sparse().indices(), A.to_sparse().values(), 8, 4)

features = 64
for size in [1e2, 1e3, 1e4, 1e5]:
    size = int(size)
    print(size)
    X = torch.rand(size, features, device=0)
    A = torch.ones((size, size), device=0).to_sparse()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    medians = kernels.dimmedian_idx(X, A.indices(), A.values(), size * size, size)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    medians_test = kernels_test.dimmedian_idx(X, A.indices(), A.values(), size * size, size)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    assert (medians == medians_test).all()

for dataset in ['cora_ml']:
    print(dataset)
    graph = prep_graph(dataset, 0, dataset_root='datasets', make_undirected=True,
                       binary_attr=False, return_original_split=False)
    attr, adj, labels = graph[:3]

    A_rows, A_cols, A_vals = adj.coo()
    A_idx = torch.stack([A_rows, A_cols], dim=0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    medians = kernels.dimmedian_idx(attr, A_idx, A_vals, adj.nnz(), adj.size(0))
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    medians_test = kernels_test.dimmedian_idx(attr, A_idx, A_vals, adj.nnz(), adj.size(0))
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    assert (medians == medians_test).all()
