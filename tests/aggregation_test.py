import os

import torch
from torch_sparse import SparseTensor

from rgnn_at_scale.aggregation import (_sparse_top_k, soft_weighted_medoid, soft_weighted_medoid_k_neighborhood,
                                       weighted_dimwise_median, weighted_medoid, weighted_medoid_k_neighborhood,
                                       soft_median)


device = 0 if torch.cuda.is_available() else 'cpu'
temperature = 1e-3

adj_bug_path = "tests/data_/adj_bug.tensor"
x_bug_path = "tests/data_/x_bug.tensor"


class TestTopK():

    def test_simple_example_cpu(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to_sparse()

        topk_values, topk_indices = _sparse_top_k(A._indices(), A._values(), A.shape[0], 2, return_sparse=False)
        assert torch.all(topk_values == torch.tensor([[0.5, 0.4], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]]))
        assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3], [0, 1], [2, 3]]))

        topk_values, topk_indices = _sparse_top_k(A._indices(), A._values(), A.shape[0], 3, return_sparse=False)
        assert torch.all(
            topk_values == torch.tensor([[0.5, 0.4, 0.3], [0.3, 0.2, 0], [0.9, 0.3, 0], [0.4, 0.4, 0.4]])
        )
        assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3, 1], [0, 1, -1], [2, 3, -1]]))

    if torch.cuda.is_available():
        def test_simple_example_cuda(self):
            A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                              [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to_sparse().cuda()
            topk_values, topk_indices = _sparse_top_k(A._indices(), A._values(), A.shape[0], 2, return_sparse=False)
            assert torch.all(topk_values == torch.tensor([[0.5, 0.4], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]]).cuda())
            assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3], [0, 1], [2, 3]]).cuda())
            topk_values, topk_indices = _sparse_top_k(A._indices(), A._values(), A.shape[0], 3, return_sparse=False)
            assert torch.all(
                topk_values == torch.tensor([[0.5, 0.4, 0.3], [0.3, 0.2, 0], [0.9, 0.3, 0], [0.4, 0.4, 0.4]]).cuda()
            )
            assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3, 1], [0, 1, -1], [2, 3, -1]]).cuda())


class TestWeightedMedoid():

    def test_simple_example_weighted(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = weighted_medoid(A, x)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

    def test_simple_example_unweighted(self):
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = weighted_medoid(A, x)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1]))

        layer_idx = 2
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[3]))

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])


class TestSoftWeightedMedoid():

    def test_simple_example_weighted(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = soft_weighted_medoid(A, x, temperature=temperature)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

    def test_simple_example_unweighted(self):
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = soft_weighted_medoid(A, x, temperature=temperature)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[1]) / 2)

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[2] + x[3]) / 2)

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])


class TestWeightedMedoidKNeighborhood():

    def test_simple_example_weighted_k2(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = weighted_medoid_k_neighborhood(A, x, k=2)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 3
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[3]))

    def test_simple_example_unweighted_k2(self):
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = weighted_medoid_k_neighborhood(A, x, k=2)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[3]))

        layer_idx = 1
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1]))

        layer_idx = 2
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[3]))

        layer_idx = 3
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[3]))

    def test_simple_example_weighted_k3(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = weighted_medoid_k_neighborhood(A, x, k=3)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

    def test_simple_example_unweighted_k3(self):
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32)
        medoids = weighted_medoid_k_neighborhood(A, x, k=3)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1]))

        layer_idx = 2
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[3]))

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])


class TestSoftWeightedMedoidKNeighborhood():

    def test_simple_example_weighted_k2(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4],
                          [0.3, 0.2, 0, 0],
                          [0, 0, 0.9, 0.0],
                          [0.4, 0, 0.4, 0.4]],
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         dtype=torch.float32,
                         device=device)
        medoids = soft_weighted_medoid_k_neighborhood(SparseTensor.from_dense(A), x, k=2, temperature=temperature)

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

    def test_simple_example_weighted_k3(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4],
                          [0.3, 0.2, 0, 0],
                          [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]],
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         dtype=torch.float32,
                         device=device)
        medoids = soft_weighted_medoid_k_neighborhood(SparseTensor.from_dense(A), x, k=3, temperature=temperature)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

    def test_simple_example_unweighted_k3(self):
        A = torch.tensor([[1, 1, 0, 1],
                          [1, 1, 0, 0],
                          [0, 0, 1, 1],
                          [1, 0, 1, 1]],
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         dtype=torch.float32,
                         device=device)
        medoids = soft_weighted_medoid_k_neighborhood(SparseTensor.from_dense(A).to(device),
                                                      x,
                                                      k=3,
                                                      temperature=temperature,
                                                      # threshold_for_dense_if_cpu=0
                                                      )

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[1]) / 2)

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[2] + x[3]) / 2)

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

    def test_disconnected_node_weighted_k2_sparse(self):
        """
        There was a bug in the soft_weighted_medoid_k_neighborhood method where nodes which have
        no outgoing edges will result in a RuntimeError caused by a size missmatch when doing
        the final matrix multuply when using the sparse implementation
        """

        A = torch.tensor([[0.5, 0.3, 0, 0.3],
                          [0.3, 0.2, 0, 0],
                          [0, 0, 0.9, 0],
                          [0, 0, 0, 0]],
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         requires_grad=True,
                         dtype=torch.float32,
                         device=device)

        A_sparse_tensor = SparseTensor.from_dense(A)

        medoids = soft_weighted_medoid_k_neighborhood(A_sparse_tensor,
                                                      x,
                                                      k=2,
                                                      temperature=temperature,
                                                      # forcing sparse implementation
                                                      threshold_for_dense_if_cpu=0)

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

        # just checking that we *can* compute the gradient,
        # not whether its actually correct
        medoids.sum().backward()
        assert x.grad is not None

    def test_disconnected_node_weighted_k2(self):
        """
        There was a bug in the soft_weighted_medoid_k_neighborhood method where nodes which have
        no outgoing edges will produce NaN embeddings for this node when using the dense
        cpu implementation
        """
        A = torch.tensor([[0.5, 0.3, 0, 0.4],
                          [0.3, 0.2, 0, 0],
                          [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]],
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         requires_grad=True,
                         dtype=torch.float32,
                         device=device)

        A_sparse_tensor = SparseTensor.from_dense(A)

        medoids = soft_weighted_medoid_k_neighborhood(A_sparse_tensor,
                                                      x,
                                                      k=2,
                                                      temperature=temperature)

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

        medoids.sum().backward()
        assert x.grad is not None

    def test_batched_node_weighted_k2_sparse(self):

        A = torch.tensor([[0.5, 0.3, 0, 0],
                          [0.0, 0.0, 0.9, 0],
                          [0.4, 0, 0.4, 0.4]],
                         #  requires_grad=True,
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         requires_grad=True,
                         dtype=torch.float32,
                         device=device)

        A_sparse_tensor = SparseTensor.from_dense(A)

        medoids = soft_weighted_medoid_k_neighborhood(A_sparse_tensor,
                                                      x,
                                                      k=2,
                                                      temperature=temperature,
                                                      # forcing sparse implementation
                                                      threshold_for_dense_if_cpu=0)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 2
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[2]) / 2)
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[3]) / 2)
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[2] + x[3]) / 2))

        # just checking that we *can* compute the gradient,
        # not whether its actually correct
        medoids.sum().backward()
        assert x.grad is not None

    def test_batched_node_weighted_k2(self):

        A = torch.tensor([[0.5, 0.3, 0, 0.3],
                          [0.0, 0.0, 0.9, 0],
                          [0.4, 0, 0.4, 0.4]],
                         dtype=torch.float32,
                         device=device)
        x = torch.tensor([[-10, 10, 10],
                          [-1, 1, 1],
                          [0, 0, 0],
                          [10, -10, -10]],
                         requires_grad=True,
                         dtype=torch.float32,
                         device=device)

        A_sparse_tensor = SparseTensor.from_dense(A)

        medoids = soft_weighted_medoid_k_neighborhood(A_sparse_tensor,
                                                      x,
                                                      k=2,
                                                      temperature=temperature)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 2
        assert (torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[2]) / 2)
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[3]) / 2)
                or torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[2] + x[3]) / 2))

        # just checking that we *can* compute the gradient,
        # not whether its actually correct
        medoids.sum().backward()
        assert x.grad is not None

    if os.path.isfile(adj_bug_path) and os.path.isfile(x_bug_path):
        def test_weighted_k2_out_of_bounds(self):
            """
            This particular setting threw an error:
            IndexError: index 2029 is out of bounds for dimension 0 with size 2027
            This test is to make sure it's fixed now and should just check that
            no error is thrown anymore

            The cause of the error was the partial_distance_matrix function that couldn't
            handle some batched adjacency matrices.
            """

            A = torch.load(adj_bug_path).to(device)
            x = torch.load(x_bug_path).to(device)

            A_sparse_tensor = SparseTensor.from_dense(A)

            soft_weighted_medoid_k_neighborhood(A_sparse_tensor,
                                                x,
                                                k=32,
                                                temperature=temperature,
                                                # forcing sparse implementation
                                                threshold_for_dense_if_cpu=0)


class TestWeightedDimwiseMedian():

    def test_simple_example_weighted(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to(device)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).to(device)
        median = weighted_dimwise_median(A.to_sparse(), x)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[0])

        layer_idx = 2
        assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[2])

        layer_idx = 3
        assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[2])

    def test_simple_example_unweighted(self):
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float32).to(device)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 0], [0, 0, 1], [10, -10, -10]], dtype=torch.float32).to(device)
        median = weighted_dimwise_median(A.to_sparse(), x)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(
            (median[layer_idx] == row_sum[layer_idx] * x[0])
            | (median[layer_idx] == row_sum[layer_idx] * x[1])
        )

        layer_idx = 2
        assert torch.all(
            (median[layer_idx] == row_sum[layer_idx] * x[2])
            | (median[layer_idx] == row_sum[layer_idx] * x[3])
        )

        layer_idx = 3
        assert median[layer_idx][0] == row_sum[layer_idx] * x[2][0]
        assert median[layer_idx][1] == row_sum[layer_idx] * x[2][1]
        assert median[layer_idx][2] == row_sum[layer_idx] * x[1][2]


class TestSoftMedian():

    if torch.cuda.is_available():
        def test_simple_example_weighted(self):
            A = torch.tensor([[0.5, 0.3, 0, 0.4],
                              [0.3, 0.2, 0, 0],
                              [0, 0, 0.9, 0.3],
                              [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to(device)
            x = torch.tensor([[-10, 10, 10],
                              [-1, 1, 1],
                              [0, 0, 0],
                              [10, -10, -10]], dtype=torch.float32).to(device)

            A_sparse_tensor = SparseTensor.from_dense(A)
            median = soft_median(A_sparse_tensor, x, temperature=temperature)

            row_sum = A.sum(-1)
            layer_idx = 0
            assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[1])

            layer_idx = 1
            assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[0])

            layer_idx = 2
            assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[2])

            layer_idx = 3
            assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[2])

    if torch.cuda.is_available():
        def test_simple_example_unweighted(self):
            A = torch.tensor([[1, 1, 0, 1],
                              [1, 1, 0, 0],
                              [0, 0, 1, 1],
                              [0, 1, 1, 1]], dtype=torch.float32).to(device)
            x = torch.tensor([[-10, 10, 10],
                              [-1, 1, 0],
                              [0, 0, 1],
                              [10, -10, -10]], dtype=torch.float32).to(device)

            A_sparse_tensor = SparseTensor.from_dense(A)
            median = soft_median(A_sparse_tensor, x, temperature=temperature)

            row_sum = A.sum(-1)
            layer_idx = 0
            assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[1])

            layer_idx = 1
            assert torch.all(
                (median[layer_idx] == row_sum[layer_idx] * x[0])
                | (median[layer_idx] == row_sum[layer_idx] * x[1])
            )

            layer_idx = 2
            assert torch.all(
                (median[layer_idx] == row_sum[layer_idx] * x[2])
                | (median[layer_idx] == row_sum[layer_idx] * x[3])
            )

            layer_idx = 3
            assert torch.all(median[layer_idx] == row_sum[layer_idx] * x[2])
