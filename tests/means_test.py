import torch

from rgnn.means import (_sparse_top_k, soft_weighted_medoid, soft_weighted_medoid_k_neighborhood,
                        weighted_dimwise_median, weighted_medoid, weighted_medoid_k_neighborhood)


device = 0 if torch.cuda.is_available() else 'cpu'
temperature = 1e-3


class TestTopK():

    def test_simple_example_cpu(self):
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to_sparse()
        topk_values, topk_indices = _sparse_top_k(A, 2, return_sparse=False)
        assert torch.all(topk_values == torch.tensor([[0.5, 0.4], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]]))
        assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3], [0, 1], [2, 3]]))
        topk_values, topk_indices = _sparse_top_k(A, 3, return_sparse=False)
        assert torch.all(
            topk_values == torch.tensor([[0.5, 0.4, 0.3], [0.3, 0.2, 0], [0.9, 0.3, 0], [0.4, 0.4, 0.4]])
        )
        assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3, 1], [0, 1, -1], [2, 3, -1]]))

    if torch.cuda.is_available():
        def test_simple_example_cuda(self):
            A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                              [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to_sparse().cuda()
            topk_values, topk_indices = _sparse_top_k(A, 2, return_sparse=False)
            assert torch.all(topk_values == torch.tensor([[0.5, 0.4], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]]).cuda())
            assert torch.all(topk_indices[:-1] == torch.tensor([[0, 3], [0, 1], [2, 3]]).cuda())
            topk_values, topk_indices = _sparse_top_k(A, 3, return_sparse=False)
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
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to(device)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).to(device)
        medoids = soft_weighted_medoid_k_neighborhood(A.to_sparse(), x, k=2, temperature=temperature)

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
        A = torch.tensor([[0.5, 0.3, 0, 0.4], [0.3, 0.2, 0, 0], [0, 0, 0.9, 0.3],
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).to(device)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).to(device)
        medoids = soft_weighted_medoid_k_neighborhood(A.to_sparse(), x, k=3, temperature=temperature)

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
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32).to(device)
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).to(device)
        medoids = soft_weighted_medoid_k_neighborhood(A.to_sparse(), x, k=3, temperature=temperature)

        row_sum = A.sum(-1)
        layer_idx = 0
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[1])

        layer_idx = 1
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[0] + x[1]) / 2)

        layer_idx = 2
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * (x[2] + x[3]) / 2)

        layer_idx = 3
        assert torch.all(medoids[layer_idx] == row_sum[layer_idx] * x[2])


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
