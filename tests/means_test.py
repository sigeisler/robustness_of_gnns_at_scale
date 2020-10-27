import torch

from rgnn.means import (soft_weighted_medoid, soft_weighted_medoid_k_neighborhood,
                        weighted_dimwise_median, weighted_medoid, weighted_medoid_k_neighborhood)


temperature = 1e-3


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
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).cuda()
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).cuda()
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
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).cuda()
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).cuda()
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
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32).cuda()
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).cuda()
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
                          [0.4, 0, 0.4, 0.4]], dtype=torch.float32).cuda()
        x = torch.tensor([[-10, 10, 10], [-1, 1, 1], [0, 0, 0], [10, -10, -10]], dtype=torch.float32).cuda()
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
        A = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float32).cuda()
        x = torch.tensor([[-10, 10, 10], [-1, 1, 0], [0, 0, 1], [10, -10, -10]], dtype=torch.float32).cuda()
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
