# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.metrics.correlation import point_biserial_correlation, tetrachoric_correlation


class TestTetrachoricCorrelation:
    def test_output_shape(self):
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        r = tetrachoric_correlation(data)
        assert r.shape == (10, 10)

    def test_diagonal_is_one(self):
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        r = tetrachoric_correlation(data)
        assert torch.allclose(r.diag(), torch.ones(10), atol=1e-5)

    def test_symmetric(self):
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        r = tetrachoric_correlation(data)
        assert torch.allclose(r, r.T, atol=1e-5)

    def test_range(self):
        torch.manual_seed(42)
        data = torch.bernoulli(torch.full((100, 10), 0.5))
        r = tetrachoric_correlation(data)
        assert (r >= -1.01).all()
        assert (r <= 1.01).all()

    def test_handles_nan(self):
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        data[0, 0] = float("nan")
        data[1, 1] = float("nan")
        r = tetrachoric_correlation(data)
        assert not torch.isnan(r).any()


class TestPointBiserialCorrelation:
    def test_basic(self):
        torch.manual_seed(42)
        continuous = torch.randn(100)
        binary = (continuous > 0).float()
        r = point_biserial_correlation(continuous, binary)
        # Should be strongly positive since binary encodes sign of continuous
        assert r.item() > 0.5

    def test_2d_binary(self):
        torch.manual_seed(42)
        continuous = torch.randn(100)
        binary = torch.bernoulli(torch.full((100, 5), 0.5))
        r = point_biserial_correlation(continuous, binary)
        assert r.shape == (5,)
