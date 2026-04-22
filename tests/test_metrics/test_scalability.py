# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from torch_measure.metrics.scalability import mokken_scalability


class TestMokkenScalability:
    def test_output_keys(self):
        torch.manual_seed(42)
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        result = mokken_scalability(data)
        assert "H" in result
        assert "H_items" in result
        assert "H_pairs" in result

    def test_output_shapes(self):
        torch.manual_seed(42)
        data = torch.bernoulli(torch.full((50, 10), 0.5))
        result = mokken_scalability(data)
        assert isinstance(result["H"], float)
        assert result["H_items"].shape == (10,)
        assert result["H_pairs"].shape == (10, 10)

    def test_structured_data_high_h(self):
        """Data from a strong Guttman pattern should have high H."""
        torch.manual_seed(42)
        n_subjects = 100
        ability = torch.linspace(-3, 3, n_subjects)
        difficulty = torch.linspace(-2, 2, 10)
        logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
        data = torch.bernoulli(torch.sigmoid(logit * 3))  # steep slopes
        result = mokken_scalability(data)
        assert result["H"] > 0.3

    def test_random_data_low_h(self):
        """Random data should have low scalability."""
        torch.manual_seed(42)
        data = torch.bernoulli(torch.full((100, 10), 0.5))
        result = mokken_scalability(data)
        assert result["H"] < 0.5
