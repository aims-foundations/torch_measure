# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.cat.fisher import fisher_information


class TestFisherInformation:
    def test_shape_1d(self):
        ability = torch.tensor([0.0, 1.0, -1.0])
        difficulty = torch.tensor([0.0, 0.5, -0.5, 1.0])
        info = fisher_information(ability, difficulty)
        assert info.shape == (3, 4)

    def test_shape_scalar(self):
        ability = torch.tensor(0.0)
        difficulty = torch.tensor([0.0, 0.5, -0.5])
        info = fisher_information(ability, difficulty)
        assert info.shape == (3,)

    def test_maximum_at_matching_difficulty(self):
        """Fisher information is maximized when ability == difficulty (for Rasch)."""
        ability = torch.tensor([0.0])
        difficulty = torch.linspace(-3, 3, 100)
        info = fisher_information(ability, difficulty)
        # Max info should be near difficulty=0 (matching ability)
        max_idx = info.squeeze().argmax()
        assert abs(difficulty[max_idx].item()) < 0.1

    def test_nonnegative(self):
        ability = torch.randn(10)
        difficulty = torch.randn(20)
        info = fisher_information(ability, difficulty)
        assert (info >= 0).all()

    def test_with_discrimination(self):
        ability = torch.tensor([0.0])
        difficulty = torch.tensor([0.0])
        disc_low = torch.tensor([0.5])
        disc_high = torch.tensor([2.0])
        info_low = fisher_information(ability, difficulty, disc_low)
        info_high = fisher_information(ability, difficulty, disc_high)
        # Higher discrimination -> higher information
        assert info_high.item() > info_low.item()

    def test_rasch_max_is_025(self):
        """For Rasch (a=1), max Fisher info is 0.25 when theta==b."""
        ability = torch.tensor([0.0])
        difficulty = torch.tensor([0.0])
        info = fisher_information(ability, difficulty)
        assert abs(info.item() - 0.25) < 1e-5
