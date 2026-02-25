# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Tests for Fisher-information-based standard error functions."""

import pytest
import torch

from torch_measure.metrics.uncertainty import (
    ability_standard_errors,
    difficulty_standard_errors,
    discrimination_standard_errors,
)


@pytest.fixture
def irt_params():
    """Standard IRT parameters for testing."""
    torch.manual_seed(42)
    ability = torch.randn(30)
    difficulty = torch.randn(50)
    discrimination = torch.ones(50) * 1.5
    return ability, difficulty, discrimination


class TestAbilityStandardErrors:
    def test_output_shape(self, irt_params):
        ability, difficulty, _ = irt_params
        se = ability_standard_errors(ability, difficulty)
        assert se.shape == (30,)

    def test_positive(self, irt_params):
        ability, difficulty, _ = irt_params
        se = ability_standard_errors(ability, difficulty)
        assert (se > 0).all()

    def test_more_items_smaller_se(self):
        ability = torch.zeros(5)
        diff_10 = torch.randn(10)
        diff_50 = torch.cat([diff_10, torch.randn(40)])
        se_10 = ability_standard_errors(ability, diff_10)
        se_50 = ability_standard_errors(ability, diff_50)
        # More items should give smaller SE (on average)
        assert se_50.mean() < se_10.mean()

    def test_higher_discrimination_smaller_se(self):
        ability = torch.zeros(5)
        difficulty = torch.randn(20)
        disc_low = torch.ones(20) * 0.5
        disc_high = torch.ones(20) * 2.0
        se_low = ability_standard_errors(ability, difficulty, disc_low)
        se_high = ability_standard_errors(ability, difficulty, disc_high)
        assert (se_high < se_low).all()

    def test_extreme_ability_larger_se(self):
        difficulty = torch.zeros(30)
        ability_center = torch.zeros(1)
        ability_extreme = torch.tensor([5.0])
        se_center = ability_standard_errors(ability_center, difficulty)
        se_extreme = ability_standard_errors(ability_extreme, difficulty)
        assert se_extreme.item() > se_center.item()

    def test_rasch_vs_2pl(self, irt_params):
        ability, difficulty, discrimination = irt_params
        se_rasch = ability_standard_errors(ability, difficulty)
        se_2pl = ability_standard_errors(ability, difficulty, discrimination)
        # With discrimination > 1, 2PL should have smaller SE
        assert se_2pl.mean() < se_rasch.mean()


class TestDifficultyStandardErrors:
    def test_output_shape(self, irt_params):
        ability, difficulty, _ = irt_params
        response_matrix = torch.ones(30, 50)
        se = difficulty_standard_errors(ability, difficulty, response_matrix)
        assert se.shape == (50,)

    def test_positive(self, irt_params):
        ability, difficulty, _ = irt_params
        response_matrix = torch.ones(30, 50)
        se = difficulty_standard_errors(ability, difficulty, response_matrix)
        assert (se > 0).all()

    def test_mask_fewer_observations_larger_se(self, irt_params):
        ability, difficulty, _ = irt_params
        response_matrix = torch.ones(30, 50)
        mask_full = torch.ones(30, 50, dtype=torch.bool)
        mask_half = torch.ones(30, 50, dtype=torch.bool)
        mask_half[:15, :] = False  # only half the subjects observed

        se_full = difficulty_standard_errors(ability, difficulty, response_matrix, mask=mask_full)
        se_half = difficulty_standard_errors(ability, difficulty, response_matrix, mask=mask_half)
        assert (se_half > se_full).all()

    def test_with_discrimination(self, irt_params):
        ability, difficulty, discrimination = irt_params
        response_matrix = torch.ones(30, 50)
        se = difficulty_standard_errors(ability, difficulty, response_matrix, discrimination)
        assert se.shape == (50,)
        assert (se > 0).all()


class TestDiscriminationStandardErrors:
    def test_output_shape(self, irt_params):
        ability, difficulty, discrimination = irt_params
        response_matrix = torch.ones(30, 50)
        se = discrimination_standard_errors(ability, difficulty, discrimination, response_matrix)
        assert se.shape == (50,)

    def test_positive(self, irt_params):
        ability, difficulty, discrimination = irt_params
        response_matrix = torch.ones(30, 50)
        se = discrimination_standard_errors(ability, difficulty, discrimination, response_matrix)
        assert (se > 0).all()

    def test_more_subjects_smaller_se(self):
        torch.manual_seed(0)
        difficulty = torch.randn(10)
        discrimination = torch.ones(10)
        response_matrix_20 = torch.ones(20, 10)
        response_matrix_50 = torch.ones(50, 10)
        ability_20 = torch.randn(20)
        ability_50 = torch.randn(50)

        se_20 = discrimination_standard_errors(ability_20, difficulty, discrimination, response_matrix_20)
        se_50 = discrimination_standard_errors(ability_50, difficulty, discrimination, response_matrix_50)
        assert se_50.mean() < se_20.mean()

    def test_mask_fewer_observations_larger_se(self, irt_params):
        ability, difficulty, discrimination = irt_params
        response_matrix = torch.ones(30, 50)
        mask_full = torch.ones(30, 50, dtype=torch.bool)
        mask_half = torch.ones(30, 50, dtype=torch.bool)
        mask_half[:15, :] = False

        se_full = discrimination_standard_errors(ability, difficulty, discrimination, response_matrix, mask=mask_full)
        se_half = discrimination_standard_errors(ability, difficulty, discrimination, response_matrix, mask=mask_half)
        assert (se_half > se_full).all()
