# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from torch_measure.data.masking import col_mask, l_mask, random_mask, row_mask


class TestRandomMask:
    def test_output_types(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = random_mask(observed)
        assert train.dtype == torch.bool
        assert test.dtype == torch.bool

    def test_no_overlap(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = random_mask(observed)
        assert not (train & test).any()

    def test_covers_observed(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = random_mask(observed)
        assert (train | test).all()

    def test_approximate_fraction(self):
        torch.manual_seed(42)
        observed = torch.ones(100, 200, dtype=torch.bool)
        train, test = random_mask(observed, train_frac=0.8)
        frac = train.float().mean().item()
        assert 0.7 < frac < 0.9


class TestLMask:
    def test_output_shapes(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = l_mask(observed)
        assert train.shape == (10, 20)
        assert test.shape == (10, 20)

    def test_no_overlap(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = l_mask(observed)
        assert not (train & test).any()


class TestRowMask:
    def test_no_overlap(self):
        observed = torch.ones(20, 10, dtype=torch.bool)
        train, test = row_mask(observed)
        assert not (train & test).any()

    def test_covers_observed(self):
        observed = torch.ones(20, 10, dtype=torch.bool)
        train, test = row_mask(observed)
        assert (train | test).all()


class TestColMask:
    def test_no_overlap(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = col_mask(observed)
        assert not (train & test).any()

    def test_covers_observed(self):
        observed = torch.ones(10, 20, dtype=torch.bool)
        train, test = col_mask(observed)
        assert (train | test).all()
