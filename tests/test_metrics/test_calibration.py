# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.metrics.calibration import brier_score, expected_calibration_error


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        predicted = torch.tensor([0.0, 0.0, 1.0, 1.0])
        observed = torch.tensor([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(predicted, observed)
        assert ece < 0.1

    def test_worst_calibration(self):
        """Completely wrong predictions should have high ECE."""
        predicted = torch.tensor([0.0, 0.0, 1.0, 1.0])
        observed = torch.tensor([1.0, 1.0, 0.0, 0.0])
        ece = expected_calibration_error(predicted, observed)
        assert ece > 0.5

    def test_range(self):
        torch.manual_seed(42)
        predicted = torch.rand(100)
        observed = torch.bernoulli(predicted)
        ece = expected_calibration_error(predicted, observed)
        assert 0 <= ece <= 1

    def test_empty_after_mask(self):
        predicted = torch.tensor([0.5])
        observed = torch.tensor([float("nan")])
        ece = expected_calibration_error(predicted, observed)
        assert ece == 0.0


class TestBrierScore:
    def test_perfect_predictions(self):
        predicted = torch.tensor([0.0, 1.0, 0.0, 1.0])
        observed = torch.tensor([0.0, 1.0, 0.0, 1.0])
        bs = brier_score(predicted, observed)
        assert bs < 1e-5

    def test_worst_predictions(self):
        predicted = torch.tensor([1.0, 0.0])
        observed = torch.tensor([0.0, 1.0])
        bs = brier_score(predicted, observed)
        assert abs(bs - 1.0) < 1e-5

    def test_range(self):
        torch.manual_seed(42)
        predicted = torch.rand(100)
        observed = torch.bernoulli(torch.full((100,), 0.5))
        bs = brier_score(predicted, observed)
        assert 0 <= bs <= 1
