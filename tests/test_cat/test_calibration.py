# Copyright (c) 2026 AIMS Foundations. MIT License.

import math

import pytest
import torch

from torch_measure.cat.calibration import (
    AnchorCalibrator,
    _effective_lam,
    _fit_temperature,
)


class TestFitTemperature:
    def test_recovers_known_temperature(self):
        torch.manual_seed(0)
        T_true = 2.5
        z = torch.linspace(-3, 3, 256, dtype=torch.float64)
        y = torch.bernoulli(torch.sigmoid(z / T_true)).to(torch.float64)
        T_hat = _fit_temperature(z, y, (0.1, 10.0))
        assert abs(T_hat - T_true) < 0.15

    def test_unanimous_labels_pick_bound(self):
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.zeros(3, dtype=torch.float64)  # all-wrong -> push T up
        T_hat = _fit_temperature(z, y, (0.1, 10.0))
        assert T_hat > 5.0

    def test_empty_returns_one(self):
        z = torch.tensor([], dtype=torch.float64)
        y = torch.tensor([], dtype=torch.float64)
        assert _fit_temperature(z, y, (0.1, 10.0)) == 1.0

    def test_scale_equivariance(self):
        """Scaling logits by k should scale the recovered T by k."""
        torch.manual_seed(1)
        z = torch.randn(200, dtype=torch.float64)
        y = torch.bernoulli(torch.sigmoid(z)).to(torch.float64)
        T1 = _fit_temperature(z, y, (0.1, 10.0))
        T2 = _fit_temperature(3.0 * z, y, (0.1, 30.0))
        assert abs(T2 / T1 - 3.0) < 0.05


class TestEffectiveLam:
    def test_global_returns_zero(self):
        z = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = torch.tensor([1.0, 0.0])
        assert _effective_lam(z, y, 1.0, "global", 0.7, 0.1) == 0.0

    def test_label_var_unanimous_is_zero(self):
        z = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64)
        y = torch.ones(3)
        assert _effective_lam(z, y, 1.0, "label_var", 0.7, 0.1) == 0.0

    def test_label_var_balanced_peaks(self):
        z = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        eff = _effective_lam(z, y, 1.0, "label_var", 0.7, 0.1)
        # I = 0.25, eff = 0.7 * 0.25 / (0.25 + 0.1)
        assert abs(eff - 0.7 * 0.25 / 0.35) < 1e-6

    def test_fisher_saturated_collapses(self):
        """Model already confident (large |z|) -> Fisher info -> 0."""
        z = torch.tensor([10.0, -10.0, 10.0], dtype=torch.float64)
        y = torch.tensor([1.0, 0.0, 1.0])
        eff = _effective_lam(z, y, 1.0, "fisher", 0.7, 0.1)
        assert eff < 0.05

    def test_fisher_at_chance_is_max(self):
        """All z=0 -> p=0.5 -> Fisher info = 1 -> eff = lam_max * 1/(1+kappa)."""
        z = torch.zeros(4, dtype=torch.float64)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # z=0 makes leverage=0; the formula handles this by returning info=0.
        # The interesting case is small but nonzero z.
        z = torch.tensor([0.1, -0.1, 0.1, -0.1], dtype=torch.float64)
        eff = _effective_lam(z, y, 1.0, "fisher", 0.7, 0.1)
        assert eff > 0.6  # close to lam_max * 1/(1+0.1) ~= 0.636


class TestAnchorCalibrator:
    def _synth(self, T_per_cat, n_per_cat=64, seed=0):
        torch.manual_seed(seed)
        z_all, y_all, c_all = [], [], []
        for cat, T in enumerate(T_per_cat):
            z = torch.randn(n_per_cat, dtype=torch.float64) * 2.0
            y = torch.bernoulli(torch.sigmoid(z / T)).to(torch.float64)
            z_all.append(z)
            y_all.append(y)
            c_all.append(torch.full((n_per_cat,), cat, dtype=torch.long))
        return torch.cat(z_all), torch.cat(y_all), torch.cat(c_all)

    def test_recovers_per_category_temperature(self):
        T_true = [0.7, 1.5, 3.0]
        z, y, c = self._synth(T_true, n_per_cat=512)
        cal = AnchorCalibrator(gate="label_var", lam_T_max=0.95, kappa=0.01).fit(z, y, c)
        for cat, T in enumerate(T_true):
            assert abs(cal.T_by_cat[cat] - T) < 0.25, f"cat {cat}: {cal.T_by_cat[cat]} vs {T}"

    def test_identity_when_already_calibrated(self):
        z, y, c = self._synth([1.0, 1.0], n_per_cat=1024)
        cal = AnchorCalibrator(gate="global", bias_alpha=0.0).fit(z, y, c)
        p_raw = torch.sigmoid(z).clamp(0.02, 0.98)
        p_cal = cal.transform(z, c)
        assert (p_cal - p_raw).abs().mean() < 0.02

    def test_cold_category_falls_back(self):
        z, y, c = self._synth([1.0], n_per_cat=8)
        cal = AnchorCalibrator().fit(z, y, c)
        # Query a category not in the fit set.
        q_z = torch.tensor([0.5, -0.5])
        q_c = torch.tensor([99, 99])
        p = cal.transform(q_z, q_c)
        # Should use T_global, bias=0.
        expected = torch.sigmoid(q_z / cal.T_global).clamp(0.02, 0.98)
        assert torch.allclose(p, expected.to(p.dtype), atol=1e-6)

    def test_empty_fit_is_identity(self):
        cal = AnchorCalibrator(bias_alpha=0.0).fit(torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.long))
        z = torch.tensor([0.5, -0.5])
        p = cal.transform(z, torch.zeros(2, dtype=torch.long))
        assert torch.allclose(p, torch.sigmoid(z).clamp(0.02, 0.98), atol=1e-6)

    def test_single_anchor_per_category_is_finite(self):
        z = torch.tensor([0.3, -0.4])
        y = torch.tensor([1.0, 0.0])
        c = torch.tensor([0, 1])
        cal = AnchorCalibrator(gate="fisher").fit(z, y, c)
        p = cal.transform(torch.tensor([0.0, 0.0]), torch.tensor([0, 1]))
        assert torch.isfinite(p).all()

    def test_output_clipped(self):
        cal = AnchorCalibrator(clip_range=(0.1, 0.9))
        cal.T_global = 1.0
        p = cal.transform(torch.tensor([100.0, -100.0]), torch.tensor([0, 0]))
        assert p[0] == pytest.approx(0.9) and p[1] == pytest.approx(0.1)

    def test_invalid_gate_rejected(self):
        with pytest.raises(ValueError):
            AnchorCalibrator(gate="banana")  # type: ignore[arg-type]

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            AnchorCalibrator().fit(torch.zeros(3), torch.zeros(2), torch.zeros(3, dtype=torch.long))

    def test_fit_transform_equivalence(self):
        z, y, c = self._synth([1.5, 0.5], n_per_cat=32)
        cal_a = AnchorCalibrator().fit(z, y, c)
        p_a = cal_a.transform(z, c)
        p_b = AnchorCalibrator().fit_transform(z, y, c, z, c)
        assert torch.allclose(p_a, p_b)

    def test_preserves_input_shape(self):
        cal = AnchorCalibrator().fit(
            torch.tensor([0.1, -0.2, 0.3, 0.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([0, 0, 1, 1]),
        )
        z = torch.randn(3, 4)
        c = torch.zeros(3, 4, dtype=torch.long)
        p = cal.transform(z, c)
        assert p.shape == z.shape

    def test_residual_recovery(self):
        """Symmetric anchors with a constant residual -> bias term recovers it."""
        # 4 anchors, half label=1 half label=0, all with z=0 -> T well-determined
        # at T_global; residual is the obs_logit average which is ~0 for balanced.
        # Use unbalanced labels to make a real residual.
        z = torch.tensor([0.0, 0.0, 0.0, 0.0])
        y = torch.tensor([1.0, 1.0, 1.0, 0.0])  # 75% accuracy at z=0
        c = torch.tensor([0, 0, 0, 0])
        cal = AnchorCalibrator(gate="global", lam_bias=1.0, bias_alpha=1.0).fit(z, y, c)
        # Expected residual: 0.5 * logit(0.95) + 0.5 * (avg) - 0
        expected = 0.5 * (math.log(0.95 / 0.05) + math.log(0.05 / 0.95)) * 0 + (
            0.75 * math.log(0.95 / 0.05) + 0.25 * math.log(0.05 / 0.95)
        )
        assert abs(cal.bias_by_cat[0] - expected) < 1e-6
