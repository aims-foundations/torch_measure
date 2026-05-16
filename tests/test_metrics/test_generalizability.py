# Copyright (c) 2026 AIMS Foundations. MIT License.

import numpy as np
import pandas as pd
import pytest

from torch_measure.metrics.generalizability import (
    d_study,
    g_coefficient,
    variance_components,
)


def _synth_crossed_design(
    n_p: int,
    n_i: int,
    n_r: int,
    sigma_p: float = 1.0,
    sigma_i: float = 0.7,
    sigma_pi: float = 0.5,
    sigma_e: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate balanced long-form data with known variance components."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma_p, size=n_p)
    b = rng.normal(0.0, sigma_i, size=n_i)
    c = rng.normal(0.0, sigma_pi, size=(n_p, n_i))
    e = rng.normal(0.0, sigma_e, size=(n_p, n_i, n_r))
    y = a[:, None, None] + b[None, :, None] + c[:, :, None] + e
    rows = [(f"s{p}", f"i{i}", r, float(y[p, i, r])) for p in range(n_p) for i in range(n_i) for r in range(n_r)]
    return pd.DataFrame(rows, columns=["subject_id", "item_id", "trial", "response"])


class TestVarianceComponents:
    def test_returns_expected_keys(self):
        df = _synth_crossed_design(n_p=20, n_i=10, n_r=2)
        vc = variance_components(df)
        for key in (
            "subject",
            "item",
            "subject_item",
            "residual",
            "n_subjects",
            "n_items",
            "n_reps_harmonic",
            "identifiable",
            "method",
        ):
            assert key in vc
        assert vc["method"] == "moments"
        assert vc["n_subjects"] == 20
        assert vc["n_items"] == 10
        assert vc["n_reps_harmonic"] == pytest.approx(2.0)

    def test_recovers_known_components(self):
        # Henderson Method I recovers the SAMPLE variance of each effect (not
        # the population sigma it was drawn from), so we compare against
        # ddof=1 sample variances of the realized draws.
        rng = np.random.default_rng(42)
        n_p, n_i, n_r = 100, 30, 4
        a = rng.normal(0.0, 1.0, size=n_p)
        b = rng.normal(0.0, 0.7, size=n_i)
        c = rng.normal(0.0, 0.5, size=(n_p, n_i))
        e = rng.normal(0.0, 0.4, size=(n_p, n_i, n_r))
        y = a[:, None, None] + b[None, :, None] + c[:, :, None] + e

        rows = [(f"s{p}", f"i{i}", r, float(y[p, i, r])) for p in range(n_p) for i in range(n_i) for r in range(n_r)]
        df = pd.DataFrame(rows, columns=["subject_id", "item_id", "trial", "response"])
        vc = variance_components(df)

        assert vc["subject"] == pytest.approx(a.var(ddof=1), rel=0.05)
        assert vc["item"] == pytest.approx(b.var(ddof=1), rel=0.15)
        assert vc["subject_item"] == pytest.approx(c.var(ddof=1), rel=0.1)
        assert vc["residual"] == pytest.approx(e.var(ddof=1), rel=0.05)

    def test_identifiability_flag_when_no_reps(self):
        df = _synth_crossed_design(n_p=30, n_i=15, n_r=1)
        vc = variance_components(df)
        assert vc["identifiable"]["residual"] is False
        assert vc["residual"] == 0.0
        assert vc["identifiable"]["subject"] is True
        assert vc["identifiable"]["item"] is True
        assert vc["identifiable"]["subject_item"] is True

    def test_unbalanced_raises_on_missing_cell(self):
        df = _synth_crossed_design(n_p=10, n_i=5, n_r=2)
        df = df[~((df["subject_id"] == "s0") & (df["item_id"] == "i0"))]
        with pytest.raises(ValueError, match="Unbalanced design"):
            variance_components(df)

    def test_variable_n_reps_per_cell_uses_harmonic_mean(self):
        df = _synth_crossed_design(n_p=8, n_i=5, n_r=3)
        df = df[~((df["subject_id"] == "s0") & (df["item_id"] == "i0") & (df["trial"] == 2))]
        vc = variance_components(df)
        assert 0.0 < vc["n_reps_harmonic"] < 3.0
        assert vc["identifiable"]["residual"] is True

    def test_method_reml_not_implemented(self):
        df = _synth_crossed_design(n_p=5, n_i=4, n_r=2)
        with pytest.raises(NotImplementedError):
            variance_components(df, method="reml")

    def test_unknown_method_raises(self):
        df = _synth_crossed_design(n_p=5, n_i=4, n_r=2)
        with pytest.raises(ValueError, match="Unknown method"):
            variance_components(df, method="bogus")

    def test_missing_columns_raises(self):
        df = _synth_crossed_design(n_p=5, n_i=4, n_r=2).drop(columns=["item_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            variance_components(df)

    def test_non_numeric_response_raises(self):
        df = _synth_crossed_design(n_p=5, n_i=4, n_r=2)
        df["response"] = df["response"].astype(str)
        with pytest.raises(ValueError, match="must be numeric"):
            variance_components(df)

    def test_too_few_subjects_or_items_raises(self):
        df = _synth_crossed_design(n_p=1, n_i=5, n_r=2)
        with pytest.raises(ValueError, match="at least 2 subjects and 2 items"):
            variance_components(df)

    def test_custom_column_names(self):
        df = _synth_crossed_design(n_p=10, n_i=5, n_r=2).rename(
            columns={
                "subject_id": "model",
                "item_id": "task",
                "trial": "rep",
                "response": "score",
            }
        )
        vc = variance_components(df, subject_col="model", item_col="task", trial_col="rep", response_col="score")
        assert vc["n_subjects"] == 10
        assert vc["n_items"] == 5


class TestGCoefficient:
    def _vc(self) -> dict:
        return {
            "subject": 1.0,
            "item": 0.5,
            "subject_item": 0.3,
            "residual": 0.2,
        }

    def test_in_unit_interval(self):
        g = g_coefficient(self._vc(), n_items=20, n_reps=1, type="absolute")
        assert 0.0 <= g <= 1.0

    def test_grows_with_n_items(self):
        vc = self._vc()
        g_small = g_coefficient(vc, n_items=5, n_reps=1, type="absolute")
        g_large = g_coefficient(vc, n_items=30, n_reps=1, type="absolute")
        assert g_large > g_small

    def test_relative_ge_absolute(self):
        vc = self._vc()
        g_rel = g_coefficient(vc, n_items=20, n_reps=1, type="relative")
        g_abs = g_coefficient(vc, n_items=20, n_reps=1, type="absolute")
        assert g_rel >= g_abs

    def test_zero_components_returns_zero(self):
        vc = {"subject": 0.0, "item": 0.0, "subject_item": 0.0, "residual": 0.0}
        assert g_coefficient(vc, n_items=10, n_reps=1) == 0.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="type must be"):
            g_coefficient(self._vc(), n_items=10, n_reps=1, type="bogus")

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="Missing required keys"):
            g_coefficient({"subject": 1.0}, n_items=10, n_reps=1)

    def test_invalid_design_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            g_coefficient(self._vc(), n_items=0, n_reps=1)


class TestDStudy:
    def _vc(self) -> dict:
        return {
            "subject": 1.0,
            "item": 0.5,
            "subject_item": 0.3,
            "residual": 0.2,
        }

    def test_shape_and_columns(self):
        df = d_study(self._vc(), n_items_grid=[5, 10, 25], n_reps_grid=[1, 3])
        assert len(df) == 3 * 2
        assert set(df.columns) == {
            "n_items",
            "n_reps",
            "g_relative",
            "g_absolute",
            "se_relative",
            "se_absolute",
        }

    def test_se_decreases_with_n_items(self):
        df = d_study(self._vc(), n_items_grid=[5, 10, 25, 50], n_reps_grid=[1])
        se = df.sort_values("n_items")["se_absolute"].to_numpy()
        assert np.all(np.diff(se) < 0)

    def test_g_increases_with_n_items(self):
        df = d_study(self._vc(), n_items_grid=[5, 10, 25, 50], n_reps_grid=[1])
        g = df.sort_values("n_items")["g_absolute"].to_numpy()
        assert np.all(np.diff(g) > 0)

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            d_study(self._vc(), n_items_grid=[], n_reps_grid=[1])


def test_end_to_end_pipeline():
    """variance_components -> g_coefficient -> d_study composes cleanly."""
    df = _synth_crossed_design(n_p=40, n_i=15, n_r=3, seed=0)
    vc = variance_components(df)
    g = g_coefficient(vc, n_items=15, n_reps=3, type="absolute")
    proj = d_study(vc, n_items_grid=[15, 30], n_reps_grid=[1, 3])
    assert 0.0 < g < 1.0
    assert len(proj) == 4
