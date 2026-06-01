# Copyright (c) 2026 AIMS Foundations. MIT License.

import numpy as np
import pandas as pd
import pytest

from torch_measure.metrics.generalizability import (
    bootstrap_variance_components,
    d_study,
    g_coefficient,
    intraclass_correlation,
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


class TestIntraclassCorrelation:
    def _vc(self) -> dict:
        return {
            "subject": 1.0,
            "item": 0.5,
            "subject_item": 0.3,
            "residual": 0.2,
            "n_items": 10,
        }

    def test_in_unit_interval(self):
        for form in ("ICC2", "ICC3", "ICC2k", "ICC3k"):
            icc = intraclass_correlation(self._vc(), form=form)
            assert 0.0 <= icc <= 1.0

    def test_equivalence_with_g_coefficient(self):
        vc = self._vc()
        k = 8
        assert intraclass_correlation(vc, "ICC2k", n_items=k) == pytest.approx(
            g_coefficient(vc, n_items=k, n_reps=1, type="absolute")
        )
        assert intraclass_correlation(vc, "ICC3k", n_items=k) == pytest.approx(
            g_coefficient(vc, n_items=k, n_reps=1, type="relative")
        )

    def test_consistency_ge_absolute(self):
        vc = self._vc()
        assert intraclass_correlation(vc, "ICC3") >= intraclass_correlation(vc, "ICC2")
        assert intraclass_correlation(vc, "ICC3k") >= intraclass_correlation(vc, "ICC2k")

    def test_average_ge_single(self):
        vc = self._vc()
        assert intraclass_correlation(vc, "ICC3k", n_items=10) >= intraclass_correlation(vc, "ICC3")
        assert intraclass_correlation(vc, "ICC2k", n_items=10) >= intraclass_correlation(vc, "ICC2")

    def test_defaults_n_items_from_dict(self):
        vc = self._vc()
        assert intraclass_correlation(vc, "ICC3k") == pytest.approx(
            intraclass_correlation(vc, "ICC3k", n_items=vc["n_items"])
        )

    def test_icc1_raises(self):
        with pytest.raises(ValueError, match="one-way model"):
            intraclass_correlation(self._vc(), form="ICC1")

    def test_unknown_form_raises(self):
        with pytest.raises(ValueError, match="Unknown form"):
            intraclass_correlation(self._vc(), form="bogus")

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="Missing required keys"):
            intraclass_correlation({"subject": 1.0}, form="ICC3")

    def test_zero_components_returns_zero(self):
        vc = {"subject": 0.0, "item": 0.0, "subject_item": 0.0, "residual": 0.0, "n_items": 5}
        assert intraclass_correlation(vc, "ICC2") == 0.0

    def test_from_real_variance_components(self):
        df = _synth_crossed_design(n_p=60, n_i=12, n_r=2, seed=1)
        vc = variance_components(df)
        icc3k = intraclass_correlation(vc, "ICC3k")
        assert icc3k == pytest.approx(g_coefficient(vc, n_items=vc["n_items"], n_reps=1, type="relative"))


class TestBootstrapVarianceComponents:
    def _df(self, seed: int = 0) -> pd.DataFrame:
        return _synth_crossed_design(n_p=20, n_i=8, n_r=2, seed=seed)

    def test_output_structure(self):
        out = bootstrap_variance_components(self._df(), n_boot=50, seed=42)
        for k in ("subject", "item", "subject_item", "residual"):
            assert k in out
            assert k in out["ci"]
            lo, hi = out["ci"][k]
            assert lo <= hi
            assert out["samples"][k].shape == (50,)
        assert out["n_boot"] == 50
        assert out["ci_level"] == 0.95

    def test_point_matches_variance_components(self):
        df = self._df()
        out = bootstrap_variance_components(df, n_boot=10, seed=0)
        vc = variance_components(df)
        for k in ("subject", "item", "subject_item", "residual"):
            assert out[k] == pytest.approx(vc[k])

    def test_reproducibility_under_seed(self):
        df = self._df()
        a = bootstrap_variance_components(df, n_boot=30, seed=123)
        b = bootstrap_variance_components(df, n_boot=30, seed=123)
        for k in ("subject", "item", "subject_item", "residual"):
            np.testing.assert_allclose(a["samples"][k], b["samples"][k])
            assert a["ci"][k] == b["ci"][k]

    def test_different_seeds_differ(self):
        df = self._df()
        a = bootstrap_variance_components(df, n_boot=30, seed=1)
        b = bootstrap_variance_components(df, n_boot=30, seed=2)
        assert not np.allclose(a["samples"]["subject"], b["samples"]["subject"])

    @pytest.mark.slow
    def test_ci_brackets_point_estimate_for_dominant_component(self):
        # For the component with by far the largest signal, percentile CI on a
        # well-behaved design should bracket the point estimate.
        df = _synth_crossed_design(n_p=60, n_i=12, n_r=2, sigma_p=2.0, sigma_i=0.2, sigma_pi=0.2, sigma_e=0.2, seed=7)
        out = bootstrap_variance_components(df, n_boot=200, seed=7)
        lo, hi = out["ci"]["subject"]
        assert lo <= out["subject"] <= hi

    @pytest.mark.slow
    def test_custom_ci_level(self):
        df = self._df()
        narrow = bootstrap_variance_components(df, n_boot=200, ci=0.50, seed=5)
        wide = bootstrap_variance_components(df, n_boot=200, ci=0.95, seed=5)
        lo_n, hi_n = narrow["ci"]["subject"]
        lo_w, hi_w = wide["ci"]["subject"]
        assert (hi_w - lo_w) >= (hi_n - lo_n)

    def test_invalid_n_boot_raises(self):
        with pytest.raises(ValueError, match="n_boot must be >= 1"):
            bootstrap_variance_components(self._df(), n_boot=0)

    def test_invalid_ci_raises(self):
        with pytest.raises(ValueError, match=r"ci must be in \(0, 1\)"):
            bootstrap_variance_components(self._df(), n_boot=10, ci=1.5)

    @pytest.mark.slow
    def test_g_coefficient_ci_via_samples(self):
        # Users derive G-coefficient CIs by mapping g_coefficient over the
        # returned samples. Verify this composes cleanly.
        df = self._df()
        out = bootstrap_variance_components(df, n_boot=100, seed=0)
        g_samples = np.array(
            [
                g_coefficient(
                    {
                        "subject": s,
                        "item": i,
                        "subject_item": pi,
                        "residual": e,
                    },
                    n_items=8,
                    n_reps=2,
                    type="absolute",
                )
                for s, i, pi, e in zip(
                    out["samples"]["subject"],
                    out["samples"]["item"],
                    out["samples"]["subject_item"],
                    out["samples"]["residual"],
                    strict=True,
                )
            ]
        )
        g_lo, g_hi = np.quantile(g_samples, [0.025, 0.975])
        assert 0.0 <= g_lo <= g_hi <= 1.0


def test_end_to_end_pipeline():
    """variance_components -> g_coefficient -> d_study composes cleanly."""
    df = _synth_crossed_design(n_p=40, n_i=15, n_r=3, seed=0)
    vc = variance_components(df)
    g = g_coefficient(vc, n_items=15, n_reps=3, type="absolute")
    proj = d_study(vc, n_items_grid=[15, 30], n_reps_grid=[1, 3])
    assert 0.0 < g < 1.0
    assert len(proj) == 4
