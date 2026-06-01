# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for the diagnostics module (arXiv:2511.16842)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from torch_measure.diagnostics import (
    ItemJudge,
    average_tetrachoric_correlation,
    flag_items,
    gaussian_rank,
    item_scalability,
    item_total_correlation,
    tetrachoric_correlation,
)


def _rasch_matrix(n_subjects: int = 200, n_items: int = 30, seed: int = 0) -> torch.Tensor:
    """Generate a clean Rasch-distributed binary response matrix."""
    g = torch.Generator().manual_seed(seed)
    ability = torch.randn(n_subjects, generator=g)
    difficulty = torch.randn(n_items, generator=g)
    probs = torch.sigmoid(ability.unsqueeze(1) - difficulty.unsqueeze(0))
    return torch.bernoulli(probs, generator=g)


def _inject_anti_correlated_item(data: torch.Tensor, idx: int = 0) -> torch.Tensor:
    """Replace column ``idx`` with the inverse of the sum-score median split."""
    data = data.clone()
    other = torch.cat([data[:, :idx], data[:, idx + 1 :]], dim=1)
    sum_score = other.sum(dim=1)
    median = sum_score.median()
    data[:, idx] = (sum_score < median).float()
    return data


class TestSignalsUnderRasch:
    def test_item_total_correlation_positive_under_rasch(self):
        data = _rasch_matrix()
        scores = item_total_correlation(data)
        assert (scores > 0).all(), f"got negatives: {scores[scores <= 0]}"

    def test_item_scalability_non_negative_under_rasch(self):
        data = _rasch_matrix()
        scores = item_scalability(data)
        assert (scores >= 0).all(), f"got negatives: {scores[scores < 0]}"

    def test_average_tetrachoric_non_negative_under_rasch(self):
        data = _rasch_matrix()
        scores = average_tetrachoric_correlation(data)
        assert (scores >= 0).all(), f"got negatives: {scores[scores < 0]}"


class TestAnomalyDetection:
    def test_item_total_flags_anti_correlated_item(self):
        data = _inject_anti_correlated_item(_rasch_matrix())
        scores = item_total_correlation(data)
        assert scores[0].item() < 0


class TestTetrachoricMatrix:
    def test_shape_symmetric_unit_diagonal(self):
        data = _rasch_matrix(n_subjects=60, n_items=12)
        corr = tetrachoric_correlation(data)
        assert corr.shape == (12, 12)
        assert torch.allclose(corr, corr.T, atol=1e-6)
        assert torch.allclose(corr.diagonal(), torch.ones(12), atol=1e-6)


class TestGaussianRank:
    def test_monotonic_finite_no_nan(self):
        x = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7])
        out = gaussian_rank(x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))
        order_in = np.argsort(x.numpy())
        order_out = np.argsort(out)
        assert (order_in == order_out).all()

    def test_handles_nan_inputs(self):
        x = np.array([1.0, np.nan, 0.5, 2.0])
        out = gaussian_rank(x)
        assert np.all(np.isfinite(out))
        assert out[1] == 0.0


class TestFlagItems:
    def test_returns_dataframe_with_expected_columns(self):
        data = _rasch_matrix(n_subjects=80, n_items=15)
        df = flag_items(data)
        assert isinstance(df, pd.DataFrame)
        for col in [
            "item_idx",
            "tetrachoric_score",
            "scalability_score",
            "item_total_score",
            "ensemble_score",
            "flagged",
        ]:
            assert col in df.columns

    def test_ranks_anomalous_item_at_top(self):
        data = _inject_anti_correlated_item(_rasch_matrix(n_subjects=200, n_items=25))
        df = flag_items(data)
        top5 = df.head(5)["item_idx"].tolist()
        assert 0 in top5, f"injected bad item should rank in top 5, got {top5}"
        assert bool(df.iloc[0]["flagged"])

    def test_sorted_by_ensemble_descending(self):
        data = _rasch_matrix(n_subjects=80, n_items=15)
        df = flag_items(data)
        scores = df["ensemble_score"].to_numpy()
        assert np.all(scores[:-1] >= scores[1:])

    def test_judge_output_populated_for_flagged(self):
        data = _inject_anti_correlated_item(_rasch_matrix(n_subjects=120, n_items=20))
        calls: list[tuple[int, float]] = []

        def judge(item_text: str, item_idx: int, anomaly_score: float) -> str:
            calls.append((item_idx, anomaly_score))
            return f"judge:{item_idx}"

        df = flag_items(data, item_names=[f"q{i}" for i in range(20)], judge=judge)
        assert "judge_output" in df.columns
        flagged_rows = df[df["flagged"]]
        assert len(flagged_rows) >= 1
        assert flagged_rows["judge_output"].notna().all()
        assert (~df[~df["flagged"]]["judge_output"].notna()).all()
        assert len(calls) == int(df["flagged"].sum())

    def test_item_names_column_added(self):
        data = _rasch_matrix(n_subjects=60, n_items=10)
        names = [f"benchmark_q{i}" for i in range(10)]
        df = flag_items(data, item_names=names)
        assert "item_name" in df.columns
        assert set(df["item_name"]) == set(names)

    def test_ensemble_method_or_more_permissive_than_and(self):
        data = _rasch_matrix(n_subjects=80, n_items=15)
        df_or = flag_items(data, ensemble_method="or")
        df_and = flag_items(data, ensemble_method="and")
        assert df_or["flagged"].sum() >= df_and["flagged"].sum()

    def test_subset_of_signals_runs(self):
        data = _rasch_matrix(n_subjects=80, n_items=15)
        df = flag_items(data, signals=["item_total"])
        assert "item_total_score" in df.columns
        assert "tetrachoric_score" not in df.columns
        assert "scalability_score" not in df.columns

    def test_rejects_unknown_signal(self):
        data = _rasch_matrix(n_subjects=40, n_items=10)
        with pytest.raises(ValueError, match="Unknown signal"):
            flag_items(data, signals=["bogus"])

    def test_rejects_unknown_ensemble(self):
        data = _rasch_matrix(n_subjects=40, n_items=10)
        with pytest.raises(ValueError, match="ensemble_method"):
            flag_items(data, ensemble_method="weird")

    def test_rejects_mismatched_item_names_length(self):
        data = _rasch_matrix(n_subjects=40, n_items=10)
        with pytest.raises(ValueError, match="item_names"):
            flag_items(data, item_names=["a", "b"])


class TestItemJudgeProtocol:
    def test_plain_callable_satisfies_protocol(self):
        def judge(item_text: str, item_idx: int, anomaly_score: float) -> str:
            return ""

        assert isinstance(judge, ItemJudge)
