# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Tests for PairwiseComparisons data structure."""

import pandas as pd
import pytest
import torch

from torch_measure.data.pairwise import PairwiseComparisons


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pc():
    """A small PairwiseComparisons with 4 subjects and 5 comparisons."""
    subject_ids = ["alice", "bob", "carol", "dave"]
    # Comparisons: alice vs bob (alice wins), alice vs carol (carol wins),
    # bob vs carol (tie), bob vs dave (bob wins), carol vs dave (dave wins)
    subject_a = torch.tensor([0, 0, 1, 1, 2])
    subject_b = torch.tensor([1, 2, 2, 3, 3])
    outcome = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0])
    return PairwiseComparisons(
        subject_a=subject_a,
        subject_b=subject_b,
        outcome=outcome,
        subject_ids=subject_ids,
    )


@pytest.fixture
def full_pc():
    """PairwiseComparisons with all optional fields populated."""
    subject_ids = ["model_a", "model_b", "model_c"]
    subject_a = torch.tensor([0, 0, 1])
    subject_b = torch.tensor([1, 2, 2])
    outcome = torch.tensor([1.0, 0.5, 0.0])
    return PairwiseComparisons(
        subject_a=subject_a,
        subject_b=subject_b,
        outcome=outcome,
        subject_ids=subject_ids,
        item_ids=["q1", "q2", "q3"],
        item_contents=["What is 1+1?", "Explain gravity", "Write a poem"],
        item_idx=torch.tensor([0, 1, 2]),
        subject_metadata=[
            {"org": "openai", "model": "gpt-4"},
            {"org": "anthropic", "model": "claude-3"},
            {"org": "google", "model": "gemini"},
        ],
        comparison_metadata=[
            {"language": "English", "judge": "user_1"},
            {"language": "English", "judge": "user_2"},
            {"language": "Spanish", "judge": "user_3"},
        ],
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, simple_pc):
        assert simple_pc.n_comparisons == 5
        assert simple_pc.n_subjects == 4

    def test_dtypes(self, simple_pc):
        assert simple_pc.subject_a.dtype == torch.long
        assert simple_pc.subject_b.dtype == torch.long
        assert simple_pc.outcome.dtype == torch.float32

    def test_casts_int_to_long(self):
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0, 1], dtype=torch.int32),
            subject_b=torch.tensor([1, 0], dtype=torch.int32),
            outcome=torch.tensor([1.0, 0.0]),
            subject_ids=["a", "b"],
        )
        assert pc.subject_a.dtype == torch.long
        assert pc.subject_b.dtype == torch.long

    def test_rejects_2d_tensors(self):
        with pytest.raises(ValueError, match="1-D"):
            PairwiseComparisons(
                subject_a=torch.tensor([[0, 1]]),
                subject_b=torch.tensor([1, 0]),
                outcome=torch.tensor([1.0, 0.0]),
                subject_ids=["a", "b"],
            )

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            PairwiseComparisons(
                subject_a=torch.tensor([0, 1, 2]),
                subject_b=torch.tensor([1, 0]),
                outcome=torch.tensor([1.0, 0.0]),
                subject_ids=["a", "b", "c"],
            )

    def test_optional_fields_default_none(self, simple_pc):
        assert simple_pc.item_ids is None
        assert simple_pc.item_contents is None
        assert simple_pc.item_idx is None
        assert simple_pc.subject_metadata is None
        assert simple_pc.comparison_metadata is None

    def test_optional_fields_stored(self, full_pc):
        assert full_pc.item_ids == ["q1", "q2", "q3"]
        assert len(full_pc.item_contents) == 3
        assert full_pc.item_idx.shape == (3,)
        assert full_pc.item_idx.dtype == torch.long
        assert len(full_pc.subject_metadata) == 3
        assert len(full_pc.comparison_metadata) == 3

    def test_item_idx_cast_to_long(self):
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0]),
            subject_b=torch.tensor([1]),
            outcome=torch.tensor([1.0]),
            subject_ids=["a", "b"],
            item_ids=["q1"],
            item_idx=torch.tensor([0], dtype=torch.int32),
        )
        assert pc.item_idx.dtype == torch.long


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_n_comparisons(self, simple_pc):
        assert simple_pc.n_comparisons == 5

    def test_n_subjects(self, simple_pc):
        assert simple_pc.n_subjects == 4

    def test_n_items(self, full_pc):
        assert full_pc.n_items == 3

    def test_n_items_no_items(self, simple_pc):
        assert simple_pc.n_items == 0

    def test_shape(self, simple_pc):
        assert simple_pc.shape == (5, 4)

    def test_density(self, simple_pc):
        # 4 subjects -> 4*3/2 = 6 possible pairs, 5 observed
        assert simple_pc.density == pytest.approx(5 / 6)

    def test_density_two_subjects(self):
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0]),
            subject_b=torch.tensor([1]),
            outcome=torch.tensor([1.0]),
            subject_ids=["a", "b"],
        )
        # 1 possible pair, 1 observed
        assert pc.density == pytest.approx(1.0)

    def test_density_single_subject(self):
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0]),
            subject_b=torch.tensor([0]),
            outcome=torch.tensor([0.5]),
            subject_ids=["a"],
        )
        assert pc.density == 0.0


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


class TestWinRates:
    def test_shape(self, simple_pc):
        wr = simple_pc.win_rates()
        assert wr.shape == (4,)

    def test_values(self):
        """Subject 0 wins both comparisons -> win rate = 1.0."""
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0, 0]),
            subject_b=torch.tensor([1, 2]),
            outcome=torch.tensor([1.0, 1.0]),
            subject_ids=["a", "b", "c"],
        )
        wr = pc.win_rates()
        assert wr[0].item() == pytest.approx(1.0)
        assert wr[1].item() == pytest.approx(0.0)
        assert wr[2].item() == pytest.approx(0.0)

    def test_tie_counts_as_half(self):
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0]),
            subject_b=torch.tensor([1]),
            outcome=torch.tensor([0.5]),
            subject_ids=["a", "b"],
        )
        wr = pc.win_rates()
        assert wr[0].item() == pytest.approx(0.5)
        assert wr[1].item() == pytest.approx(0.5)


class TestToWinMatrix:
    def test_shape(self, simple_pc):
        mat = simple_pc.to_win_matrix()
        assert mat.shape == (4, 4)

    def test_diagonal_is_nan(self, simple_pc):
        mat = simple_pc.to_win_matrix()
        for i in range(4):
            assert torch.isnan(mat[i, i])

    def test_symmetry(self):
        """win_rate(i, j) + win_rate(j, i) should equal 1 for observed pairs."""
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0, 0]),
            subject_b=torch.tensor([1, 1]),
            outcome=torch.tensor([1.0, 0.0]),
            subject_ids=["a", "b"],
        )
        mat = pc.to_win_matrix()
        assert mat[0, 1].item() + mat[1, 0].item() == pytest.approx(1.0)

    def test_unobserved_pairs_are_nan(self):
        pc = PairwiseComparisons(
            subject_a=torch.tensor([0]),
            subject_b=torch.tensor([1]),
            outcome=torch.tensor([1.0]),
            subject_ids=["a", "b", "c"],
        )
        mat = pc.to_win_matrix()
        # (0,2) and (1,2) never observed
        assert torch.isnan(mat[0, 2])
        assert torch.isnan(mat[1, 2])
        assert torch.isnan(mat[2, 0])
        assert torch.isnan(mat[2, 1])


class TestTo:
    def test_returns_new_instance(self, simple_pc):
        moved = simple_pc.to("cpu")
        assert moved is not simple_pc

    def test_preserves_data(self, simple_pc):
        moved = simple_pc.to("cpu")
        assert torch.equal(moved.subject_a, simple_pc.subject_a)
        assert torch.equal(moved.outcome, simple_pc.outcome)
        assert moved.subject_ids == simple_pc.subject_ids

    def test_preserves_optional_fields(self, full_pc):
        moved = full_pc.to("cpu")
        assert moved.item_ids == full_pc.item_ids
        assert moved.item_contents == full_pc.item_contents
        assert torch.equal(moved.item_idx, full_pc.item_idx)
        assert moved.comparison_metadata == full_pc.comparison_metadata

    def test_to_without_item_idx(self, simple_pc):
        moved = simple_pc.to("cpu")
        assert moved.item_idx is None


class TestFromDataframe:
    def test_basic(self):
        df = pd.DataFrame({
            "model_a": ["gpt-4", "gpt-4", "claude"],
            "model_b": ["claude", "gemini", "gemini"],
            "outcome": [1.0, 0.5, 0.0],
        })
        pc = PairwiseComparisons.from_dataframe(df)
        assert pc.n_comparisons == 3
        assert pc.n_subjects == 3
        assert sorted(pc.subject_ids) == ["claude", "gemini", "gpt-4"]

    def test_custom_columns(self):
        df = pd.DataFrame({
            "a": ["x", "y"],
            "b": ["y", "x"],
            "result": [1.0, 0.0],
        })
        pc = PairwiseComparisons.from_dataframe(df, subject_a_col="a", subject_b_col="b", outcome_col="result")
        assert pc.n_comparisons == 2
        assert pc.n_subjects == 2


class TestRepr:
    def test_repr(self, simple_pc):
        r = repr(simple_pc)
        assert "PairwiseComparisons" in r
        assert "n_comparisons=5" in r
        assert "n_subjects=4" in r
