# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for BradleyTerry pairwise comparison model."""

import pytest
import torch

from torch_measure.data.pairwise import PairwiseComparisons
from torch_measure.models.bradley_terry import BradleyTerry


class TestInit:
    def test_parameter_shape(self):
        model = BradleyTerry(n_subjects=5)
        assert model.ability.shape == (5,)

    def test_n_subjects(self):
        model = BradleyTerry(n_subjects=10)
        assert model.n_subjects == 10

    def test_repr(self):
        model = BradleyTerry(n_subjects=7)
        assert "BradleyTerry" in repr(model)
        assert "n_subjects=7" in repr(model)

    def test_initial_abilities_zero(self):
        model = BradleyTerry(n_subjects=4)
        assert torch.allclose(model.ability.data, torch.zeros(4))


class TestPredict:
    def test_shape(self):
        model = BradleyTerry(n_subjects=5)
        probs = model.predict()
        assert probs.shape == (5, 5)

    def test_diagonal_is_half(self):
        model = BradleyTerry(n_subjects=5)
        with torch.no_grad():
            model.ability.copy_(torch.randn(5))
        probs = model.predict()
        assert torch.allclose(probs.diag(), torch.full((5,), 0.5))

    def test_symmetry(self):
        """P(a > b) + P(b > a) = 1 for all pairs."""
        model = BradleyTerry(n_subjects=5)
        with torch.no_grad():
            model.ability.copy_(torch.randn(5))
        probs = model.predict()
        assert torch.allclose(probs + probs.T, torch.ones(5, 5), atol=1e-6)

    def test_higher_ability_higher_prob(self):
        """Subject with higher ability should have P > 0.5 against lower."""
        model = BradleyTerry(n_subjects=2)
        with torch.no_grad():
            model.ability.copy_(torch.tensor([2.0, -1.0]))
        probs = model.predict()
        assert probs[0, 1].item() > 0.5  # subject 0 beats subject 1
        assert probs[1, 0].item() < 0.5  # subject 1 loses to subject 0

    def test_equal_ability_gives_half(self):
        model = BradleyTerry(n_subjects=3)
        with torch.no_grad():
            model.ability.copy_(torch.tensor([1.0, 1.0, 1.0]))
        probs = model.predict()
        assert torch.allclose(probs, torch.full((3, 3), 0.5))


class TestPredictPairwise:
    def test_shape(self):
        model = BradleyTerry(n_subjects=5)
        a = torch.tensor([0, 1, 2])
        b = torch.tensor([1, 2, 3])
        probs = model.predict_pairwise(a, b)
        assert probs.shape == (3,)

    def test_matches_predict_matrix(self):
        """predict_pairwise results should match the corresponding entries in predict()."""
        model = BradleyTerry(n_subjects=4)
        with torch.no_grad():
            model.ability.copy_(torch.randn(4))
        a = torch.tensor([0, 1, 2, 0])
        b = torch.tensor([1, 2, 3, 3])
        pairwise_probs = model.predict_pairwise(a, b)
        matrix_probs = model.predict()
        for k in range(4):
            assert pairwise_probs[k].item() == pytest.approx(matrix_probs[a[k], b[k]].item(), abs=1e-6)


class TestForward:
    def test_forward_equals_predict(self):
        model = BradleyTerry(n_subjects=5)
        with torch.no_grad():
            model.ability.copy_(torch.randn(5))
        assert torch.allclose(model.forward(), model.predict())


class TestFitMLE:
    def test_fit_reduces_loss(self, small_pairwise_comparisons):
        model = BradleyTerry(n_subjects=small_pairwise_comparisons.n_subjects)
        history = model.fit(small_pairwise_comparisons, method="mle", max_epochs=200, verbose=False)
        losses = history["losses"]
        assert losses[-1] < losses[0]

    def test_fit_recovers_ranking(self, seed):
        """Fitted abilities should correlate strongly with ground-truth."""
        n_subjects = 6
        true_ability = torch.linspace(-2, 2, n_subjects)

        # Generate many comparisons for reliable recovery
        a_list, b_list, y_list = [], [], []
        for i in range(n_subjects):
            for j in range(i + 1, n_subjects):
                prob = torch.sigmoid(true_ability[i] - true_ability[j])
                for _ in range(50):  # 50 comparisons per pair
                    a_list.append(i)
                    b_list.append(j)
                    y_list.append(torch.bernoulli(prob).item())

        comparisons = PairwiseComparisons(
            subject_a=torch.tensor(a_list),
            subject_b=torch.tensor(b_list),
            outcome=torch.tensor(y_list),
            subject_ids=[f"s{i}" for i in range(n_subjects)],
        )

        model = BradleyTerry(n_subjects=n_subjects)
        model.fit(comparisons, method="mle", max_epochs=500, verbose=False)

        # Pearson correlation between fitted and true abilities should be high
        fitted = model.ability.data
        correlation = torch.corrcoef(torch.stack([true_ability, fitted]))[0, 1]
        assert correlation.item() > 0.9

    def test_fit_returns_history(self, small_pairwise_comparisons):
        model = BradleyTerry(n_subjects=small_pairwise_comparisons.n_subjects)
        history = model.fit(small_pairwise_comparisons, max_epochs=10, verbose=False)
        assert "losses" in history
        assert len(history["losses"]) <= 10
        assert all(isinstance(v, float) for v in history["losses"])


class TestFitJML:
    def test_fit_jml(self, small_pairwise_comparisons):
        model = BradleyTerry(n_subjects=small_pairwise_comparisons.n_subjects)
        history = model.fit(small_pairwise_comparisons, method="jml", max_epochs=100, verbose=False)
        losses = history["losses"]
        assert losses[-1] < losses[0]

    def test_unknown_method_raises(self, small_pairwise_comparisons):
        model = BradleyTerry(n_subjects=small_pairwise_comparisons.n_subjects)
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(small_pairwise_comparisons, method="em", verbose=False)
