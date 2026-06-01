# Copyright (c) 2026 AIMS Foundations. MIT License.

import pytest

from tests.conftest import to_long_triple
from torch_measure.fitting.em import em_fit
from torch_measure.models import BetaRasch, LogisticFM, Rasch


class TestEMFit:
    def test_returns_both_phases(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = em_fit(model, s_idx, i_idx, r, max_epochs=20, verbose=False)
        assert "losses_item" in history
        assert "losses_ability" in history
        assert len(history["losses_item"]) > 0
        assert len(history["losses_ability"]) > 0

    def test_item_loss_decreases(self, small_response_matrix):
        # Regression for GH issue #3: item phase marginalizes latent ability correctly.
        model = Rasch(n_subjects=20, n_items=30)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = em_fit(model, s_idx, i_idx, r, max_epochs=50, verbose=False)
        assert history["losses_item"][-1] < history["losses_item"][0]

    def test_ability_loss_decreases(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = em_fit(model, s_idx, i_idx, r, max_epochs=50, verbose=False)
        assert history["losses_ability"][-1] < history["losses_ability"][0]


class TestEMFitLogisticFM:
    def test_logisticfm_k1_fits(self, small_response_matrix):
        """LogisticFM with K=1 can be fit via em_fit."""
        model = LogisticFM(n_subjects=20, n_items=30, n_factors=1)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = em_fit(model, s_idx, i_idx, r, max_epochs=20, verbose=False)
        assert len(history["losses_item"]) > 0
        assert len(history["losses_ability"]) > 0

    def test_logisticfm_k1_loss_decreases(self, small_response_matrix):
        """Item-phase loss decreases for LogisticFM K=1 under em_fit."""
        model = LogisticFM(n_subjects=20, n_items=30, n_factors=1)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = em_fit(model, s_idx, i_idx, r, max_epochs=50, verbose=False)
        assert history["losses_item"][-1] < history["losses_item"][0]

    def test_logisticfm_multifactor_raises(self, small_response_matrix):
        """em_fit rejects LogisticFM with K>1 (no multi-dim quadrature yet)."""
        model = LogisticFM(n_subjects=20, n_items=30, n_factors=3)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        with pytest.raises(ValueError, match="1-dimensional"):
            em_fit(model, s_idx, i_idx, r, max_epochs=5, verbose=False)


class TestEMFitBeta:
    def test_beta_returns_both_phases(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(small_beta_response_matrix, method="em", max_epochs=20, verbose=False)
        assert "losses_item" in history
        assert "losses_ability" in history
        assert len(history["losses_item"]) > 0
        assert len(history["losses_ability"]) > 0
