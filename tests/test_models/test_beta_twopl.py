# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Tests for BetaTwoPL model."""

import torch

from torch_measure.models import BetaTwoPL, TwoPL
from torch_measure.models._base import IRTModel


class TestBetaTwoPL:
    def test_init(self):
        model = BetaTwoPL(n_subjects=10, n_items=20)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.discrimination.shape == (20,)
        assert (model.discrimination > 0).all()
        assert model.phi == 10.0

    def test_init_custom_phi(self):
        model = BetaTwoPL(n_subjects=10, n_items=20, phi=25.0)
        assert model.phi == 25.0

    def test_predict_shape(self):
        model = BetaTwoPL(n_subjects=10, n_items=20)
        probs = model.predict()
        assert probs.shape == (10, 20)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_identical_to_twopl(self):
        """BetaTwoPL.predict() should produce the same output as TwoPL.predict()."""
        twopl = TwoPL(n_subjects=10, n_items=20)
        beta_twopl = BetaTwoPL(n_subjects=10, n_items=20)
        with torch.no_grad():
            beta_twopl.ability.copy_(twopl.ability)
            beta_twopl.difficulty.copy_(twopl.difficulty)
            beta_twopl._discrimination_raw.copy_(twopl._discrimination_raw)
        assert torch.allclose(twopl.predict(), beta_twopl.predict())

    def test_fit_reduces_loss(self, small_beta_response_matrix):
        model = BetaTwoPL(n_subjects=20, n_items=30)
        history = model.fit(small_beta_response_matrix, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_discrimination_positive(self):
        model = BetaTwoPL(n_subjects=5, n_items=10)
        assert (model.discrimination > 0).all()

    def test_isinstance_hierarchy(self):
        """BetaTwoPL should be a subclass of TwoPL and IRTModel."""
        model = BetaTwoPL(n_subjects=5, n_items=10)
        assert isinstance(model, TwoPL)
        assert isinstance(model, IRTModel)
