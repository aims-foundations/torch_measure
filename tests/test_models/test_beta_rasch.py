# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for BetaRasch model."""

import torch

from torch_measure.models import BetaRasch, Rasch
from torch_measure.models._base import IRTModel


class TestBetaRasch:
    def test_init(self):
        model = BetaRasch(n_subjects=10, n_items=20)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.ability.shape == (10,)
        assert model.difficulty.shape == (20,)
        assert model.phi == 10.0

    def test_init_custom_phi(self):
        model = BetaRasch(n_subjects=10, n_items=20, phi=5.0)
        assert model.phi == 5.0

    def test_predict_shape(self):
        model = BetaRasch(n_subjects=10, n_items=20)
        probs = model.predict()
        assert probs.shape == (10, 20)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_known_values(self):
        model = BetaRasch(n_subjects=2, n_items=3)
        with torch.no_grad():
            model.ability.copy_(torch.tensor([2.0, -2.0]))
            model.difficulty.copy_(torch.tensor([0.0, 0.0, 0.0]))
        probs = model.predict()
        assert probs[0, 0] > 0.8
        assert probs[1, 0] < 0.2

    def test_predict_identical_to_rasch(self):
        """BetaRasch.predict() should produce the same output as Rasch.predict()."""
        rasch = Rasch(n_subjects=10, n_items=20)
        beta_rasch = BetaRasch(n_subjects=10, n_items=20)
        with torch.no_grad():
            beta_rasch.ability.copy_(rasch.ability)
            beta_rasch.difficulty.copy_(rasch.difficulty)
        assert torch.allclose(rasch.predict(), beta_rasch.predict())

    def test_fit_reduces_loss(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(small_beta_response_matrix, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_fit_with_mask(self, small_beta_response_matrix):
        mask = torch.ones_like(small_beta_response_matrix, dtype=torch.bool)
        mask[:5, :5] = False
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(small_beta_response_matrix, mask=mask, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0

    def test_forward_equals_predict(self):
        model = BetaRasch(n_subjects=10, n_items=20)
        assert torch.allclose(model.forward(), model.predict())

    def test_isinstance_hierarchy(self):
        """BetaRasch should be a subclass of Rasch and IRTModel."""
        model = BetaRasch(n_subjects=5, n_items=10)
        assert isinstance(model, Rasch)
        assert isinstance(model, IRTModel)
