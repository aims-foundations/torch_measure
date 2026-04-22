# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from torch_measure.models import ThreePL


class TestThreePL:
    def test_init(self):
        model = ThreePL(n_subjects=10, n_items=20)
        assert model.guessing.shape == (20,)
        assert (model.guessing >= 0).all()
        assert (model.guessing <= 1).all()

    def test_predict_lower_bound(self):
        """3PL probabilities should never go below guessing parameter."""
        model = ThreePL(n_subjects=5, n_items=10)
        probs = model.predict()
        # With very low ability, prob should approach guessing (not 0)
        with torch.no_grad():
            model.ability.fill_(-100.0)
        probs = model.predict()
        assert (probs >= model.guessing.unsqueeze(0) - 0.01).all()

    def test_fit_reduces_loss(self, small_response_matrix):
        model = ThreePL(n_subjects=20, n_items=30)
        history = model.fit(small_response_matrix, max_epochs=100, verbose=False)
        assert history["losses"][-1] < history["losses"][0]
