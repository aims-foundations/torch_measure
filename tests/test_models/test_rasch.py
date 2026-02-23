# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.models import Rasch


class TestRasch:
    def test_init(self):
        model = Rasch(n_subjects=10, n_items=20)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.ability.shape == (10,)
        assert model.difficulty.shape == (20,)

    def test_predict_shape(self):
        model = Rasch(n_subjects=10, n_items=20)
        probs = model.predict()
        assert probs.shape == (10, 20)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_known_values(self):
        model = Rasch(n_subjects=2, n_items=3)
        with torch.no_grad():
            model.ability.copy_(torch.tensor([2.0, -2.0]))
            model.difficulty.copy_(torch.tensor([0.0, 0.0, 0.0]))
        probs = model.predict()
        # High ability -> high probability
        assert probs[0, 0] > 0.8
        # Low ability -> low probability
        assert probs[1, 0] < 0.2

    def test_fit_reduces_loss(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        history = model.fit(small_response_matrix, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_fit_with_mask(self, small_response_matrix):
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        mask[:5, :5] = False  # hide some entries
        model = Rasch(n_subjects=20, n_items=30)
        history = model.fit(small_response_matrix, mask=mask, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0

    def test_forward_equals_predict(self):
        model = Rasch(n_subjects=10, n_items=20)
        assert torch.allclose(model.forward(), model.predict())
