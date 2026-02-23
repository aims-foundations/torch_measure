# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.models import TwoPL


class TestTwoPL:
    def test_init(self):
        model = TwoPL(n_subjects=10, n_items=20)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.discrimination.shape == (20,)
        assert (model.discrimination > 0).all()

    def test_predict_shape(self):
        model = TwoPL(n_subjects=10, n_items=20)
        probs = model.predict()
        assert probs.shape == (10, 20)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_discrimination_positive(self):
        model = TwoPL(n_subjects=5, n_items=10)
        assert (model.discrimination > 0).all()

    def test_fit_reduces_loss(self, small_response_matrix):
        model = TwoPL(n_subjects=20, n_items=30)
        history = model.fit(small_response_matrix, max_epochs=100, verbose=False)
        assert history["losses"][-1] < history["losses"][0]
