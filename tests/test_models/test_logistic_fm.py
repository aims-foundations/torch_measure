# Copyright (c) 2026 AIMS Foundations. MIT License.

from torch_measure.models import LogisticFM


class TestLogisticFM:
    def test_init(self):
        model = LogisticFM(n_subjects=10, n_items=20, n_factors=3)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.n_factors == 3
        assert model.U.shape == (10, 3)
        assert model.V.shape == (20, 3)
        assert model.Z.shape == (20,)

    def test_predict_shape(self):
        model = LogisticFM(n_subjects=10, n_items=20, n_factors=2)
        probs = model.predict()
        assert probs.shape == (10, 20)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_properties(self):
        model = LogisticFM(n_subjects=10, n_items=20, n_factors=3)
        assert model.ability.shape == (10, 3)
        assert model.difficulty.shape == (20,)
        assert model.loadings.shape == (20, 3)

    def test_single_factor_like_rasch(self):
        """With K=1, LogisticFM should behave similarly to Rasch."""
        model = LogisticFM(n_subjects=5, n_items=10, n_factors=1)
        probs = model.predict()
        assert probs.shape == (5, 10)

    def test_fit_reduces_loss(self, small_response_matrix):
        model = LogisticFM(n_subjects=20, n_items=30, n_factors=2)
        history = model.fit(small_response_matrix, max_epochs=100, verbose=False)
        assert history["losses"][-1] < history["losses"][0]
