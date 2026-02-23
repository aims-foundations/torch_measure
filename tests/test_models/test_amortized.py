# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.models import AmortizedIRT


class TestAmortizedIRT:
    def test_init(self):
        model = AmortizedIRT(n_subjects=10, n_items=20, embedding_dim=64)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.embedding_dim == 64
        assert model.ability.shape == (10,)

    def test_set_embeddings(self):
        model = AmortizedIRT(n_subjects=5, n_items=10, embedding_dim=32)
        embeddings = torch.randn(10, 32)
        model.set_embeddings(embeddings)
        assert model._embeddings is not None
        assert model._embeddings.shape == (10, 32)

    def test_set_embeddings_wrong_size(self):
        model = AmortizedIRT(n_subjects=5, n_items=10, embedding_dim=32)
        embeddings = torch.randn(5, 32)  # wrong n_items
        try:
            model.set_embeddings(embeddings)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_predict_shape(self):
        model = AmortizedIRT(n_subjects=5, n_items=10, embedding_dim=32, pl=2)
        embeddings = torch.randn(10, 32)
        model.set_embeddings(embeddings)
        probs = model.predict()
        assert probs.shape == (5, 10)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_requires_embeddings(self):
        model = AmortizedIRT(n_subjects=5, n_items=10, embedding_dim=32)
        try:
            model.predict()
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_pl_modes(self):
        """Test 1PL, 2PL, and 3PL amortized modes."""
        embeddings = torch.randn(10, 32)
        for pl in [1, 2, 3]:
            model = AmortizedIRT(n_subjects=5, n_items=10, embedding_dim=32, pl=pl)
            model.set_embeddings(embeddings)
            probs = model.predict()
            assert probs.shape == (5, 10)

            if pl >= 2:
                assert model.discrimination is not None
            else:
                assert model.discrimination is None

            if pl == 3:
                assert model.guessing is not None
            else:
                assert model.guessing is None

    def test_fit_reduces_loss(self, small_response_matrix):
        model = AmortizedIRT(n_subjects=20, n_items=30, embedding_dim=16, pl=2)
        embeddings = torch.randn(30, 16)
        history = model.fit(small_response_matrix, embeddings, max_epochs=50, verbose=False)
        assert history["losses"][-1] < history["losses"][0]
