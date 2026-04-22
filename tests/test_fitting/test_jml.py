# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from torch_measure.fitting.jml import jml_fit
from torch_measure.models import BetaRasch, Rasch


class TestJMLFit:
    def test_reduces_loss(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = jml_fit(model, small_response_matrix, mask, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_with_regularization(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = jml_fit(model, small_response_matrix, mask, max_epochs=50, regularization=0.1, verbose=False)
        assert len(history["losses"]) > 0


class TestJMLFitBeta:
    def test_beta_reduces_loss(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(small_beta_response_matrix, method="jml", max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]
