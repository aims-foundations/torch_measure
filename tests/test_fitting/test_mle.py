# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from tests.conftest import to_long_triple
from torch_measure.fitting.mle import mle_fit
from torch_measure.models import BetaRasch, Rasch


class TestMLEFit:
    def test_reduces_loss(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = mle_fit(model, s_idx, i_idx, r, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_lbfgs_optimizer(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = mle_fit(model, s_idx, i_idx, r, max_epochs=50, verbose=False, optimizer_cls="lbfgs")
        assert len(history["losses"]) > 0

    def test_convergence_stops_early(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        s_idx, i_idx, r = to_long_triple(small_response_matrix)
        history = mle_fit(
            model, s_idx, i_idx, r, max_epochs=10000, convergence_tol=1e-4, verbose=False
        )
        # Should stop well before 10000 epochs
        assert len(history["losses"]) < 10000

    def test_with_mask(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        mask[:5, :5] = False
        s_idx, i_idx, r = to_long_triple(small_response_matrix, mask)
        history = mle_fit(model, s_idx, i_idx, r, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0


class TestMLEFitBeta:
    def test_beta_reduces_loss(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(small_beta_response_matrix, max_epochs=100, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_beta_lbfgs(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(
            small_beta_response_matrix, max_epochs=50, verbose=False, optimizer_cls="lbfgs"
        )
        assert len(history["losses"]) > 0
