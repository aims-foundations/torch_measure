# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.fitting.em import em_fit
from torch_measure.models import BetaRasch, Rasch


class TestEMFit:
    def test_returns_both_phases(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = em_fit(model, small_response_matrix, mask, max_epochs=20, verbose=False)
        assert "losses_item" in history
        assert "losses_ability" in history
        assert len(history["losses_item"]) > 0
        assert len(history["losses_ability"]) > 0

    def test_item_loss_decreases(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = em_fit(model, small_response_matrix, mask, max_epochs=50, verbose=False)
        assert history["losses_item"][-1] < history["losses_item"][0]

    def test_ability_loss_decreases(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = em_fit(model, small_response_matrix, mask, max_epochs=50, verbose=False)
        assert history["losses_ability"][-1] < history["losses_ability"][0]


class TestEMFitBeta:
    def test_beta_returns_both_phases(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(
            small_beta_response_matrix, method="em", max_epochs=20, verbose=False
        )
        assert "losses_item" in history
        assert "losses_ability" in history
        assert len(history["losses_item"]) > 0
        assert len(history["losses_ability"]) > 0
