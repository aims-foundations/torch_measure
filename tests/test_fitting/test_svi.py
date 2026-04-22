# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for Stochastic Variational Inference fitting."""

import pytest
import torch

pytest.importorskip("pyro", reason="pyro-ppl required for SVI tests")

from torch_measure.fitting.svi import svi_fit
from torch_measure.models import BetaRasch, BetaTwoPL, Rasch, ThreePL, TwoPL


class TestSVIFitRasch:
    def test_reduces_loss(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = svi_fit(model, small_response_matrix, mask, max_epochs=200, verbose=False)
        assert len(history["losses"]) == 200
        # ELBO should decrease (loss goes down)
        assert history["losses"][-1] < history["losses"][0]

    def test_updates_parameters(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        ability_before = model.ability.clone()
        difficulty_before = model.difficulty.clone()
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        svi_fit(model, small_response_matrix, mask, max_epochs=100, verbose=False)
        assert not torch.allclose(model.ability, ability_before)
        assert not torch.allclose(model.difficulty, difficulty_before)

    def test_via_model_fit(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        history = model.fit(small_response_matrix, method="svi", max_epochs=100, verbose=False)
        assert "losses" in history
        assert len(history["losses"]) == 100


class TestSVIFitTwoPL:
    def test_reduces_loss(self, small_response_matrix):
        model = TwoPL(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = svi_fit(model, small_response_matrix, mask, max_epochs=200, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_updates_discrimination(self, small_response_matrix):
        model = TwoPL(n_subjects=20, n_items=30)
        disc_before = model._discrimination_raw.clone()
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        svi_fit(model, small_response_matrix, mask, max_epochs=200, verbose=False)
        assert not torch.allclose(model._discrimination_raw, disc_before)


class TestSVIFitThreePL:
    def test_reduces_loss(self, small_response_matrix):
        model = ThreePL(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        history = svi_fit(model, small_response_matrix, mask, max_epochs=200, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_updates_guessing(self, small_response_matrix):
        model = ThreePL(n_subjects=20, n_items=30)
        guess_before = model._guessing_raw.clone()
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        svi_fit(model, small_response_matrix, mask, max_epochs=200, verbose=False)
        assert not torch.allclose(model._guessing_raw, guess_before)


class TestSVIFitBeta:
    def test_beta_rasch_reduces_loss(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_beta_response_matrix, dtype=torch.bool)
        history = svi_fit(model, small_beta_response_matrix, mask, max_epochs=200, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_beta_twopl_reduces_loss(self, small_beta_response_matrix):
        model = BetaTwoPL(n_subjects=20, n_items=30)
        mask = torch.ones_like(small_beta_response_matrix, dtype=torch.bool)
        history = svi_fit(model, small_beta_response_matrix, mask, max_epochs=200, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_beta_via_model_fit(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = model.fit(
            small_beta_response_matrix, method="svi", max_epochs=100, verbose=False
        )
        assert "losses" in history
        assert len(history["losses"]) == 100


class TestSVIPosterior:
    """Tests for posterior extraction from SVI fitting."""

    def test_rasch_posterior_keys(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        history = svi_fit(model, small_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert "ability" in posterior
        assert "difficulty" in posterior
        assert "discrimination" not in posterior
        assert "guessing" not in posterior

    def test_twopl_posterior_keys(self, small_response_matrix):
        model = TwoPL(n_subjects=20, n_items=30)
        history = svi_fit(model, small_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert "ability" in posterior
        assert "difficulty" in posterior
        assert "discrimination" in posterior
        assert "guessing" not in posterior

    def test_threepl_posterior_keys(self, small_response_matrix):
        model = ThreePL(n_subjects=20, n_items=30)
        history = svi_fit(model, small_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert "ability" in posterior
        assert "difficulty" in posterior
        assert "discrimination" in posterior
        assert "guessing" in posterior

    def test_beta_rasch_posterior_keys(self, small_beta_response_matrix):
        model = BetaRasch(n_subjects=20, n_items=30)
        history = svi_fit(model, small_beta_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert "ability" in posterior
        assert "difficulty" in posterior

    def test_beta_twopl_posterior_keys(self, small_beta_response_matrix):
        model = BetaTwoPL(n_subjects=20, n_items=30)
        history = svi_fit(model, small_beta_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert "discrimination" in posterior

    def test_posterior_shapes(self, small_response_matrix):
        model = ThreePL(n_subjects=20, n_items=30)
        history = svi_fit(model, small_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert posterior["ability"]["loc"].shape == (20,)
        assert posterior["ability"]["scale"].shape == (20,)
        assert posterior["difficulty"]["loc"].shape == (30,)
        assert posterior["difficulty"]["scale"].shape == (30,)
        assert posterior["discrimination"]["loc"].shape == (30,)
        assert posterior["discrimination"]["scale"].shape == (30,)
        assert posterior["guessing"]["loc"].shape == (30,)
        assert posterior["guessing"]["scale"].shape == (30,)

    def test_scales_positive(self, small_response_matrix):
        model = TwoPL(n_subjects=20, n_items=30)
        history = svi_fit(model, small_response_matrix, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert (posterior["ability"]["scale"] > 0).all()
        assert (posterior["difficulty"]["scale"] > 0).all()
        assert (posterior["discrimination"]["scale"] > 0).all()

    def test_locs_match_model_params(self, small_response_matrix):
        model = Rasch(n_subjects=20, n_items=30)
        history = svi_fit(model, small_response_matrix, max_epochs=200, verbose=False)
        posterior = history["posterior"]
        assert torch.allclose(posterior["ability"]["loc"], model.ability.data)
        assert torch.allclose(posterior["difficulty"]["loc"], model.difficulty.data)


class TestSVIFitWithMask:
    def test_handles_missing_data(self, small_response_matrix):
        data = small_response_matrix.clone()
        data[:5, :5] = float("nan")
        model = Rasch(n_subjects=20, n_items=30)
        history = model.fit(data, method="svi", max_epochs=100, verbose=False)
        assert len(history["losses"]) == 100
