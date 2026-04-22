# Copyright (c) 2026 AIMS Foundations. MIT License.

"""SVI fitting tests for the Testlet Rasch model."""

import pytest
import torch

from torch_measure.fitting.svi import svi_fit
from torch_measure.models.testlet import TestletRasch

pytest.importorskip("pyro", reason="pyro-ppl required for SVI tests")


class TestSVIFitTestletRasch:
    def test_reduces_loss(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        mask = torch.ones_like(responses, dtype=torch.bool)
        history = svi_fit(model, responses, mask, max_epochs=200, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_updates_parameters(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        ability_before = model.ability.clone()
        testlet_before = model.testlet_effect.clone()
        mask = torch.ones_like(responses, dtype=torch.bool)
        svi_fit(model, responses, mask, max_epochs=200, verbose=False)
        assert not torch.allclose(model.ability, ability_before)
        assert not torch.allclose(model.testlet_effect, testlet_before)

    def test_via_model_fit(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = model.fit(responses, method="svi", max_epochs=100, verbose=False)
        assert "losses" in history
        assert len(history["losses"]) == 100


class TestSVITestletPosterior:
    def test_posterior_keys(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = svi_fit(model, responses, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert "ability" in posterior
        assert "difficulty" in posterior
        assert "testlet_effect" in posterior
        assert "testlet_scale" in posterior

    def test_posterior_shapes(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = svi_fit(model, responses, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert posterior["ability"]["loc"].shape == (20,)
        assert posterior["ability"]["scale"].shape == (20,)
        assert posterior["difficulty"]["loc"].shape == (30,)
        assert posterior["difficulty"]["scale"].shape == (30,)
        assert posterior["testlet_effect"]["loc"].shape == (20, 6)
        assert posterior["testlet_effect"]["scale"].shape == (20, 6)
        assert posterior["testlet_scale"]["loc"].shape == (6,)
        assert posterior["testlet_scale"]["scale"].shape == (6,)

    def test_scales_positive(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = svi_fit(model, responses, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert (posterior["ability"]["scale"] > 0).all()
        assert (posterior["difficulty"]["scale"] > 0).all()
        assert (posterior["testlet_effect"]["scale"] > 0).all()
        assert (posterior["testlet_scale"]["scale"] > 0).all()

    def test_locs_match_model_params(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = svi_fit(model, responses, max_epochs=100, verbose=False)
        posterior = history["posterior"]
        assert torch.allclose(posterior["ability"]["loc"], model.ability.data)
        assert torch.allclose(posterior["difficulty"]["loc"], model.difficulty.data)
        assert torch.allclose(posterior["testlet_effect"]["loc"], model.testlet_effect.data)
