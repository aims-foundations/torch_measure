# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for the Testlet Rasch model."""

import pytest
import torch

from torch_measure.models.testlet import TestletRasch, build_testlet_map


class TestBuildTestletMap:
    def test_basic(self):
        item_ids = ["t1:0", "t1:1", "t2:0", "t2:1", "t2:2"]
        tmap, names = build_testlet_map(item_ids)
        assert tmap.tolist() == [0, 0, 1, 1, 1]
        assert names == ["t1", "t2"]

    def test_custom_separator(self):
        item_ids = ["t1/0", "t1/1", "t2/0"]
        tmap, names = build_testlet_map(item_ids, separator="/")
        assert tmap.tolist() == [0, 0, 1]
        assert names == ["t1", "t2"]

    def test_single_testlet(self):
        item_ids = ["t1:0", "t1:1", "t1:2"]
        tmap, names = build_testlet_map(item_ids)
        assert tmap.tolist() == [0, 0, 0]
        assert names == ["t1"]

    def test_preserves_order(self):
        item_ids = ["b:0", "a:0", "b:1"]
        tmap, names = build_testlet_map(item_ids)
        assert names == ["b", "a"]
        assert tmap.tolist() == [0, 1, 0]


def _make_testlet_map(n_items=30, n_testlets=6):
    """Create a testlet map with equal-sized testlets."""
    items_per_testlet = n_items // n_testlets
    return torch.repeat_interleave(torch.arange(n_testlets), items_per_testlet)


class TestTestletRasch:
    def test_init(self):
        tmap = _make_testlet_map()
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        assert model.n_subjects == 20
        assert model.n_items == 30
        assert model.n_testlets == 6
        assert model.ability.shape == (20,)
        assert model.difficulty.shape == (30,)
        assert model.testlet_effect.shape == (20, 6)

    def test_testlet_map_is_buffer(self):
        tmap = _make_testlet_map()
        model = TestletRasch(n_subjects=10, n_items=30, testlet_map=tmap)
        assert "testlet_map" in dict(model.named_buffers())
        assert "testlet_map" not in dict(model.named_parameters())

    def test_invalid_testlet_map_shape(self):
        tmap = torch.zeros(10, dtype=torch.long)  # wrong size
        with pytest.raises(ValueError):
            TestletRasch(n_subjects=5, n_items=30, testlet_map=tmap)

    def test_predict_shape(self):
        tmap = _make_testlet_map()
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        probs = model.predict()
        assert probs.shape == (20, 30)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_known_values(self):
        tmap = torch.tensor([0, 0, 1])
        model = TestletRasch(n_subjects=2, n_items=3, testlet_map=tmap)
        with torch.no_grad():
            model.ability.copy_(torch.tensor([2.0, -2.0]))
            model.difficulty.copy_(torch.tensor([0.0, 0.0, 0.0]))
            model.testlet_effect.zero_()
        probs = model.predict()
        # With zero testlet effects, matches standard Rasch
        assert probs[0, 0].item() > 0.8
        assert probs[1, 0].item() < 0.2

    def test_testlet_effect_modifies_probs(self):
        """Non-zero testlet effects should change probabilities for items in that testlet."""
        tmap = torch.tensor([0, 0, 1, 1])
        model = TestletRasch(n_subjects=2, n_items=4, testlet_map=tmap)
        with torch.no_grad():
            model.ability.zero_()
            model.difficulty.zero_()
            model.testlet_effect.zero_()
        probs_zero = model.predict().clone()

        with torch.no_grad():
            model.testlet_effect[0, 0] = 2.0
        probs_effect = model.predict()
        # Items in testlet 0 for subject 0 should have higher probability
        assert probs_effect[0, 0] > probs_zero[0, 0]
        assert probs_effect[0, 1] > probs_zero[0, 1]
        # Items in testlet 1 for subject 0 should be unchanged
        assert torch.allclose(probs_effect[0, 2], probs_zero[0, 2])
        assert torch.allclose(probs_effect[0, 3], probs_zero[0, 3])
        # Subject 1 should be unchanged entirely
        assert torch.allclose(probs_effect[1], probs_zero[1])

    def test_testlet_scale_property(self):
        tmap = _make_testlet_map()
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        scale = model.testlet_scale
        assert scale.shape == (6,)
        # Initially zero (testlet_effect initialized to zeros)
        assert torch.allclose(scale, torch.zeros(6))

    def test_forward_equals_predict(self):
        tmap = _make_testlet_map()
        model = TestletRasch(n_subjects=10, n_items=30, testlet_map=tmap)
        assert torch.allclose(model.forward(), model.predict())

    def test_fit_reduces_loss(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = model.fit(responses, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_fit_with_mask(self, small_testlet_response_matrix):
        responses, tmap = small_testlet_response_matrix
        mask = torch.ones_like(responses, dtype=torch.bool)
        mask[:5, :5] = False
        model = TestletRasch(n_subjects=20, n_items=30, testlet_map=tmap)
        history = model.fit(responses, mask=mask, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0
