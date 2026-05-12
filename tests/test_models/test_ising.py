# Copyright (c) 2026 AIMS Foundations. MIT License.

import pytest
import torch

from torch_measure.models import IsingModel


class TestIsingModel:
    def test_init(self):
        model = IsingModel(n_items=10)
        assert model.n_items == 10
        assert model.thresholds.shape == (10,)
        assert model._weights_raw.shape == (10, 10)

    def test_adjacency_symmetric(self):
        model = IsingModel(n_items=8)
        W = model.adjacency
        assert W.shape == (8, 8)
        assert torch.allclose(W, W.T), "adjacency must be symmetric"
        assert torch.allclose(W.diagonal(), torch.zeros(8)), "diagonal must be zero"

    def test_adjacency_stays_symmetric_after_fit(self, small_response_matrix):
        model = IsingModel(n_items=30)
        model.fit(small_response_matrix, max_epochs=20, verbose=False)
        W = model.adjacency
        assert torch.allclose(W, W.T, atol=1e-6), "adjacency must remain symmetric after fitting"
        assert torch.allclose(W.diagonal(), torch.zeros(30), atol=1e-6), "diagonal must remain zero"

    def test_fit_reduces_loss(self, small_response_matrix):
        model = IsingModel(n_items=30)
        history = model.fit(small_response_matrix, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0], "loss should decrease"

    def test_fit_with_mask(self, small_response_matrix):
        mask = torch.ones_like(small_response_matrix, dtype=torch.bool)
        mask[:5, :5] = False
        model = IsingModel(n_items=30)
        history = model.fit(small_response_matrix, mask=mask, max_epochs=30, verbose=False)
        assert len(history["losses"]) > 0

    def test_fit_missing_nan(self):
        data = torch.bernoulli(torch.full((20, 10), 0.6))
        data[0, 0] = float("nan")
        model = IsingModel(n_items=10)
        history = model.fit(data, max_epochs=20, verbose=False)
        assert len(history["losses"]) > 0

    def test_conditional_probs_shape(self, small_response_matrix):
        model = IsingModel(n_items=30)
        probs = model.conditional_probs(small_response_matrix)
        assert probs.shape == small_response_matrix.shape
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_conditional_probs_binary_range(self):
        model = IsingModel(n_items=5)
        X = torch.bernoulli(torch.full((10, 5), 0.5))
        probs = model.conditional_probs(X)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_centrality_strength(self, small_response_matrix):
        model = IsingModel(n_items=30)
        model.fit(small_response_matrix, max_epochs=50, verbose=False)
        s = model.centrality("strength")
        assert s.shape == (30,)
        assert (s >= 0).all(), "strength centrality must be non-negative"

    def test_centrality_expected_influence(self, small_response_matrix):
        model = IsingModel(n_items=30)
        model.fit(small_response_matrix, max_epochs=50, verbose=False)
        ei = model.centrality("expected_influence")
        assert ei.shape == (30,)

    def test_centrality_invalid_raises(self, small_response_matrix):
        model = IsingModel(n_items=30)
        with pytest.raises(ValueError, match="Unknown centrality"):
            model.centrality("foobar")

    def test_known_positive_association(self):
        """Items with perfect positive correlation should have a positive edge."""
        torch.manual_seed(0)
        n = 50
        # Generate two perfectly correlated binary columns
        base = torch.bernoulli(torch.full((n,), 0.5))
        X = torch.stack([base, base, torch.bernoulli(torch.full((n,), 0.5))], dim=1)  # (n, 3)
        model = IsingModel(n_items=3)
        model.fit(X, max_epochs=300, lr=0.05, verbose=False)
        W = model.adjacency
        # Items 0 and 1 are identical → their edge weight should be positive
        assert W[0, 1].item() > 0.0, "perfectly correlated items should have a positive edge"

    def test_device_cpu(self):
        model = IsingModel(n_items=5, device="cpu")
        assert model.thresholds.device.type == "cpu"
