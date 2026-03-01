# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch
import pytest

from torch_measure.models import GaussianGraphicalModel


class TestGaussianGraphicalModel:
    def test_init(self):
        model = GaussianGraphicalModel(n_items=8, lam=0.1)
        assert model.n_items == 8
        assert model.lam == 0.1
        assert model._L_raw.shape == (8, 8)

    def test_precision_positive_definite_at_init(self):
        model = GaussianGraphicalModel(n_items=6)
        K = model.precision
        eigenvalues = torch.linalg.eigvalsh(K)
        assert (eigenvalues > 0).all(), "precision matrix must be positive definite at init"

    def test_precision_symmetric(self):
        model = GaussianGraphicalModel(n_items=6)
        K = model.precision
        assert torch.allclose(K, K.T, atol=1e-6), "precision must be symmetric"

    def test_partial_correlations_diagonal_one(self):
        model = GaussianGraphicalModel(n_items=5)
        pcor = model.partial_correlations
        assert torch.allclose(pcor.diagonal(), torch.ones(5), atol=1e-6)

    def test_partial_correlations_symmetric(self):
        model = GaussianGraphicalModel(n_items=5)
        pcor = model.partial_correlations
        assert torch.allclose(pcor, pcor.T, atol=1e-6)

    def test_partial_correlations_range(self):
        model = GaussianGraphicalModel(n_items=5)
        pcor = model.partial_correlations
        assert (pcor >= -1 - 1e-6).all() and (pcor <= 1 + 1e-6).all()

    def test_adjacency_zero_diagonal(self):
        model = GaussianGraphicalModel(n_items=5)
        W = model.adjacency
        assert torch.allclose(W.diagonal(), torch.zeros(5), atol=1e-6)

    def test_adjacency_symmetric(self):
        model = GaussianGraphicalModel(n_items=5)
        W = model.adjacency
        assert torch.allclose(W, W.T, atol=1e-6)

    def test_fit_reduces_loss(self, small_beta_response_matrix):
        model = GaussianGraphicalModel(n_items=30, lam=0.05)
        history = model.fit(small_beta_response_matrix, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0], "loss should decrease"

    def test_fit_continuous_data(self):
        torch.manual_seed(42)
        X = torch.randn(50, 10)
        model = GaussianGraphicalModel(n_items=10, lam=0.1)
        history = model.fit(X, max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_precision_positive_definite_after_fit(self):
        torch.manual_seed(0)
        X = torch.randn(40, 8)
        model = GaussianGraphicalModel(n_items=8, lam=0.1)
        model.fit(X, max_epochs=100, verbose=False)
        K = model.precision
        eigenvalues = torch.linalg.eigvalsh(K)
        assert (eigenvalues > 0).all(), "precision must remain PD after fitting"

    def test_fit_with_mask(self, small_beta_response_matrix):
        mask = torch.ones_like(small_beta_response_matrix, dtype=torch.bool)
        mask[:5, :5] = False
        model = GaussianGraphicalModel(n_items=30, lam=0.1)
        history = model.fit(small_beta_response_matrix, mask=mask, max_epochs=30, verbose=False)
        assert len(history["losses"]) > 0

    def test_fit_missing_nan(self):
        X = torch.randn(30, 8)
        X[0, 0] = float("nan")
        model = GaussianGraphicalModel(n_items=8, lam=0.1)
        history = model.fit(X, max_epochs=20, verbose=False)
        assert len(history["losses"]) > 0

    def test_lam_override_in_fit(self):
        torch.manual_seed(1)
        X = torch.randn(30, 5)
        model = GaussianGraphicalModel(n_items=5, lam=0.0)
        # Pass a different lam in fit — should not raise
        history = model.fit(X, max_epochs=30, verbose=False, lam=0.5)
        assert len(history["losses"]) > 0

    def test_centrality_strength(self):
        torch.manual_seed(0)
        X = torch.randn(40, 6)
        model = GaussianGraphicalModel(n_items=6, lam=0.1)
        model.fit(X, max_epochs=50, verbose=False)
        s = model.centrality("strength")
        assert s.shape == (6,)
        assert (s >= 0).all()

    def test_centrality_closeness(self):
        torch.manual_seed(0)
        X = torch.randn(40, 6)
        model = GaussianGraphicalModel(n_items=6, lam=0.01)
        model.fit(X, max_epochs=50, verbose=False)
        c = model.centrality("closeness")
        assert c.shape == (6,)

    def test_sparser_network_with_higher_lam(self):
        """Higher λ should produce fewer strong edges."""
        torch.manual_seed(5)
        X = torch.randn(60, 10)

        model_dense = GaussianGraphicalModel(n_items=10, lam=0.001)
        model_dense.fit(X, max_epochs=200, verbose=False)

        model_sparse = GaussianGraphicalModel(n_items=10, lam=1.0)
        model_sparse.fit(X, max_epochs=200, verbose=False)

        dense_strength = model_dense.centrality("strength").mean().item()
        sparse_strength = model_sparse.centrality("strength").mean().item()
        assert sparse_strength < dense_strength, "higher λ should produce a sparser network"

    def test_device_cpu(self):
        model = GaussianGraphicalModel(n_items=4, device="cpu")
        assert model._L_raw.device.type == "cpu"
