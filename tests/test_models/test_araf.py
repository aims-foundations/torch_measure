# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for the ARAF (Amortized Response/Agent Factor) model."""

import pytest
import torch

from torch_measure.models import ARAF
from torch_measure.models._predictor import predict_dense


def _synthetic_low_rank_dataset(
    n_subjects: int,
    n_items: int,
    embedding_dim: int,
    true_rank: int,
    binary: bool,
    seed: int = 0,
):
    """Build a synthetic dataset whose true factor rank is ``true_rank``."""
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(n_items, embedding_dim, generator=g)
    W_true = torch.randn(true_rank, embedding_dim, generator=g) * 0.5
    theta_true = torch.randn(n_subjects, true_rank, generator=g)
    diff = torch.randn(n_items, generator=g) * 0.3

    logits = theta_true @ (embeddings @ W_true.T).T + diff.unsqueeze(0)
    probs = torch.sigmoid(logits)
    responses = torch.bernoulli(probs, generator=g) if binary else probs.clamp(1e-3, 1 - 1e-3)
    mask = torch.ones_like(responses, dtype=torch.bool)
    return embeddings, responses, mask, probs


class TestARAFConstruction:
    def test_init_shapes(self):
        m = ARAF(n_subjects=10, n_items=20, embedding_dim=8, latent_dim=4)
        assert m.theta.shape == (10, 4)
        assert m.theta_bias.shape == (10,)
        assert m.W.shape == (4, 8)
        assert m.tau_raw.shape == (4,)
        assert m.global_bias.shape == (1,)

    def test_set_embeddings_validates_shape(self):
        m = ARAF(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=2)
        with pytest.raises(ValueError):
            m.set_embeddings(torch.randn(9, 8))
        with pytest.raises(ValueError):
            m.set_embeddings(torch.randn(10, 7))
        with pytest.raises(ValueError):
            m.set_embeddings(torch.randn(10))

    def test_predict_requires_embeddings(self):
        m = ARAF(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=2)
        with pytest.raises(RuntimeError):
            m.dense_predict()


class TestARAFForward:
    def test_dense_predict_shape_and_range(self):
        m = ARAF(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        m.set_embeddings(torch.randn(10, 8))
        m.eval()
        probs = m.dense_predict()
        assert probs.shape == (5, 10)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_matches_dense_gather(self):
        m = ARAF(n_subjects=4, n_items=6, embedding_dim=8, latent_dim=3)
        m.set_embeddings(torch.randn(6, 8))
        m.eval()

        dense = m.dense_predict()
        s_idx = torch.tensor([0, 0, 1, 2, 3, 3])
        i_idx = torch.tensor([0, 5, 2, 3, 1, 4])
        gathered = dense[s_idx, i_idx]
        per_row = m.predict({"subject_idx": s_idx, "item_idx": i_idx})
        torch.testing.assert_close(gathered, per_row)

    def test_predict_dense_helper_reshapes(self):
        m = ARAF(n_subjects=3, n_items=5, embedding_dim=4, latent_dim=2)
        m.set_embeddings(torch.randn(5, 4))
        m.eval()
        out = predict_dense(m)
        assert out.shape == (3, 5)


class TestARAFFit:
    def test_bernoulli_fit_reduces_loss(self):
        n_subjects, n_items, d, true_rank = 20, 25, 8, 3
        emb, y, mask, _ = _synthetic_low_rank_dataset(
            n_subjects, n_items, d, true_rank, binary=True, seed=0
        )

        m = ARAF(n_subjects=n_subjects, n_items=n_items, embedding_dim=d, latent_dim=10, dropout=0.0)
        history = m.fit(
            y,
            embeddings=emb,
            mask=mask,
            likelihood="bernoulli",
            epochs=120,
            lambda_tau=0.5,
            verbose=False,
        )
        assert history["losses"][-1] < history["losses"][0]

    def test_beta_fit_reduces_loss(self):
        n_subjects, n_items, d, true_rank = 20, 25, 8, 3
        emb, y, mask, _ = _synthetic_low_rank_dataset(
            n_subjects, n_items, d, true_rank, binary=False, seed=1
        )

        m = ARAF(n_subjects=n_subjects, n_items=n_items, embedding_dim=d, latent_dim=10, dropout=0.0)
        history = m.fit(
            y,
            embeddings=emb,
            mask=mask,
            likelihood="beta",
            beta_phi=20.0,
            epochs=120,
            lambda_tau=0.5,
            verbose=False,
        )
        assert history["losses"][-1] < history["losses"][0]

    def test_unknown_likelihood_raises(self):
        m = ARAF(n_subjects=4, n_items=6, embedding_dim=4, latent_dim=2)
        emb = torch.randn(6, 4)
        y = torch.bernoulli(torch.full((4, 6), 0.5))
        mask = torch.ones_like(y, dtype=torch.bool)
        with pytest.raises(ValueError):
            m.fit(y, embeddings=emb, mask=mask, likelihood="poisson", epochs=2, verbose=False)


class TestARAFArd:
    def test_ard_prunes_inactive_dims(self):
        n_subjects, n_items, d, true_rank = 30, 40, 8, 2
        emb, y, mask, _ = _synthetic_low_rank_dataset(
            n_subjects, n_items, d, true_rank, binary=True, seed=2
        )

        m = ARAF(
            n_subjects=n_subjects,
            n_items=n_items,
            embedding_dim=d,
            latent_dim=12,
            dropout=0.0,
        )
        m.fit(
            y,
            embeddings=emb,
            mask=mask,
            likelihood="bernoulli",
            epochs=600,
            lambda_tau=2.0,
            verbose=False,
        )
        n_active = m.active_dims.numel()
        assert n_active < 12, f"ARD did not prune any dims (active={n_active})"

    def test_no_ard_keeps_all_dims(self):
        m = ARAF(
            n_subjects=6,
            n_items=8,
            embedding_dim=4,
            latent_dim=3,
            dropout=0.0,
            use_ard=False,
        )
        torch.testing.assert_close(m.get_tau(), torch.ones(3))


class TestARAFAdapt:
    def test_adapt_freezes_item_params(self):
        n_subjects, n_items, d = 12, 15, 6
        emb, y, mask, _ = _synthetic_low_rank_dataset(
            n_subjects, n_items, d, true_rank=2, binary=True, seed=3
        )

        m = ARAF(n_subjects=n_subjects, n_items=n_items, embedding_dim=d, latent_dim=4, dropout=0.0)
        m.fit(y, embeddings=emb, mask=mask, likelihood="bernoulli", epochs=80, verbose=False)

        W_before = m.W.detach().clone()
        diff_w_before = m.difficulty_proj.weight.detach().clone()
        tau_before = m.tau_raw.detach().clone()

        m.adapt(y, mask=mask, likelihood="bernoulli", epochs=30, verbose=False)

        torch.testing.assert_close(m.W.detach(), W_before)
        torch.testing.assert_close(m.difficulty_proj.weight.detach(), diff_w_before)
        torch.testing.assert_close(m.tau_raw.detach(), tau_before)

    def test_adapt_restores_grad(self):
        m = ARAF(n_subjects=6, n_items=8, embedding_dim=4, latent_dim=2, dropout=0.0)
        emb = torch.randn(8, 4)
        y = torch.bernoulli(torch.full((6, 8), 0.5))
        mask = torch.ones_like(y, dtype=torch.bool)
        m.fit(y, embeddings=emb, mask=mask, epochs=20, verbose=False)
        m.adapt(y, mask=mask, epochs=10, verbose=False)
        assert all(p.requires_grad for p in m.parameters())
