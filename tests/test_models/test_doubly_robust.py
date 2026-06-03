# Copyright (c) 2026 AIMS Foundations. MIT License.

import numpy as np
import torch

from torch_measure.models import DoublyRobustModel, Rasch
from torch_measure.models._predictor import predict_dense


def _make_sparse_rasch(
    n_subjects: int = 40,
    n_items: int = 30,
    obs_rate: float = 0.6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a Rasch response matrix with MNAR missingness."""
    rng = torch.Generator().manual_seed(seed)
    ability = torch.randn(n_subjects, generator=rng)
    difficulty = torch.randn(n_items, generator=rng)
    logits = ability.unsqueeze(1) - difficulty.unsqueeze(0)
    probs = torch.sigmoid(logits)
    data_full = torch.bernoulli(probs, generator=rng)

    # MNAR: high-ability subjects less likely observed on easy items
    obs_logits = -0.4 * ability.unsqueeze(1) + 0.4 * difficulty.unsqueeze(0)
    obs_probs = torch.sigmoid(obs_logits).clamp(0.2, 0.95)
    obs_mask = torch.bernoulli(obs_probs, generator=rng).bool()
    # guarantee at least one obs per subject
    for i in range(n_subjects):
        if not obs_mask[i].any():
            obs_mask[i, 0] = True

    data_sparse = data_full.clone()
    data_sparse[~obs_mask] = float("nan")
    return data_sparse, data_full, ability, difficulty


class TestDoublyRobustModel:

    def test_init_freezes_base(self):
        base = Rasch(n_subjects=10, n_items=20)
        dr = DoublyRobustModel(base)
        for p in dr._base.parameters():
            assert not p.requires_grad

    def test_init_correction_shape(self):
        base = Rasch(n_subjects=15, n_items=25)
        dr = DoublyRobustModel(base)
        assert dr.correction_ability.shape == (15,)
        assert dr.correction_difficulty.shape == (25,)
        assert dr.n_subjects == 15
        assert dr.n_items == 25

    def test_predict_shape_and_range(self):
        base = Rasch(n_subjects=10, n_items=20)
        dr = DoublyRobustModel(base)
        probs = predict_dense(dr)
        assert probs.shape == (10, 20)
        assert (probs > 0).all()
        assert (probs < 1).all()

    def test_zero_correction_matches_base(self):
        base = Rasch(n_subjects=10, n_items=20)
        dr = DoublyRobustModel(base)
        # correction params initialized to zero → correction = sigmoid(0) - 0.5 = 0
        base_probs = predict_dense(base)
        dr_probs = predict_dense(dr)
        torch.testing.assert_close(dr_probs, base_probs, atol=1e-6, rtol=0)

    def test_fit_reduces_loss(self):
        data_sparse, _, _, _ = _make_sparse_rasch(30, 20, seed=5)
        base = Rasch(n_subjects=30, n_items=20)
        base.fit(data_sparse, max_epochs=50, verbose=False)

        dr = DoublyRobustModel(base)
        history = dr.fit(data_sparse, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 1
        assert history["losses"][-1] < history["losses"][0]

    def test_fit_changes_correction_params(self):
        data_sparse, _, _, _ = _make_sparse_rasch(30, 20, seed=7)
        base = Rasch(n_subjects=30, n_items=20)
        base.fit(data_sparse, max_epochs=50, verbose=False)

        dr = DoublyRobustModel(base)
        before_ability = dr.correction_ability.detach().clone()
        dr.fit(data_sparse, max_epochs=50, verbose=False)
        assert not torch.allclose(dr.correction_ability, before_ability)

    def test_base_params_unchanged_after_fit(self):
        data_sparse, _, _, _ = _make_sparse_rasch(30, 20, seed=9)
        base = Rasch(n_subjects=30, n_items=20)
        base.fit(data_sparse, max_epochs=50, verbose=False)

        ability_before = base.ability.detach().clone()
        difficulty_before = base.difficulty.detach().clone()

        dr = DoublyRobustModel(base)
        dr.fit(data_sparse, max_epochs=50, verbose=False)

        torch.testing.assert_close(base.ability, ability_before)
        torch.testing.assert_close(base.difficulty, difficulty_before)

    def test_improves_prediction_on_sparse_data(self):
        """DR model should predict held-out cells better than base alone."""
        torch.manual_seed(42)
        data_sparse, data_full, ability, difficulty = _make_sparse_rasch(
            n_subjects=60, n_items=40, seed=11
        )

        base = Rasch(n_subjects=60, n_items=40)
        base.fit(data_sparse, max_epochs=200, verbose=False)

        dr = DoublyRobustModel(base)
        dr.fit(data_sparse, max_epochs=200, verbose=False)

        # Evaluate on all cells
        base_preds = predict_dense(base).detach()
        dr_preds = predict_dense(dr).detach()

        base_mse = ((base_preds - data_full) ** 2).mean().item()
        dr_mse = ((dr_preds - data_full) ** 2).mean().item()

        # DR should not be substantially worse
        assert dr_mse < base_mse + 0.02, (
            f"DR MSE {dr_mse:.4f} much worse than base {base_mse:.4f}"
        )

    def test_propensity_clipping(self):
        data_sparse, _, _, _ = _make_sparse_rasch(20, 15, seed=13)
        base = Rasch(n_subjects=20, n_items=15)
        base.fit(data_sparse, max_epochs=30, verbose=False)

        dr = DoublyRobustModel(base, clip_propensity=(0.1, 0.9))
        dr.fit(data_sparse, max_epochs=30, verbose=False)

        # Should not produce NaN/Inf
        preds = predict_dense(dr)
        assert torch.isfinite(preds).all()

    def test_complete_data_correction_near_zero(self):
        """On fully observed data, correction should stay small."""
        torch.manual_seed(99)
        n_s, n_i = 20, 15
        ability = torch.randn(n_s)
        difficulty = torch.randn(n_i)
        logits = ability.unsqueeze(1) - difficulty.unsqueeze(0)
        data = torch.bernoulli(torch.sigmoid(logits))

        base = Rasch(n_subjects=n_s, n_items=n_i)
        base.fit(data, max_epochs=100, verbose=False)

        dr = DoublyRobustModel(base)
        dr.fit(data, max_epochs=100, verbose=False)

        # Correction params should remain near zero since no missingness bias
        assert dr.correction_ability.abs().mean().item() < 0.5
        assert dr.correction_difficulty.abs().mean().item() < 0.5

    def test_forward_equals_predict(self):
        base = Rasch(n_subjects=10, n_items=20)
        dr = DoublyRobustModel(base)
        from torch_measure.models._predictor import cartesian_query
        query = cartesian_query(10, 20)
        torch.testing.assert_close(dr(query), dr.predict(query))
