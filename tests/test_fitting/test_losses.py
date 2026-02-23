# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Tests for loss functions."""

import torch

from torch_measure.fitting._losses import bernoulli_nll, beta_nll


class TestBernoulliNLL:
    def test_perfect_prediction(self):
        """Loss should be low when predictions match observations."""
        probs = torch.tensor([0.99, 0.01, 0.99])
        obs = torch.tensor([1.0, 0.0, 1.0])
        loss = bernoulli_nll(probs, obs)
        assert loss.item() < 0.05

    def test_bad_prediction(self):
        """Loss should be high when predictions are wrong."""
        probs = torch.tensor([0.01, 0.99, 0.01])
        obs = torch.tensor([1.0, 0.0, 1.0])
        loss = bernoulli_nll(probs, obs)
        assert loss.item() > 2.0

    def test_gradient_flows(self):
        probs = torch.tensor([0.5, 0.5], requires_grad=True)
        obs = torch.tensor([1.0, 0.0])
        loss = bernoulli_nll(probs, obs)
        loss.backward()
        assert probs.grad is not None

    def test_output_scalar(self):
        probs = torch.tensor([0.5, 0.3, 0.8])
        obs = torch.tensor([1.0, 0.0, 1.0])
        loss = bernoulli_nll(probs, obs)
        assert loss.ndim == 0


class TestBetaNLL:
    def test_good_vs_bad_prediction(self):
        """Good predictions should yield lower loss than bad ones."""
        mu = torch.tensor([0.8, 0.2, 0.5])
        obs = torch.tensor([0.8, 0.2, 0.5])
        good_loss = beta_nll(mu, obs, phi=10.0).item()

        bad_mu = torch.tensor([0.2, 0.8, 0.1])
        bad_loss = beta_nll(bad_mu, obs, phi=10.0).item()
        assert good_loss < bad_loss

    def test_higher_phi_penalizes_more(self):
        """Higher phi should penalize deviations from mu more."""
        mu = torch.tensor([0.7])
        obs = torch.tensor([0.5])
        loss_low_phi = beta_nll(mu, obs, phi=2.0).item()
        loss_high_phi = beta_nll(mu, obs, phi=50.0).item()
        assert loss_high_phi > loss_low_phi

    def test_gradient_flows(self):
        mu = torch.tensor([0.5, 0.5], requires_grad=True)
        obs = torch.tensor([0.7, 0.3])
        loss = beta_nll(mu, obs, phi=10.0)
        loss.backward()
        assert mu.grad is not None

    def test_output_scalar(self):
        mu = torch.tensor([0.5, 0.3, 0.8])
        obs = torch.tensor([0.6, 0.2, 0.9])
        loss = beta_nll(mu, obs, phi=10.0)
        assert loss.ndim == 0

    def test_symmetric_at_half(self):
        """Beta NLL should be symmetric around 0.5 when mu=0.5."""
        mu = torch.tensor([0.5])
        obs_low = torch.tensor([0.3])
        obs_high = torch.tensor([0.7])
        loss_low = beta_nll(mu, obs_low, phi=10.0).item()
        loss_high = beta_nll(mu, obs_high, phi=10.0).item()
        assert abs(loss_low - loss_high) < 1e-5
