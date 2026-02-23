# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Shared test fixtures for torch_measure tests."""

import pytest
import torch


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def small_response_matrix(seed):
    """A small synthetic binary response matrix (20 subjects x 30 items)."""
    n_subjects, n_items = 20, 30
    # Generate from a simple IRT model
    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)
    logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
    probs = torch.sigmoid(logit)
    responses = torch.bernoulli(probs)
    return responses


@pytest.fixture
def medium_response_matrix(seed):
    """A medium synthetic response matrix (50 subjects x 100 items)."""
    n_subjects, n_items = 50, 100
    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)
    logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
    probs = torch.sigmoid(logit)
    responses = torch.bernoulli(probs)
    return responses


@pytest.fixture
def response_matrix_with_nans(small_response_matrix):
    """Response matrix with 10% missing data."""
    data = small_response_matrix.clone()
    mask = torch.rand_like(data) < 0.1
    data[mask] = float("nan")
    return data


@pytest.fixture
def small_beta_response_matrix(seed):
    """A small synthetic continuous response matrix (20 subjects x 30 items)."""
    n_subjects, n_items = 20, 30
    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)
    logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
    mu = torch.sigmoid(logit)
    phi = 10.0
    alpha = mu * phi
    beta_param = (1.0 - mu) * phi
    dist = torch.distributions.Beta(alpha, beta_param)
    responses = dist.sample()
    return responses.clamp(1e-6, 1 - 1e-6)


@pytest.fixture
def medium_beta_response_matrix(seed):
    """A medium synthetic continuous response matrix (50 subjects x 100 items)."""
    n_subjects, n_items = 50, 100
    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)
    logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
    mu = torch.sigmoid(logit)
    phi = 10.0
    alpha = mu * phi
    beta_param = (1.0 - mu) * phi
    dist = torch.distributions.Beta(alpha, beta_param)
    responses = dist.sample()
    return responses.clamp(1e-6, 1 - 1e-6)


@pytest.fixture
def known_irt_params(seed):
    """Known IRT parameters for validation."""
    n_subjects, n_items = 30, 50
    ability = torch.linspace(-2, 2, n_subjects)
    difficulty = torch.linspace(-2, 2, n_items)
    discrimination = torch.ones(n_items) * 1.5
    guessing = torch.ones(n_items) * 0.2
    return {
        "ability": ability,
        "difficulty": difficulty,
        "discrimination": discrimination,
        "guessing": guessing,
        "n_subjects": n_subjects,
        "n_items": n_items,
    }
