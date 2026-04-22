# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Shared test fixtures for torch_measure tests."""

import pytest
import torch

from torch_measure.data.pairwise import PairwiseComparisons


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


@pytest.fixture
def small_testlet_response_matrix(seed):
    """Synthetic binary response matrix with testlet structure (20 subjects x 30 items, 6 testlets)."""
    n_subjects, n_items, n_testlets = 20, 30, 6
    items_per_testlet = n_items // n_testlets  # 5

    testlet_map = torch.repeat_interleave(torch.arange(n_testlets), items_per_testlet)

    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)

    testlet_scales = torch.tensor([0.3, 0.5, 0.1, 0.4, 0.2, 0.6])
    testlet_effect = torch.randn(n_subjects, n_testlets) * testlet_scales.unsqueeze(0)

    logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)
    logit = logit + testlet_effect[:, testlet_map]
    probs = torch.sigmoid(logit)
    responses = torch.bernoulli(probs)
    return responses, testlet_map


@pytest.fixture
def small_pairwise_comparisons(seed):
    """Synthetic pairwise comparisons (10 subjects, ~450 comparisons).

    Generated from known Bradley-Terry abilities so that fitted models
    should recover the ground-truth ranking.
    """
    n_subjects = 10
    ability = torch.linspace(-2, 2, n_subjects)

    # Generate all ordered pairs, sample each once
    subject_a_list, subject_b_list, outcome_list = [], [], []
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            prob = torch.sigmoid(ability[i] - ability[j])
            outcome = torch.bernoulli(prob)
            subject_a_list.append(i)
            subject_b_list.append(j)
            outcome_list.append(outcome.item())

    subject_ids = [f"model_{i}" for i in range(n_subjects)]
    return PairwiseComparisons(
        subject_a=torch.tensor(subject_a_list, dtype=torch.long),
        subject_b=torch.tensor(subject_b_list, dtype=torch.long),
        outcome=torch.tensor(outcome_list, dtype=torch.float32),
        subject_ids=subject_ids,
    )
