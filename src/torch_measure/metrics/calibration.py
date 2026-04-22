# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Calibration metrics for predicted probabilities.

Consolidated from predictive-eval/utils/metrics.py compute_ece.
"""

from __future__ import annotations

import torch


def expected_calibration_error(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor | None = None,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    ECE = 0 means perfectly calibrated.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted probabilities.
    observed : torch.Tensor
        Observed binary outcomes.
    mask : torch.Tensor | None
        Boolean mask of entries to evaluate.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        ECE value in [0, 1].
    """
    if mask is None:
        mask = ~torch.isnan(observed)

    p = predicted[mask].flatten()
    y = observed[mask].flatten().float()

    if len(p) == 0:
        return 0.0

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (p >= bin_boundaries[i]) & (p < bin_boundaries[i + 1])
        if i == n_bins - 1:  # include right boundary for last bin
            in_bin = in_bin | (p == bin_boundaries[i + 1])

        n_in_bin = in_bin.sum().item()
        if n_in_bin == 0:
            continue

        avg_predicted = p[in_bin].mean().item()
        avg_observed = y[in_bin].mean().item()
        ece += abs(avg_predicted - avg_observed) * (n_in_bin / len(p))

    return ece


def brier_score(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Compute the Brier score (mean squared error of probabilities).

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted probabilities.
    observed : torch.Tensor
        Observed binary outcomes.
    mask : torch.Tensor | None
        Boolean mask.

    Returns
    -------
    float
        Brier score in [0, 1]. Lower is better.
    """
    if mask is None:
        mask = ~torch.isnan(observed)

    p = predicted[mask].flatten()
    y = observed[mask].flatten().float()

    if len(p) == 0:
        return 0.0

    return ((p - y) ** 2).mean().item()
