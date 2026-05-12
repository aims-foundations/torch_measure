# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Reliability and fit statistics for measurement models.

Consolidated from predictive-eval/utils/metrics.py and factor-model/calibration/metrics.py.
"""

from __future__ import annotations

import torch


def infit_statistics(predicted: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute Rasch infit (information-weighted) mean square statistics per item.

    Infit is sensitive to unexpected responses near item difficulty.
    Values near 1.0 indicate good fit. Values > 1.3 indicate underfit (noise),
    values < 0.7 indicate overfit (Guttman pattern).

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted probabilities (n_subjects, n_items).
    observed : torch.Tensor
        Observed binary responses (n_subjects, n_items).
    mask : torch.Tensor | None
        Boolean mask of entries to include.

    Returns
    -------
    torch.Tensor
        Infit statistics per item, shape (n_items,).
    """
    if mask is None:
        mask = ~torch.isnan(observed)

    p = predicted.clamp(1e-7, 1 - 1e-7)
    variance = p * (1 - p)  # Bernoulli variance
    residual_sq = (observed - p) ** 2

    # Weighted mean square: sum(residual^2) / sum(variance) per item
    numerator = (residual_sq * mask.float()).sum(dim=0)
    denominator = (variance * mask.float()).sum(dim=0).clamp(min=1e-10)

    return numerator / denominator


def outfit_statistics(
    predicted: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute Rasch outfit (unweighted) mean square statistics per item.

    Outfit is sensitive to unexpected responses far from item difficulty.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted probabilities (n_subjects, n_items).
    observed : torch.Tensor
        Observed binary responses (n_subjects, n_items).
    mask : torch.Tensor | None
        Boolean mask of entries to include.

    Returns
    -------
    torch.Tensor
        Outfit statistics per item, shape (n_items,).
    """
    if mask is None:
        mask = ~torch.isnan(observed)

    p = predicted.clamp(1e-7, 1 - 1e-7)
    variance = p * (1 - p)
    standardized_residual_sq = ((observed - p) ** 2) / variance

    # Simple mean of standardized residuals per item
    numerator = (standardized_residual_sq * mask.float()).sum(dim=0)
    count = mask.float().sum(dim=0).clamp(min=1)

    return numerator / count


def item_total_correlation(data: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute corrected item-total correlation for each item.

    For each item, computes the Pearson correlation between the item
    responses and the total score excluding that item.

    Parameters
    ----------
    data : torch.Tensor
        Binary response matrix (n_subjects, n_items).
    mask : torch.Tensor | None
        Boolean mask.

    Returns
    -------
    torch.Tensor
        Item-total correlations, shape (n_items,).
    """
    if mask is None:
        mask = ~torch.isnan(data)

    data_clean = data.clone()
    data_clean[~mask] = 0.0

    total = (data_clean * mask.float()).sum(dim=1)  # (N,)
    n_items = data.shape[1]
    correlations = []

    for j in range(n_items):
        item_mask = mask[:, j]
        item_vals = data[item_mask, j]
        corrected_total = total[item_mask] - item_vals  # exclude item j

        if len(item_vals) < 3 or item_vals.std() < 1e-10 or corrected_total.std() < 1e-10:
            correlations.append(torch.tensor(0.0))
            continue

        # Pearson correlation
        x = item_vals - item_vals.mean()
        y = corrected_total - corrected_total.mean()
        r = (x * y).sum() / (x.norm() * y.norm() + 1e-10)
        correlations.append(r)

    return torch.stack(correlations)


def cronbach_alpha(data: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Compute Cronbach's alpha reliability coefficient.

    Parameters
    ----------
    data : torch.Tensor
        Response matrix (n_subjects, n_items).
    mask : torch.Tensor | None
        Boolean mask.

    Returns
    -------
    float
        Cronbach's alpha.
    """
    if mask is None:
        mask = ~torch.isnan(data)

    data_clean = data.clone()
    data_clean[~mask] = 0.0

    k = data.shape[1]
    item_vars = []
    for j in range(k):
        m = mask[:, j]
        vals = data[m, j]
        if len(vals) > 1:
            item_vars.append(vals.var().item())
        else:
            item_vars.append(0.0)

    sum_item_var = sum(item_vars)
    total = (data_clean * mask.float()).sum(dim=1)

    # Only use subjects with all items observed for total variance
    all_observed = mask.all(dim=1)
    total_var = total.var().item() if all_observed.sum() < 3 else total[all_observed].var().item()

    if total_var < 1e-10:
        return 0.0

    return (k / (k - 1)) * (1 - sum_item_var / total_var)
