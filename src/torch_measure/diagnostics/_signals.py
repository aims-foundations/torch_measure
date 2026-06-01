# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Per-item signal functions used by the diagnostics ensemble.

The three signals are re-exported from :mod:`torch_measure.metrics` so
diagnostics callers have a single import surface. Under a Rasch model these
quantities are theoretically non-negative (Corollaries 1 and 2 in
arXiv:2511.16842), so negative values are evidence of item misfit.
"""

from __future__ import annotations

import torch

from torch_measure.metrics.correlation import tetrachoric_correlation
from torch_measure.metrics.reliability import item_total_correlation
from torch_measure.metrics.scalability import mokken_scalability

__all__ = [
    "average_tetrachoric_correlation",
    "item_scalability",
    "item_total_correlation",
    "tetrachoric_correlation",
]


def average_tetrachoric_correlation(
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    min_pairs: int = 5,
) -> torch.Tensor:
    """Average pairwise tetrachoric correlation between item j and all others.

    Wraps :func:`torch_measure.metrics.correlation.tetrachoric_correlation`
    and averages each row over its off-diagonal entries. Under a Rasch model
    all pairwise tetrachoric correlations are non-negative, so a negative
    average flags item j as inconsistent with the assumed unidimensional
    structure.

    Parameters
    ----------
    data : torch.Tensor
        Binary response matrix of shape ``(n_subjects, n_items)``. NaN
        entries are treated as missing.
    mask : torch.Tensor | None
        Unused, kept for signature parity with the other signal helpers.
    min_pairs : int
        Forwarded to :func:`tetrachoric_correlation`. Pairs with fewer
        observations contribute 0.

    Returns
    -------
    torch.Tensor
        Average off-diagonal tetrachoric correlation per item, shape
        ``(n_items,)``.
    """
    del mask
    corr = tetrachoric_correlation(data, min_pairs=min_pairs)
    n_items = corr.shape[0]
    if n_items < 2:
        return torch.zeros(n_items, dtype=corr.dtype, device=corr.device)
    row_sum = corr.sum(dim=1) - corr.diagonal()
    return row_sum / (n_items - 1)


def item_scalability(
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-item Mokken / Loevinger scalability coefficient.

    Thin wrapper around
    :func:`torch_measure.metrics.scalability.mokken_scalability` that
    returns only the per-item ``H_items`` tensor. Under a Rasch model
    every entry is non-negative.

    Parameters
    ----------
    data : torch.Tensor
        Binary response matrix of shape ``(n_subjects, n_items)``.
    mask : torch.Tensor | None
        Optional boolean mask of valid observations.

    Returns
    -------
    torch.Tensor
        Per-item scalability coefficient of shape ``(n_items,)``.
    """
    return mokken_scalability(data, mask=mask)["H_items"]
