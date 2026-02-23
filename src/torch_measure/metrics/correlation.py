# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Correlation metrics for binary response data.

Consolidated from factor-model/calibration/util.py tetrachoric_matrix_torch.
"""

from __future__ import annotations

import torch


def tetrachoric_correlation(data: torch.Tensor, min_pairs: int = 5) -> torch.Tensor:
    """Compute the tetrachoric correlation matrix for binary data.

    Uses the cosine-pi approximation:
        r = cos(pi / (1 + sqrt(AD / BC)))

    where A, B, C, D are the counts in the 2x2 contingency table for each
    pair of items.

    Parameters
    ----------
    data : torch.Tensor
        Binary response matrix (n_subjects, n_items) with values 0, 1, or NaN.
    min_pairs : int
        Minimum number of valid pairs required. Pairs with fewer observations
        get correlation 0.

    Returns
    -------
    torch.Tensor
        Tetrachoric correlation matrix of shape (n_items, n_items).
    """
    valid = ~torch.isnan(data)
    data_clean = data.clone()
    data_clean[~valid] = 0.0

    # Compute 2x2 contingency tables for all pairs
    # A = both 1, D = both 0, B = (i=1, j=0), C = (i=0, j=1)
    both_valid = valid.float().T @ valid.float()  # (M, M)
    a = data_clean.T @ data_clean  # both correct
    d = (1 - data_clean).T @ (1 - data_clean) * (valid.float().T @ valid.float()).clamp(min=1).reciprocal()
    d = d * both_valid  # both incorrect

    # Recompute properly
    x = data_clean * valid.float()
    x_not = (1 - data_clean) * valid.float()

    a = x.T @ x  # (1,1) cell
    d = x_not.T @ x_not  # (0,0) cell
    b = x.T @ x_not  # (1,0) cell
    c = x_not.T @ x  # (0,1) cell

    # Tetrachoric approximation: r = cos(pi / (1 + sqrt(AD/BC)))
    ad = a * d
    bc = b * c
    bc = bc.clamp(min=1e-10)  # avoid division by zero
    ratio = (ad / bc).clamp(min=0)
    sqrt_ratio = torch.sqrt(ratio)

    r = torch.cos(torch.pi / (1 + sqrt_ratio))

    # Handle edge cases
    r[both_valid < min_pairs] = 0.0
    r.fill_diagonal_(1.0)

    # Symmetrize
    r = (r + r.T) / 2

    return r


def point_biserial_correlation(
    continuous: torch.Tensor,
    binary: torch.Tensor,
) -> torch.Tensor:
    """Compute point-biserial correlation between continuous and binary variables.

    Parameters
    ----------
    continuous : torch.Tensor
        Continuous variable (e.g., total score) of shape (N,).
    binary : torch.Tensor
        Binary variable (e.g., item response) of shape (N,) or (N, M).

    Returns
    -------
    torch.Tensor
        Correlation(s). Scalar if binary is 1D, shape (M,) if 2D.
    """
    if binary.ndim == 1:
        binary = binary.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    mask = ~torch.isnan(binary) & ~torch.isnan(continuous.unsqueeze(1))
    results = []

    for j in range(binary.shape[1]):
        m = mask[:, j]
        x = continuous[m]
        y = binary[m, j]

        group1 = x[y == 1]
        group0 = x[y == 0]

        if len(group1) < 2 or len(group0) < 2:
            results.append(torch.tensor(0.0))
            continue

        m1, m0 = group1.mean(), group0.mean()
        n1, n0 = len(group1), len(group0)
        n = n1 + n0
        s = x.std()

        if s < 1e-10:
            results.append(torch.tensor(0.0))
            continue

        rpb = (m1 - m0) / s * torch.sqrt(torch.tensor(n1 * n0 / n**2, dtype=torch.float32))
        results.append(rpb)

    result = torch.stack(results)
    return result.squeeze() if squeeze else result
