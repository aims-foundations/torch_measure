# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Mokken scalability analysis.

Consolidated from fantastic-bugs/src/metrics.py.
"""

from __future__ import annotations

import torch


def mokken_scalability(data: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    """Compute Mokken scalability coefficients.

    Mokken scaling is a non-parametric IRT approach that tests whether
    items form a unidimensional scale. The H coefficient measures how
    well item pairs conform to the Guttman pattern.

    H >= 0.5: strong scale
    0.4 <= H < 0.5: medium scale
    0.3 <= H < 0.4: weak scale
    H < 0.3: not a scale

    Parameters
    ----------
    data : torch.Tensor
        Binary response matrix (n_subjects, n_items).
    mask : torch.Tensor | None
        Boolean mask.

    Returns
    -------
    dict
        Dictionary with:
        - 'H': Overall scalability coefficient
        - 'H_items': Per-item scalability coefficients, shape (n_items,)
        - 'H_pairs': Pairwise scalability matrix, shape (n_items, n_items)
    """
    if mask is None:
        mask = ~torch.isnan(data)

    data_clean = data.clone()
    data_clean[~mask] = 0.0
    n_items = data.shape[1]

    # Item means (facility/easiness)
    item_sums = (data_clean * mask.float()).sum(dim=0)
    item_counts = mask.float().sum(dim=0).clamp(min=1)
    item_means = item_sums / item_counts

    # Pairwise H coefficients
    h_pairs = torch.zeros(n_items, n_items)
    total_obs_error = 0.0
    total_exp_error = 0.0

    for i in range(n_items):
        for j in range(i + 1, n_items):
            # Ensure p_i >= p_j (order by difficulty)
            pi, pj = item_means[i], item_means[j]
            if pi < pj:
                pi, pj = pj, pi
                col_easy, col_hard = j, i
            else:
                col_easy, col_hard = i, j

            # Both observed
            pair_mask = mask[:, i] & mask[:, j]
            if pair_mask.sum() < 5:
                continue

            # Observed Guttman errors: easy=0, hard=1
            obs_errors = ((data_clean[pair_mask, col_easy] == 0) & (data_clean[pair_mask, col_hard] == 1)).float().sum()

            # Expected errors under independence
            n_pair = pair_mask.sum().float()
            exp_errors = n_pair * (1 - pi) * pj

            if exp_errors > 0:
                h_ij = 1 - obs_errors / exp_errors
                h_pairs[i, j] = h_ij
                h_pairs[j, i] = h_ij
                total_obs_error += obs_errors.item()
                total_exp_error += exp_errors.item()

    # Overall H
    h_overall = 1 - total_obs_error / max(total_exp_error, 1e-10)

    # Per-item H
    h_items = torch.zeros(n_items)
    for i in range(n_items):
        item_obs = 0.0
        item_exp = 0.0
        for j in range(n_items):
            if i == j:
                continue
            pi, pj = item_means[i], item_means[j]
            if pi < pj:
                pi, pj = pj, pi
                col_easy, col_hard = j, i
            else:
                col_easy, col_hard = i, j

            pair_mask = mask[:, i] & mask[:, j]
            if pair_mask.sum() < 5:
                continue

            obs_e = ((data_clean[pair_mask, col_easy] == 0) & (data_clean[pair_mask, col_hard] == 1)).float().sum()
            exp_e = pair_mask.sum().float() * (1 - pi) * pj

            item_obs += obs_e.item()
            item_exp += exp_e.item()

        h_items[i] = 1 - item_obs / max(item_exp, 1e-10)

    return {
        "H": h_overall,
        "H_items": h_items,
        "H_pairs": h_pairs,
    }
