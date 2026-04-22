# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Validity analysis metrics including Differential Item Functioning (DIF)."""

from __future__ import annotations

import torch


def differential_item_functioning(
    data: torch.Tensor,
    group: torch.Tensor,
    mask: torch.Tensor | None = None,
    method: str = "mh",
) -> dict:
    """Detect Differential Item Functioning (DIF).

    DIF occurs when subjects of equal ability from different groups have
    different probabilities of answering an item correctly.

    Parameters
    ----------
    data : torch.Tensor
        Binary response matrix (n_subjects, n_items).
    group : torch.Tensor
        Group membership for each subject (n_subjects,). Binary (0/1).
    mask : torch.Tensor | None
        Boolean mask.
    method : str
        DIF detection method. Currently supports "mh" (Mantel-Haenszel).

    Returns
    -------
    dict
        Dictionary with:
        - 'mh_statistic': Mantel-Haenszel chi-square per item, shape (n_items,)
        - 'effect_size': MH odds ratio (Delta-MH) per item, shape (n_items,)
        - 'flagged': Boolean mask of items flagged for DIF
    """
    if mask is None:
        mask = ~torch.isnan(data)

    data_clean = data.clone()
    data_clean[~mask] = 0.0

    n_items = data.shape[1]
    total_scores = (data_clean * mask.float()).sum(dim=1)

    # Bin by total score (quintiles)
    n_bins = 5
    percentiles = torch.quantile(total_scores, torch.linspace(0, 1, n_bins + 1))

    odds_ratios = torch.zeros(n_items)

    for j in range(n_items):
        alpha_sum = 0.0
        beta_sum = 0.0

        for k in range(n_bins):
            low = percentiles[k]
            high = percentiles[k + 1]
            if k < n_bins - 1:
                in_bin = (total_scores >= low) & (total_scores < high)
            else:
                in_bin = (total_scores >= low) & (total_scores <= high)

            in_bin = in_bin & mask[:, j]
            if in_bin.sum() < 2:
                continue

            g0 = in_bin & (group == 0)
            g1 = in_bin & (group == 1)

            n0 = g0.sum().float()
            n1 = g1.sum().float()
            if n0 == 0 or n1 == 0:
                continue

            # 2x2 table within this score stratum
            a = (data_clean[g1, j] == 1).sum().float()  # focal correct
            b = (data_clean[g1, j] == 0).sum().float()  # focal incorrect
            c = (data_clean[g0, j] == 1).sum().float()  # reference correct
            d = (data_clean[g0, j] == 0).sum().float()  # reference incorrect

            n_total = (a + b + c + d).clamp(min=1)
            alpha_sum += (a * d / n_total).item()
            beta_sum += (b * c / n_total).item()

        if beta_sum > 0:
            odds_ratios[j] = alpha_sum / beta_sum
        else:
            odds_ratios[j] = 1.0

    # Delta-MH effect size (ETS scale)
    delta_mh = -2.35 * torch.log(odds_ratios.clamp(min=1e-10))

    # Flag items with |Delta-MH| > 1.0 (moderate to large DIF)
    flagged = delta_mh.abs() > 1.0

    return {
        "effect_size": delta_mh,
        "odds_ratio": odds_ratios,
        "flagged": flagged,
    }
