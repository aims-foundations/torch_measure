# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Fisher-information-based standard errors for IRT parameters.

Provides frequentist uncertainty estimates by inverting the observed
Fisher information.
"""

from __future__ import annotations

import torch

from torch_measure.cat.fisher import fisher_information


def ability_standard_errors(
    ability: torch.Tensor,
    difficulty: torch.Tensor,
    discrimination: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute standard errors for ability estimates.

    SE(theta_i) = 1 / sqrt(sum_j I_j(theta_i)), where I_j is the Fisher
    information of item j evaluated at theta_i.

    Parameters
    ----------
    ability : torch.Tensor
        Subject ability values, shape (N,).
    difficulty : torch.Tensor
        Item difficulty values, shape (M,).
    discrimination : torch.Tensor | None
        Item discrimination values, shape (M,). Defaults to 1 (Rasch).

    Returns
    -------
    torch.Tensor
        Standard errors per subject, shape (N,).
    """
    info = fisher_information(ability, difficulty, discrimination)  # (N, M)
    total_info = info.sum(dim=1)  # (N,)
    return 1.0 / torch.sqrt(total_info)


def difficulty_standard_errors(
    ability: torch.Tensor,
    difficulty: torch.Tensor,
    response_matrix: torch.Tensor,
    discrimination: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute standard errors for difficulty estimates.

    SE(b_j) = 1 / sqrt(sum_i I_j(theta_i)) over observed subjects for each
    item, where I_j(theta_i) = a_j^2 * P_ij * Q_ij.

    Parameters
    ----------
    ability : torch.Tensor
        Subject ability values, shape (N,).
    difficulty : torch.Tensor
        Item difficulty values, shape (M,).
    response_matrix : torch.Tensor
        Response matrix, shape (N, M). Used only for determining observed entries.
    discrimination : torch.Tensor | None
        Item discrimination values, shape (M,). Defaults to 1 (Rasch).
    mask : torch.Tensor | None
        Boolean mask of observed entries, shape (N, M). If None, all
        non-NaN entries are treated as observed.

    Returns
    -------
    torch.Tensor
        Standard errors per item, shape (M,).
    """
    if mask is None:
        mask = ~torch.isnan(response_matrix)

    info = fisher_information(ability, difficulty, discrimination)  # (N, M)
    masked_info = info * mask.float()  # zero out unobserved
    total_info = masked_info.sum(dim=0)  # (M,)
    return 1.0 / torch.sqrt(total_info)


def discrimination_standard_errors(
    ability: torch.Tensor,
    difficulty: torch.Tensor,
    discrimination: torch.Tensor,
    response_matrix: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute standard errors for discrimination estimates.

    I(a_j) = sum_i (theta_i - b_j)^2 * P_ij * Q_ij over observed subjects.
    SE(a_j) = 1 / sqrt(I(a_j)).

    Parameters
    ----------
    ability : torch.Tensor
        Subject ability values, shape (N,).
    difficulty : torch.Tensor
        Item difficulty values, shape (M,).
    discrimination : torch.Tensor
        Item discrimination values, shape (M,).
    response_matrix : torch.Tensor
        Response matrix, shape (N, M). Used only for determining observed entries.
    mask : torch.Tensor | None
        Boolean mask of observed entries, shape (N, M). If None, all
        non-NaN entries are treated as observed.

    Returns
    -------
    torch.Tensor
        Standard errors per item, shape (M,).
    """
    if mask is None:
        mask = ~torch.isnan(response_matrix)

    logit = discrimination.unsqueeze(0) * (ability.unsqueeze(1) - difficulty.unsqueeze(0))
    p = torch.sigmoid(logit)
    q = 1.0 - p
    diff_sq = (ability.unsqueeze(1) - difficulty.unsqueeze(0)) ** 2  # (N, M)

    info_per_obs = diff_sq * p * q  # (N, M)
    masked_info = info_per_obs * mask.float()
    total_info = masked_info.sum(dim=0)  # (M,)
    return 1.0 / torch.sqrt(total_info)
