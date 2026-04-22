# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Fisher information for adaptive item selection.

Consolidated from irsl/utils.py compute_fisher_info_2pl.
"""

from __future__ import annotations

import torch


def fisher_information(
    ability: torch.Tensor,
    difficulty: torch.Tensor,
    discrimination: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Fisher information for each item at given ability levels.

    For the 2PL model: I(theta) = a^2 * P(theta) * (1 - P(theta))
    For the Rasch model (a=1): I(theta) = P(theta) * (1 - P(theta))

    Higher information means the item is more useful for estimating
    ability at that level.

    Parameters
    ----------
    ability : torch.Tensor
        Subject ability values, shape (N,) or scalar.
    difficulty : torch.Tensor
        Item difficulty values, shape (M,).
    discrimination : torch.Tensor | None
        Item discrimination values, shape (M,). Defaults to 1 (Rasch).

    Returns
    -------
    torch.Tensor
        Fisher information matrix, shape (N, M) or (M,) if ability is scalar.
    """
    if ability.ndim == 0:
        ability = ability.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # P(correct) = sigmoid(a * (theta - b))
    logit = ability.unsqueeze(1) - difficulty.unsqueeze(0)

    if discrimination is not None:
        logit = discrimination.unsqueeze(0) * logit

    p = torch.sigmoid(logit)
    q = 1 - p

    info = p * q

    if discrimination is not None:
        info = discrimination.unsqueeze(0) ** 2 * info

    return info.squeeze(0) if squeeze else info
