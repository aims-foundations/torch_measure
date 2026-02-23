# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Loss functions for IRT model fitting."""

from __future__ import annotations

import torch
from torch.distributions import Bernoulli


def bernoulli_nll(predicted_probs: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Bernoulli negative log-likelihood for binary responses.

    Parameters
    ----------
    predicted_probs : torch.Tensor
        Model-predicted probabilities, should be in (0, 1).
    observed : torch.Tensor
        Observed binary responses (0 or 1).

    Returns
    -------
    torch.Tensor
        Scalar mean NLL.
    """
    return -Bernoulli(probs=predicted_probs).log_prob(observed).mean()


def beta_nll(predicted_probs: torch.Tensor, observed: torch.Tensor, phi: float = 10.0) -> torch.Tensor:
    """Beta negative log-likelihood for continuous (0, 1) responses.

    Uses a Beta distribution parameterized by mean ``mu`` and precision
    ``phi``: ``a = mu * phi``, ``b = (1 - mu) * phi``.

    Parameters
    ----------
    predicted_probs : torch.Tensor
        Model-predicted mean mu, should be in (0, 1).
    observed : torch.Tensor
        Observed response probabilities in (0, 1).
    phi : float
        Precision parameter of the Beta distribution. Higher values
        concentrate the distribution more tightly around mu.

    Returns
    -------
    torch.Tensor
        Scalar mean NLL.

    References
    ----------
    .. [1] Item Response Scaling Laws (ICML 2026).
    """
    a = predicted_probs * phi
    b = (1.0 - predicted_probs) * phi
    nll = -(
        (a - 1) * torch.log(observed)
        + (b - 1) * torch.log1p(-observed)
        - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    )
    return nll.mean()
