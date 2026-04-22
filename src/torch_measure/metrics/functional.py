# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Functional API for common measurement metrics.

Provides a compute_all() function that returns multiple metrics at once,
similar to the pattern in predictive-eval/utils/metrics.py compute_metrics.
"""

from __future__ import annotations

import torch

from torch_measure.metrics.calibration import brier_score, expected_calibration_error
from torch_measure.metrics.reliability import infit_statistics


def compute_all(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute a standard suite of measurement metrics.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted probabilities (n_subjects, n_items).
    observed : torch.Tensor
        Observed binary responses (n_subjects, n_items).
    mask : torch.Tensor | None
        Boolean mask of entries to evaluate.

    Returns
    -------
    dict
        Dictionary with metric names and values:
        - 'auc': Area under ROC curve
        - 'log_likelihood': Mean log-likelihood
        - 'brier': Brier score
        - 'ece': Expected Calibration Error
        - 'mae': Mean absolute error of per-subject means
        - 'mean_infit': Mean infit statistic across items
    """
    if mask is None:
        mask = ~torch.isnan(observed)

    p = predicted[mask].flatten()
    y = observed[mask].flatten().float()

    metrics = {}

    # AUC
    try:
        from torchmetrics import AUROC

        auroc = AUROC(task="binary")
        metrics["auc"] = auroc(p, y.long()).item()
    except ImportError:
        # Fallback: simple AUC approximation
        metrics["auc"] = _simple_auc(p, y)

    # Log-likelihood
    p_clamped = p.clamp(1e-7, 1 - 1e-7)
    ll = (y * torch.log(p_clamped) + (1 - y) * torch.log(1 - p_clamped)).mean()
    metrics["log_likelihood"] = ll.item()

    # Brier score
    metrics["brier"] = brier_score(predicted, observed, mask)

    # ECE
    metrics["ece"] = expected_calibration_error(predicted, observed, mask)

    # MAE of per-subject means
    pred_means = (predicted * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
    obs_means = (observed * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
    metrics["mae"] = (pred_means - obs_means).abs().mean().item()

    # Mean infit
    infit = infit_statistics(predicted, observed, mask)
    metrics["mean_infit"] = infit.mean().item()

    return metrics


def _simple_auc(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    """Simple AUC computation without torchmetrics."""
    pos = predicted[observed == 1]
    neg = predicted[observed == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney statistic
    n_correct = sum((p > n).float().item() for p in pos for n in neg[:100])
    n_total = len(pos) * min(len(neg), 100)
    return n_correct / max(n_total, 1)
