# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Transforms for response matrices."""

from __future__ import annotations

import torch


def binarize(data: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert continuous response matrix to binary.

    Parameters
    ----------
    data : torch.Tensor
        Response matrix with values in [0, 1] (may contain NaN).
    threshold : float
        Values >= threshold become 1, otherwise 0.

    Returns
    -------
    torch.Tensor
        Binary response matrix (NaN preserved).
    """
    result = (data >= threshold).float()
    result[torch.isnan(data)] = float("nan")
    return result


def normalize_rows(data: torch.Tensor) -> torch.Tensor:
    """Normalize each row to zero mean and unit variance (ignoring NaN).

    Parameters
    ----------
    data : torch.Tensor
        Response matrix (may contain NaN).

    Returns
    -------
    torch.Tensor
        Row-normalized matrix (NaN preserved).
    """
    mask = ~torch.isnan(data)
    result = data.clone()

    for i in range(data.shape[0]):
        row_vals = data[i, mask[i]]
        if len(row_vals) > 1:
            mean = row_vals.mean()
            std = row_vals.std()
            if std > 0:
                result[i, mask[i]] = (row_vals - mean) / std

    return result
