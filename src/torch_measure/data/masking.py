# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Masking strategies for train/test splitting response matrices.

Consolidated from factor-model/calibration/util.py and predictive-eval/utils/util.py.
"""

from __future__ import annotations

import torch


def random_mask(observed: torch.Tensor, train_frac: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly split observed entries into train/test masks.

    Parameters
    ----------
    observed : torch.Tensor
        Boolean mask of observed entries (n_subjects x n_items).
    train_frac : float
        Fraction of observed entries to assign to training.

    Returns
    -------
    train_mask, test_mask : tuple[torch.Tensor, torch.Tensor]
        Boolean masks for training and testing.
    """
    coin = torch.rand_like(observed.float())
    train_mask = observed & (coin < train_frac)
    test_mask = observed & ~train_mask
    return train_mask, test_mask


def l_mask(observed: torch.Tensor, row_frac: float = 0.8, col_frac: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """L-shaped masking: fully observe a subset of rows AND columns for training.

    The test set consists of the intersection of held-out rows and held-out columns.
    This tests transductive generalization (new subjects on new items).

    Parameters
    ----------
    observed : torch.Tensor
        Boolean mask of observed entries.
    row_frac : float
        Fraction of rows to fully observe in training.
    col_frac : float
        Fraction of columns to fully observe in training.

    Returns
    -------
    train_mask, test_mask : tuple[torch.Tensor, torch.Tensor]
    """
    n_rows, n_cols = observed.shape
    n_train_rows = int(n_rows * row_frac)
    n_train_cols = int(n_cols * col_frac)

    row_perm = torch.randperm(n_rows)
    col_perm = torch.randperm(n_cols)

    train_rows = torch.zeros(n_rows, dtype=torch.bool)
    train_rows[row_perm[:n_train_rows]] = True

    train_cols = torch.zeros(n_cols, dtype=torch.bool)
    train_cols[col_perm[:n_train_cols]] = True

    # L-shape: train on (train_rows x all_cols) union (all_rows x train_cols)
    train_mask = observed & (train_rows[:, None] | train_cols[None, :])
    test_mask = observed & ~train_mask
    return train_mask, test_mask


def row_mask(
    observed: torch.Tensor,
    train_frac: float = 0.8,
    exposure_rate: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-based masking: fully observe some rows, partially observe the rest.

    Parameters
    ----------
    observed : torch.Tensor
        Boolean mask of observed entries.
    train_frac : float
        Fraction of rows to fully observe.
    exposure_rate : float
        Fraction of entries to observe in held-out rows.

    Returns
    -------
    train_mask, test_mask : tuple[torch.Tensor, torch.Tensor]
    """
    n_rows = observed.shape[0]
    n_train = int(n_rows * train_frac)
    perm = torch.randperm(n_rows)

    train_mask = observed.clone()
    # For held-out rows, only expose a fraction
    for i in perm[n_train:]:
        row_observed = observed[i]
        coin = torch.rand_like(row_observed.float())
        train_mask[i] = row_observed & (coin < exposure_rate)

    test_mask = observed & ~train_mask
    return train_mask, test_mask


def col_mask(
    observed: torch.Tensor,
    train_frac: float = 0.8,
    exposure_rate: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-based masking: fully observe some columns, partially observe the rest.

    Parameters
    ----------
    observed : torch.Tensor
        Boolean mask of observed entries.
    train_frac : float
        Fraction of columns to fully observe.
    exposure_rate : float
        Fraction of entries to observe in held-out columns.

    Returns
    -------
    train_mask, test_mask : tuple[torch.Tensor, torch.Tensor]
    """
    # Transpose, apply row_mask, transpose back
    train_t, test_t = row_mask(observed.T, train_frac, exposure_rate)
    return train_t.T, test_t.T


def model_mask(
    observed: torch.Tensor,
    train_frac: float = 0.8,
    exposure_rate: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model-based masking (alias for row_mask).

    Fully observe train_frac of models, partially observe the rest.
    """
    return row_mask(observed, train_frac, exposure_rate)


def item_mask(
    observed: torch.Tensor,
    train_frac: float = 0.8,
    exposure_rate: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Item-based masking (alias for col_mask).

    Fully observe train_frac of items, partially observe the rest.
    """
    return col_mask(observed, train_frac, exposure_rate)
