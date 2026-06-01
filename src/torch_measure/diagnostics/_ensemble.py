# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Ensemble layer for the diagnostics module.

Combines the three per-item signals (average tetrachoric correlation,
Mokken scalability, item-total correlation) via a Gaussian rank transform
into a single ranked DataFrame of suspect items. Implements the procedure
described in arXiv:2511.16842 ("Fantastic Bugs and Where to Find Them in
AI Benchmarks") with the same OR / AND / Majority / mean fusion rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats

from torch_measure.diagnostics._judge import ItemJudge
from torch_measure.diagnostics._signals import (
    average_tetrachoric_correlation,
    item_scalability,
    item_total_correlation,
)

__all__ = ["flag_items", "gaussian_rank"]

_SIGNAL_FUNCTIONS = {
    "tetrachoric": average_tetrachoric_correlation,
    "scalability": item_scalability,
    "item_total": item_total_correlation,
}

_SCORE_COLUMNS = {
    "tetrachoric": "tetrachoric_score",
    "scalability": "scalability_score",
    "item_total": "item_total_score",
}


def gaussian_rank(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Map values to standard-normal scores via percentile ranks.

    Computes ``A_i = Phi^{-1}((r_i - 0.5) / N)`` where ``r_i`` is the
    rank of ``values[i]`` (ties broken by average). NaN inputs map to 0
    (the standard-normal median) so they neither flag nor protect an
    item.

    Parameters
    ----------
    values : torch.Tensor | np.ndarray
        1-D array of per-item scores.

    Returns
    -------
    np.ndarray
        Gaussian-rank transformed values, same shape as input.
    """
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    arr = arr.astype(np.float64, copy=True)
    n = arr.shape[0]
    if n == 0:
        return arr

    finite = np.isfinite(arr)
    out = np.zeros(n, dtype=np.float64)
    if finite.sum() < 2:
        return out

    ranks_finite = stats.rankdata(arr[finite], method="average")
    n_finite = ranks_finite.shape[0]
    percentile = (ranks_finite - 0.5) / n_finite
    out[finite] = stats.norm.ppf(percentile)
    return out


def flag_items(
    response_matrix: torch.Tensor,
    signals: list[str] | None = None,
    ensemble_method: str = "mean",
    threshold_percentile: float = 0.5,
    item_names: list[str] | None = None,
    item_texts: list[str] | None = None,
    judge: ItemJudge | None = None,
) -> pd.DataFrame:
    """Flag potentially invalid benchmark items via measurement diagnostics.

    Computes the requested per-item signals, converts each to an anomaly
    score on the standard-normal scale, and combines them via
    ``ensemble_method``. Under a Rasch model each raw signal is
    theoretically non-negative, so the anomaly direction is the negative
    of the raw value: more negative raw signal becomes a larger anomaly.

    Parameters
    ----------
    response_matrix : torch.Tensor
        Binary response matrix of shape ``(n_subjects, n_items)``. NaN
        entries are treated as missing throughout.
    signals : list[str] | None
        Subset of ``{"tetrachoric", "scalability", "item_total"}``.
        Default uses all three.
    ensemble_method : str
        How to combine per-signal anomalies into a single decision.
        ``"mean"`` averages the Gaussian-rank scores and flags items whose
        mean exceeds ``Phi^{-1}(threshold_percentile)``. ``"or"``,
        ``"and"``, and ``"majority"`` threshold each signal at
        ``threshold_percentile`` and apply the corresponding vote rule.
    threshold_percentile : float
        Percentile rank cutoff in ``[0, 1]``. Defaults to ``0.5`` per the
        reference paper.
    item_names : list[str] | None
        Optional labels written into the ``item_name`` column.
    item_texts : list[str] | None
        Item text supplied to ``judge``. If ``judge`` is given but this is
        ``None``, ``item_names`` (or ``f"item_{idx}"``) is used instead.
    judge : ItemJudge | None
        Optional second-pass reviewer. Called once per flagged item; its
        return string is stored in ``judge_output``.

    Returns
    -------
    pd.DataFrame
        One row per item, sorted by ``ensemble_score`` descending (most
        anomalous first). Always includes ``item_idx``,
        ``ensemble_score``, ``flagged`` plus a ``<signal>_score`` column
        for each requested signal. Includes ``item_name`` if
        ``item_names`` was provided and ``judge_output`` if ``judge`` was
        provided.
    """
    if signals is None:
        signals = ["tetrachoric", "scalability", "item_total"]
    if not signals:
        raise ValueError("`signals` must contain at least one signal name")
    unknown = set(signals) - _SIGNAL_FUNCTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown signal(s): {sorted(unknown)}. Choose from {sorted(_SIGNAL_FUNCTIONS)}")
    if ensemble_method not in {"mean", "or", "and", "majority"}:
        raise ValueError(f"Unknown ensemble_method {ensemble_method!r}")
    if not 0.0 <= threshold_percentile <= 1.0:
        raise ValueError("`threshold_percentile` must be in [0, 1]")

    n_items = response_matrix.shape[1]
    if item_names is not None and len(item_names) != n_items:
        raise ValueError(f"item_names length {len(item_names)} does not match n_items={n_items}")
    if item_texts is not None and len(item_texts) != n_items:
        raise ValueError(f"item_texts length {len(item_texts)} does not match n_items={n_items}")

    raw_scores: dict[str, np.ndarray] = {}
    anomalies: dict[str, np.ndarray] = {}
    for name in signals:
        raw = _SIGNAL_FUNCTIONS[name](response_matrix)
        raw_np = raw.detach().cpu().numpy().astype(np.float64, copy=False)
        raw_scores[name] = raw_np
        anomalies[name] = gaussian_rank(-raw_np)

    anomaly_stack = np.stack([anomalies[s] for s in signals], axis=0)
    ensemble_score = anomaly_stack.mean(axis=0)

    cutoff = float(stats.norm.ppf(threshold_percentile)) if 0.0 < threshold_percentile < 1.0 else 0.0
    if ensemble_method == "mean":
        flagged = ensemble_score > cutoff
    else:
        votes = anomaly_stack > cutoff
        if ensemble_method == "or":
            flagged = votes.any(axis=0)
        elif ensemble_method == "and":
            flagged = votes.all(axis=0)
        else:
            flagged = votes.sum(axis=0) > (len(signals) / 2)

    columns: dict[str, np.ndarray] = {"item_idx": np.arange(n_items)}
    if item_names is not None:
        columns["item_name"] = np.asarray(item_names, dtype=object)
    for name in signals:
        columns[_SCORE_COLUMNS[name]] = raw_scores[name]
    columns["ensemble_score"] = ensemble_score
    columns["flagged"] = flagged

    df = pd.DataFrame(columns)

    if judge is not None:
        judge_output: list[str | None] = [None] * n_items
        for idx in np.where(flagged)[0]:
            if item_texts is not None:
                text = item_texts[idx]
            elif item_names is not None:
                text = item_names[idx]
            else:
                text = f"item_{idx}"
            judge_output[idx] = judge(text, int(idx), float(ensemble_score[idx]))
        df["judge_output"] = judge_output

    return df.sort_values("ensemble_score", ascending=False, kind="mergesort").reset_index(drop=True)
