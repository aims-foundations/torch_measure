# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Generalizability-theory reliability for two-way crossed designs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    method: str = "moments",
) -> dict:
    """Decompose Var(response) into subject, item, subject x item, and residual facets.

    Henderson Method I (moments-based ANOVA estimator) on a person x item x
    replication crossed design. Negative variance estimates are clamped to 0.
    With one observation per cell, residual is unidentifiable.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses with columns ``subject_col``, ``item_col``,
        ``trial_col``, ``response_col``.
    subject_col, item_col, trial_col, response_col : str
        Column names; defaults match the measurement-db long-form schema.
    method : {"moments"}
        Only ``"moments"`` is implemented in v1.

    Returns
    -------
    dict
        Keys: ``subject``, ``item``, ``subject_item``, ``residual`` (variances,
        floats), ``n_subjects``, ``n_items`` (ints), ``n_reps_harmonic``
        (float; harmonic mean of cell counts), ``identifiable`` (dict[str,
        bool]), ``method`` (str).
    """
    import pandas as pd

    if method == "reml":
        raise NotImplementedError("method='reml' not implemented in v1.")
    if method != "moments":
        raise ValueError(f"Unknown method: {method!r}.")

    required = {subject_col, item_col, trial_col, response_col}
    missing = required - set(response_matrix.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    df = response_matrix[[subject_col, item_col, trial_col, response_col]].dropna(subset=[response_col])
    if not pd.api.types.is_numeric_dtype(df[response_col]):
        raise ValueError(f"{response_col!r} column must be numeric.")

    n_p = int(df[subject_col].nunique())
    n_i = int(df[item_col].nunique())
    if n_p < 2 or n_i < 2:
        raise ValueError(f"Need at least 2 subjects and 2 items; got n_subjects={n_p}, n_items={n_i}.")

    cell = df.groupby([subject_col, item_col])[response_col].agg(["mean", "count"]).reset_index()
    cell = cell.rename(columns={"mean": "_cell_mean", "count": "_cell_count"})

    if len(cell) < n_p * n_i:
        raise ValueError(
            f"Unbalanced design: {len(cell)}/{n_p * n_i} cells observed. "
            f"Every (subject, item) cell must have at least one observation."
        )

    counts = cell["_cell_count"].to_numpy(dtype=float)
    n_r = float(len(counts) / np.sum(1.0 / counts))  # harmonic mean of cell counts
    has_replications = bool(np.any(counts > 1))

    cell_table = cell.pivot(index=subject_col, columns=item_col, values="_cell_mean").to_numpy(dtype=float)
    grand_mean = cell_table.mean()
    subj_mean = cell_table.mean(axis=1)
    item_mean = cell_table.mean(axis=0)

    # ANOVA on cell means; multiply by n_r to lift to observation-level SS.
    ss_p = n_r * n_i * float(np.sum((subj_mean - grand_mean) ** 2))
    ss_i = n_r * n_p * float(np.sum((item_mean - grand_mean) ** 2))
    ss_pi = n_r * float(np.sum((cell_table - grand_mean) ** 2)) - ss_p - ss_i

    if has_replications:
        merged = df.merge(cell[[subject_col, item_col, "_cell_mean"]], on=[subject_col, item_col])
        ss_e = float(((merged[response_col] - merged["_cell_mean"]) ** 2).sum())
        df_e = int(np.sum(counts - 1))
        ms_e = ss_e / df_e if df_e > 0 else 0.0
    else:
        ms_e = 0.0

    ms_p = ss_p / (n_p - 1)
    ms_i = ss_i / (n_i - 1)
    ms_pi = ss_pi / ((n_p - 1) * (n_i - 1))

    sigma2_e = max(0.0, ms_e)
    sigma2_pi = max(0.0, (ms_pi - ms_e) / n_r)
    sigma2_i = max(0.0, (ms_i - ms_pi) / (n_p * n_r))
    sigma2_p = max(0.0, (ms_p - ms_pi) / (n_i * n_r))

    return {
        "subject": sigma2_p,
        "item": sigma2_i,
        "subject_item": sigma2_pi,
        "residual": sigma2_e,
        "n_subjects": n_p,
        "n_items": n_i,
        "n_reps_harmonic": n_r,
        "identifiable": {
            "subject": True,
            "item": True,
            "subject_item": True,
            "residual": has_replications,
        },
        "method": "moments",
    }


def g_coefficient(
    variance_components: dict,
    n_items: int,
    n_reps: int = 1,
    type: str = "absolute",
) -> float:
    """Brennan (2001) G-coefficient under a person x item x replication design.

    Relative G uses ranking-only error (subject x item + residual); absolute G
    (Phi) also includes the item main effect.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`, or any dict with keys
        ``subject``, ``item``, ``subject_item``, ``residual``.
    n_items : int
        Number of items in the projected design (>= 1).
    n_reps : int
        Replications per cell in the projected design (>= 1).
    type : {"relative", "absolute"}
        Which G-coefficient to compute.

    Returns
    -------
    float
        G-coefficient in [0, 1]. 0.0 if the denominator is numerically zero.
    """
    required = {"subject", "item", "subject_item", "residual"}
    missing = required - set(variance_components)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}.")
    if type not in {"relative", "absolute"}:
        raise ValueError(f"type must be 'relative' or 'absolute'; got {type!r}.")
    if n_items < 1 or n_reps < 1:
        raise ValueError(f"n_items and n_reps must be >= 1; got n_items={n_items}, n_reps={n_reps}.")

    s_p = float(variance_components["subject"])
    s_i = float(variance_components["item"])
    s_pi = float(variance_components["subject_item"])
    s_e = float(variance_components["residual"])

    err_relative = s_pi / n_items + s_e / (n_items * n_reps)
    err = (s_i / n_items + err_relative) if type == "absolute" else err_relative

    denom = s_p + err
    if denom < 1e-12:
        return 0.0
    return s_p / denom


def d_study(
    variance_components: dict,
    n_items_grid: Sequence[int],
    n_reps_grid: Sequence[int],
) -> pd.DataFrame:
    """Project G-coefficients and SEs over a (n_items, n_reps) design grid.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`.
    n_items_grid, n_reps_grid : sequence of int
        Candidate design dimensions to project.

    Returns
    -------
    pandas.DataFrame
        One row per (n_items, n_reps) cell with columns ``n_items``,
        ``n_reps``, ``g_relative``, ``g_absolute``, ``se_relative``,
        ``se_absolute``.
    """
    import pandas as pd

    if len(n_items_grid) == 0 or len(n_reps_grid) == 0:
        raise ValueError("n_items_grid and n_reps_grid must be non-empty.")

    s_i = float(variance_components["item"])
    s_pi = float(variance_components["subject_item"])
    s_e = float(variance_components["residual"])

    rows = []
    for n_items in n_items_grid:
        for n_reps in n_reps_grid:
            err_relative = s_pi / n_items + s_e / (n_items * n_reps)
            err_absolute = s_i / n_items + err_relative
            rows.append(
                {
                    "n_items": int(n_items),
                    "n_reps": int(n_reps),
                    "g_relative": g_coefficient(variance_components, n_items, n_reps, "relative"),
                    "g_absolute": g_coefficient(variance_components, n_items, n_reps, "absolute"),
                    "se_relative": float(np.sqrt(err_relative)),
                    "se_absolute": float(np.sqrt(err_absolute)),
                }
            )
    return pd.DataFrame(rows)
