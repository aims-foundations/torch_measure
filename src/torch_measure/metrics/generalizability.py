# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Generalizability-theory reliability for crossed random-facet designs.

Supports both two-way (person x item x replication) and fully crossed multi-facet designs via keyword-only arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from torch_measure.metrics._generalizability import (
    _anova_components,
    _bootstrap_resample,
    _build_cell_means,
    _compute_ms_error,
    _g_coefficients,
    _parse_interactions,
    _validate_design,
)

if TYPE_CHECKING:
    import pandas as pd


def variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    method: str = "moments",
    *,
    facet_cols: Sequence[str] | None = None,
    object_facet: str | None = None,
    interactions: str | Sequence = "all",
) -> dict:
    """Decompose Var(response) into subject, item, subject x item, and residual facets.

    Henderson Method I (moments-based ANOVA estimator) on a person x item x
    replication crossed design. Negative variance estimates are clamped to 0.
    With one observation per cell, residual is unidentifiable.

    When *facet_cols* is provided, estimates components for a balanced, fully
    crossed multi-facet design instead.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses with columns ``subject_col``, ``item_col``,
        ``trial_col``, ``response_col``.
    subject_col, item_col, trial_col, response_col : str
        Column names; defaults match the measurement-db long-form schema.
    method : {"moments"}
        Only ``"moments"`` is implemented in v1.
    facet_cols : sequence of str, keyword-only, optional
        Column names for each facet (>= 2). Activates multi-facet mode.
        The design must be fully crossed and balanced.
    object_facet : str, keyword-only, optional
        Which facet is the object of measurement.
        Required when *facet_cols* is provided. Must be one of *facet_cols*.
    interactions : ``"all"`` or sequence of tuples, keyword-only, optional
        ``"all"`` estimates every interaction; pass a list of tuples to
        select specific ones (main effects are always included).

    Returns
    -------
    dict
        **Two-facet mode:**

        Keys: ``subject``, ``item``, ``subject_item``, ``residual`` (variances,
        floats), ``n_subjects``, ``n_items`` (ints), ``n_reps_harmonic``
        (float; harmonic mean of cell counts), ``identifiable`` (dict[str,
        bool]), ``method`` (str).

        **Multi-facet mode:**

        Keys: ``components`` (dict keyed by facet name (str) for main effects
        or tuple for interactions, plus ``"residual"``), ``object_facet``
        (str), ``facets`` (list[str]), ``n_levels`` (dict[str, int]),
        ``n_reps`` (int), ``identifiable`` (dict[str | tuple, bool]),
        ``method`` (str).
    """
    import pandas as pd

    if method == "reml":
        raise NotImplementedError("method='reml' not implemented in v1.")
    if method != "moments":
        raise ValueError(f"Unknown method: {method!r}.")

    # --- multi-facet path ---
    if facet_cols is not None:
        facet_list = list(facet_cols)
        if len(facet_list) < 2:
            raise ValueError("facet_cols must contain at least 2 facets.")
        if object_facet is None:
            raise ValueError("object_facet is required when facet_cols is given.")
        if object_facet not in facet_list:
            raise ValueError(f"object_facet={object_facet!r} not in facet_cols.")
        df = response_matrix[facet_list + [response_col]].dropna(subset=[response_col])
        n_levels, n_reps, sorted_levels = _validate_design(df, facet_list, response_col)
        means = _build_cell_means(df, facet_list, response_col, n_levels, sorted_levels)
        ms_e = _compute_ms_error(df, facet_list, response_col, n_levels, n_reps)
        selected = _parse_interactions(interactions, facet_list)
        components, identifiability, raw_components = _anova_components(
            means, facet_list, n_levels, n_reps, ms_e, selected
        )
        return {
            "components": components,
            "raw_components": raw_components,
            "object_facet": object_facet,
            "facets": facet_list,
            "n_levels": n_levels,
            "n_reps": n_reps,
            "identifiable": identifiability,
            "method": "moments",
        }

    # --- two-facet path ---
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
    n_items: int | None = None,
    n_reps: int | None = None,
    type: str = "absolute",
    *,
    facet_sizes: dict[str, int] | None = None,
) -> float:
    """Brennan (2001) G-coefficient under a crossed random-facet design.

    Relative G uses ranking-only error (interactions with the object of
    measurement + residual); absolute G (Phi) also includes all other
    main effects.

    For subject x item designs, pass *n_items* (and optionally *n_reps*).
    For designs with 3+ facets, pass *facet_sizes* instead.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`.
    n_reps : int, optional
        Projected within-cell replications (>= 1). Defaults to 1 in
        two-facet mode, or the observed value in multi-facet mode.
    type : {"relative", "absolute"}
        Which G-coefficient to compute.

    *Two-facet parameters:*

    n_items : int, optional
        Number of items in the projected design (>= 1).

    *Multi-facet parameters:*

    facet_sizes : dict[str, int], keyword-only, optional
        Projected sizes for every non-object facet. The object facet is
        read from *variance_components* automatically.

    Returns
    -------
    float
        G-coefficient in [0, 1]. 0.0 if the denominator is numerically zero.
    """
    if type not in {"relative", "absolute"}:
        raise ValueError(f"type must be 'relative' or 'absolute'; got {type!r}.")
    if facet_sizes is not None and n_items is not None:
        raise TypeError("Pass either facet_sizes (multi-facet) or n_items/n_reps (two-facet), not both.")

    # --- multi-facet path ---
    if facet_sizes is not None:
        for v in facet_sizes.values():
            if v < 1:
                raise ValueError(f"All facet_sizes must be >= 1; got {facet_sizes}.")
        if n_reps is not None and n_reps < 1:
            raise ValueError(f"n_reps must be >= 1; got {n_reps}.")
        sigma_p, rel_err, abs_err = _g_coefficients(variance_components, facet_sizes, n_reps=n_reps)
        err = abs_err if type == "absolute" else rel_err
        denom = sigma_p + err
        return sigma_p / denom if denom > 1e-12 else 0.0

    # --- two-facet path ---
    if n_reps is None:
        n_reps = 1
    if n_items is None:
        raise TypeError("n_items is required when facet_sizes is not provided.")

    required = {"subject", "item", "subject_item", "residual"}
    missing = required - set(variance_components)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}.")
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


def intraclass_correlation(
    variance_components: dict,
    form: str = "ICC3k",
    n_items: int | None = None,
) -> float:
    """Intraclass correlation coefficient from two-way variance components.

    Subjects are targets, items are raters. ICC2/ICC3 are single-rater
    (absolute agreement / consistency); ICC2k/ICC3k average over k raters and
    equal the absolute / relative :func:`g_coefficient` at ``n_reps=1``.
    One-way forms (ICC1) need a one-way model and are not supported here.
    Multi-facet variance components are not supported; use
    :func:`g_coefficient` with *facet_sizes* instead.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`, or any dict with keys
        ``subject``, ``item``, ``subject_item``, ``residual``.
    form : {"ICC2", "ICC3", "ICC2k", "ICC3k"}
        Which coefficient to compute. The ``k`` forms average over raters.
    n_items : int | None
        Number of raters k for the ``k`` forms; defaults to
        ``variance_components["n_items"]``.

    Returns
    -------
    float
        ICC in [0, 1]. 0.0 if the denominator is numerically zero.
    """
    if "facets" in variance_components:
        raise TypeError(
            "intraclass_correlation() does not support multi-facet variance "
            "components. Use g_coefficient() with facet_sizes instead."
        )

    required = {"subject", "item", "subject_item", "residual"}
    missing = required - set(variance_components)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}.")
    if form in {"ICC1", "ICC1k"}:
        raise ValueError(f"{form} requires a one-way model and is not supported; use ICC2/ICC3/ICC2k/ICC3k.")
    if form not in {"ICC2", "ICC3", "ICC2k", "ICC3k"}:
        raise ValueError(f"Unknown form: {form!r}. Expected one of ICC2, ICC3, ICC2k, ICC3k.")

    s_p = float(variance_components["subject"])
    s_i = float(variance_components["item"])
    s_pi = float(variance_components["subject_item"])
    s_e = float(variance_components["residual"])

    averaged = form.endswith("k")
    absolute = form in {"ICC2", "ICC2k"}

    if averaged:
        k = n_items if n_items is not None else int(variance_components["n_items"])
        if k < 1:
            raise ValueError(f"n_items must be >= 1; got {k}.")
    else:
        k = 1

    err = (s_i + s_pi + s_e) if absolute else (s_pi + s_e)
    err = err / k

    denom = s_p + err
    if denom < 1e-12:
        return 0.0
    return s_p / denom


def d_study(
    variance_components: dict,
    n_items_grid: Sequence[int] | None = None,
    n_reps_grid: Sequence[int] | None = None,
    *,
    design_grid: dict[str, Sequence[int]] | None = None,
    n_reps_grid_multi: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Project G-coefficients and SEs over a design grid.

    For subject x item designs, pass *n_items_grid* and *n_reps_grid*.
    For designs with 3+ facets, pass *design_grid* instead.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`.

    *Two-facet parameters:*

    n_items_grid : sequence of int, optional
        Candidate numbers of items to project.
    n_reps_grid : sequence of int, optional
        Candidate numbers of within-cell replications to project.

    *Multi-facet parameters:*

    design_grid : dict[str, sequence of int], keyword-only, optional
        Mapping from non-object facet name to candidate sizes.
    n_reps_grid_multi : sequence of int, keyword-only, optional
        Candidate within-cell replications for multi-facet mode.
        Defaults to ``[observed_n_reps]``.

    Returns
    -------
    pandas.DataFrame
        One row per design cell with columns ``g_relative``, ``g_absolute``,
        ``se_relative``, ``se_absolute``.

        **Two-facet mode:** also includes ``n_items``, ``n_reps``.

        **Multi-facet mode:** one ``n_<facet>`` column per non-object facet,
        plus ``n_reps`` if *n_reps_grid_multi* is provided.
    """
    from itertools import product as iterproduct

    import pandas as pd

    if design_grid is not None and (n_items_grid is not None or n_reps_grid is not None):
        raise TypeError("Pass either design_grid (multi-facet) or n_items_grid/n_reps_grid (two-facet), not both.")

    # --- multi-facet path ---
    if design_grid is not None:
        facet_names = list(design_grid.keys())
        for f in facet_names:
            if len(design_grid[f]) == 0:
                raise ValueError(f"design_grid[{f!r}] must be non-empty.")

        reps_list = list(n_reps_grid_multi) if n_reps_grid_multi is not None else [None]
        grids = [design_grid[f] for f in facet_names]
        rows = []
        for combo in iterproduct(*grids):
            ss = dict(zip(facet_names, combo, strict=True))
            for nr in reps_list:
                sigma_p, rel_err, abs_err = _g_coefficients(variance_components, ss, n_reps=nr)
                g_rel = sigma_p / (sigma_p + rel_err) if (sigma_p + rel_err) > 1e-12 else 0.0
                g_abs = sigma_p / (sigma_p + abs_err) if (sigma_p + abs_err) > 1e-12 else 0.0
                row = {f"n_{f}": int(v) for f, v in ss.items()}
                if n_reps_grid_multi is not None:
                    row["n_reps"] = int(nr)
                row.update(
                    {
                        "g_relative": g_rel,
                        "g_absolute": g_abs,
                        "se_relative": float(np.sqrt(rel_err)),
                        "se_absolute": float(np.sqrt(abs_err)),
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)

    # --- two-facet path ---
    if n_items_grid is None or n_reps_grid is None:
        raise TypeError("n_items_grid and n_reps_grid are required when design_grid is not provided.")
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


def bayesian_variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    n_warmup: int = 200,
    n_samples: int = 500,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Full-Bayes variance components via Hamiltonian Monte Carlo (textbook Ch 5).

    Fits the hierarchical Bernoulli model
        logit P(Y_ij=1) = mu + theta_i - beta_j + gamma_ij
    where theta_i ~ N(0, sigma_p), beta_j ~ N(0, sigma_i), gamma_ij ~ N(0, sigma_pi),
    and sigma_p, sigma_i, sigma_pi ~ HalfNormal(1) using NUTS (No-U-Turn Sampler).

    Returns the same top-level keys as variance_components() so all downstream
    functions (g_coefficient, d_study, intraclass_correlation) work unchanged.
    Unlike variance_components(), residual is always 0.0 and unidentifiable:
    the Bernoulli likelihood has no separate additive noise term, so there is
    nothing analogous to observation-level residual variance to estimate,
    regardless of how many replications per cell are present.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses with columns subject_col, item_col, trial_col, response_col.
    subject_col, item_col, trial_col, response_col : str
        Column names; defaults match measurement-db long-form schema.
    n_warmup : int
        NUTS warmup (adaptation) steps. 200 is the minimum; 500+ is safer for
        large datasets.
    n_samples : int
        Post-warmup samples to draw. 500 gives stable posterior means; 1000+ for
        accurate credible intervals.
    seed : int | None
        Pyro RNG seed for reproducibility.
    verbose : bool
        Show NUTS progress bar.

    Returns
    -------
    dict
        Top-level keys match variance_components() for drop-in compatibility:
        subject, item, subject_item, residual, n_subjects, n_items,
        n_reps_harmonic, identifiable, method.
        Additional keys: posterior_samples, credible_intervals, diagnostics,
        n_warmup, n_samples, ci_level.
    """
    import pandas as pd
    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.infer import MCMC, NUTS

    # Non-centered hierarchical Bayesian G-theory model (textbook Ch 5, logit link).
    # Non-centered form: theta_i = sigma_p * z_p[i] instead of theta_i ~ N(0, sigma_p).
    # Required for NUTS to avoid funnel geometry when sigma_pi is near zero.
    def _gtheory_model(s_idx, i_idx, n_s, n_i, obs=None):
        mu = pyro.sample("mu", dist.Normal(0.0, 2.0))
        sigma_p = pyro.sample("sigma_p", dist.HalfNormal(1.0))
        sigma_i = pyro.sample("sigma_i", dist.HalfNormal(1.0))
        sigma_pi = pyro.sample("sigma_pi", dist.HalfNormal(1.0))
        z_p = pyro.sample("z_p", dist.Normal(0.0, 1.0).expand([n_s]).to_event(1))
        z_i = pyro.sample("z_i", dist.Normal(0.0, 1.0).expand([n_i]).to_event(1))
        z_pi = pyro.sample("z_pi", dist.Normal(0.0, 1.0).expand([n_s, n_i]).to_event(2))
        # Textbook formula: logit P(Y_ij=1) = mu + theta_i - beta_j + gamma_ij
        logit = mu + sigma_p * z_p[s_idx] - sigma_i * z_i[i_idx] + sigma_pi * z_pi[s_idx, i_idx]
        with pyro.plate("obs", s_idx.shape[0]):
            pyro.sample("response", dist.Bernoulli(logits=logit), obs=obs)

    # -- Input validation (mirrors variance_components()) --------------------
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

    n_obs_per_cell = df.groupby([subject_col, item_col])[response_col].count()
    if len(n_obs_per_cell) < n_p * n_i:
        raise ValueError(
            f"Unbalanced design: {len(n_obs_per_cell)}/{n_p * n_i} cells observed. "
            f"Every (subject, item) cell must have at least one observation."
        )

    counts = n_obs_per_cell.to_numpy(dtype=float)
    n_r = float(len(counts) / np.sum(1.0 / counts))  # harmonic mean of cell counts

    # -- Index encoding -------------------------------------------------------
    subject_ids = sorted(df[subject_col].unique())
    item_ids = sorted(df[item_col].unique())
    s_to_idx = {s: k for k, s in enumerate(subject_ids)}
    i_to_idx = {it: k for k, it in enumerate(item_ids)}

    subject_idx = torch.tensor([s_to_idx[s] for s in df[subject_col]], dtype=torch.long)
    item_idx = torch.tensor([i_to_idx[it] for it in df[item_col]], dtype=torch.long)
    response = torch.tensor(df[response_col].to_numpy(dtype="float32"), dtype=torch.float32)

    # -- NUTS sampling -------------------------------------------------------
    if seed is not None:
        pyro.set_rng_seed(seed)

    kernel = NUTS(_gtheory_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        warmup_steps=n_warmup,
        disable_progbar=not verbose,
    )
    mcmc.run(subject_idx, item_idx, n_p, n_i, obs=response)

    # -- Extract posterior samples -------------------------------------------
    raw = mcmc.get_samples()
    sigma_p_samp = raw["sigma_p"].numpy()
    sigma_i_samp = raw["sigma_i"].numpy()
    sigma_pi_samp = raw["sigma_pi"].numpy()

    sigma2_p_samp = sigma_p_samp**2
    sigma2_i_samp = sigma_i_samp**2
    sigma2_pi_samp = sigma_pi_samp**2

    alpha = 0.025
    ci_p = (float(np.quantile(sigma2_p_samp, alpha)), float(np.quantile(sigma2_p_samp, 1 - alpha)))
    ci_i = (float(np.quantile(sigma2_i_samp, alpha)), float(np.quantile(sigma2_i_samp, 1 - alpha)))
    ci_pi = (float(np.quantile(sigma2_pi_samp, alpha)), float(np.quantile(sigma2_pi_samp, 1 - alpha)))

    # -- Diagnostics ---------------------------------------------------------
    diag = mcmc.diagnostics()

    def _stat(site: str, key: str) -> float:
        if site not in diag:
            return float("nan")
        v = diag[site][key]
        return float(torch.as_tensor(v).float().mean().item())

    # diag["divergences"] maps each chain to the list of sample indices where
    # NUTS reported a divergent transition. (Pyro's MCMC has no
    # get_extra_fields() -- that is a NumPyro-only API.)
    n_divergences = sum(len(steps) for steps in diag.get("divergences", {}).values())

    return {
        # Drop-in compatible keys (same as variance_components() output)
        "subject": float(sigma2_p_samp.mean()),
        "item": float(sigma2_i_samp.mean()),
        "subject_item": float(sigma2_pi_samp.mean()),
        "residual": 0.0,
        "n_subjects": n_p,
        "n_items": n_i,
        "n_reps_harmonic": n_r,
        "identifiable": {
            "subject": True,
            "item": True,
            "subject_item": True,
            "residual": False,
        },
        "method": "hmc",
        # Bayesian-specific keys
        "posterior_samples": {
            "sigma_p": sigma_p_samp,
            "sigma_i": sigma_i_samp,
            "sigma_pi": sigma_pi_samp,
            "sigma2_p": sigma2_p_samp,
            "sigma2_i": sigma2_i_samp,
            "sigma2_pi": sigma2_pi_samp,
        },
        "credible_intervals": {
            "subject": ci_p,
            "item": ci_i,
            "subject_item": ci_pi,
        },
        "diagnostics": {
            "r_hat": {
                "sigma_p": _stat("sigma_p", "r_hat"),
                "sigma_i": _stat("sigma_i", "r_hat"),
                "sigma_pi": _stat("sigma_pi", "r_hat"),
            },
            "n_eff": {
                "sigma_p": _stat("sigma_p", "n_eff"),
                "sigma_i": _stat("sigma_i", "n_eff"),
                "sigma_pi": _stat("sigma_pi", "n_eff"),
            },
            "divergences": n_divergences,
        },
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "ci_level": 0.95,
    }


def bootstrap_variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    method: str = "moments",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int | None = None,
    *,
    facet_cols: Sequence[str] | None = None,
    object_facet: str | None = None,
    interactions: str | Sequence = "all",
) -> dict:
    """Nonparametric bootstrap CIs for variance components.

    Each bootstrap draw resamples subjects (or *object_facet* levels in
    multi-facet mode) with replacement, relabels duplicates so each
    draw is treated as a distinct unit, and re-fits :func:`variance_components`.
    Percentile CIs are reported. The full bootstrap distribution is also returned so callers
    can derive CIs for any function of the components (e.g. :func:`g_coefficient`).

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses, same schema as :func:`variance_components`.
    subject_col, item_col, trial_col, response_col, method : str
        Forwarded to :func:`variance_components`.
    n_boot : int
        Number of bootstrap replicates (>= 1).
    ci : float
        Confidence level in (0, 1).
    seed : int | None
        Seed for ``numpy.random.default_rng``.
    facet_cols : sequence of str, keyword-only, optional
        Activates multi-facet mode.
    object_facet : str, keyword-only, optional
        Object of measurement (required with *facet_cols*).
    interactions : ``"all"`` or sequence, keyword-only, optional
        Forwarded to :func:`variance_components`.

    Returns
    -------
    dict
        All keys from :func:`variance_components`, plus:

        - ``"ci"`` — percentile CIs per component (dict[key, tuple[float, float]]).
        - ``"samples"`` — bootstrap draws per component (dict[key, ndarray]).
        - ``"n_boot"`` (int), ``"ci_level"`` (float).

        In multi-facet mode, ``ci`` and ``samples`` are keyed by component
        name (str or tuple), matching the ``"components"`` dict.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}.")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}.")

    if facet_cols is not None:
        # --- multi-facet ---
        if object_facet is None:
            raise ValueError("object_facet is required when facet_cols is given.")

        vc_kwargs: dict = dict(
            response_col=response_col,
            method=method,
            facet_cols=facet_cols,
            object_facet=object_facet,
            interactions=interactions,
        )
        point = variance_components(response_matrix, **vc_kwargs)

        samples_np, ci_dict, n_boot_out, ci_level = _bootstrap_resample(
            df=response_matrix,
            object_col=object_facet,
            vc_caller=lambda df: variance_components(df, **vc_kwargs),
            extract_components=lambda vc: vc["components"],
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            point_estimate=point,
        )
    else:
        # --- two-facet ---
        vc_kwargs = dict(
            subject_col=subject_col,
            item_col=item_col,
            trial_col=trial_col,
            response_col=response_col,
            method=method,
        )
        point = variance_components(response_matrix, **vc_kwargs)

        _component_keys = ("subject", "item", "subject_item", "residual")
        samples_np, ci_dict, n_boot_out, ci_level = _bootstrap_resample(
            df=response_matrix,
            object_col=subject_col,
            vc_caller=lambda df: variance_components(df, **vc_kwargs),
            extract_components=lambda vc: {k: vc[k] for k in _component_keys},
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            point_estimate=point,
        )

    return {
        **point,
        "ci": ci_dict,
        "samples": samples_np,
        "n_boot": n_boot_out,
        "ci_level": ci_level,
    }
