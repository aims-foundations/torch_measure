# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Stochastic Variational Inference fitting via Pyro (optional dependency).

Consolidated from safety-irt/model/irt.py.
Requires: pip install torch-measure[bayesian]
"""

from __future__ import annotations

import torch


def _detect_model_type(model):
    """Detect model type from its attributes.

    Returns
    -------
    str
        One of "1pl", "2pl", "3pl".
    """
    has_disc = hasattr(model, "discrimination") or hasattr(model, "_discrimination_raw")
    has_guess = hasattr(model, "guessing") or hasattr(model, "_guessing_raw")
    if has_guess:
        return "3pl"
    if has_disc:
        return "2pl"
    return "1pl"


def _is_beta_model(model):
    """Check if the model uses Beta likelihood."""
    return hasattr(model, "phi")


def svi_fit(
    model,
    response_matrix: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_epochs: int = 4000,
    lr: float = 0.01,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Fit an IRT model via Stochastic Variational Inference using Pyro.

    Supports Rasch (1PL), 2PL, 3PL, and their Beta variants.
    Uses N(0,1) priors on ability and difficulty, LogNormal(0,0.5) on
    discrimination, and Beta(1,4) on guessing.

    Parameters
    ----------
    model : IRTModel
        The IRT model to fit (Rasch, TwoPL, ThreePL, BetaRasch, BetaTwoPL).
    response_matrix : torch.Tensor
        Response matrix (n_subjects, n_items). Binary for standard IRT,
        continuous in (0,1) for Beta IRT.
    mask : torch.Tensor | None
        Boolean mask of observed entries.
    max_epochs : int
        Number of SVI steps.
    lr : float
        Learning rate for ClippedAdam.
    verbose : bool
        Show progress.

    Returns
    -------
    dict
        Training history with 'losses' key (ELBO values).
    """
    try:
        import pyro
        import pyro.distributions as dist
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoNormal
        from pyro.optim import ClippedAdam
    except ImportError as err:
        raise ImportError(
            "Bayesian SVI fitting requires pyro-ppl. "
            "Install with: pip install torch-measure[bayesian]"
        ) from err

    if mask is None:
        mask = ~torch.isnan(response_matrix) & (response_matrix != -1)

    device = response_matrix.device
    observed_responses = response_matrix[mask].float()
    n_subjects = response_matrix.shape[0]
    n_items = response_matrix.shape[1]

    # Extract indices of observed entries
    obs_indices = mask.nonzero(as_tuple=False)
    subject_idx = obs_indices[:, 0]
    item_idx = obs_indices[:, 1]

    model_type = _detect_model_type(model)
    beta_model = _is_beta_model(model)
    phi = getattr(model, "phi", 10.0) if beta_model else None

    pyro.clear_param_store()

    def pyro_model(subject_idx, item_idx, obs):
        # Priors on ability and difficulty
        ability = pyro.sample(
            "ability",
            dist.Normal(torch.zeros(n_subjects, device=device), 1.0).to_event(1),
        )
        difficulty = pyro.sample(
            "difficulty",
            dist.Normal(torch.zeros(n_items, device=device), 1.0).to_event(1),
        )

        # Compute logit
        logit = ability[subject_idx] - difficulty[item_idx]

        # 2PL / 3PL: add discrimination prior
        if model_type in ("2pl", "3pl"):
            discrimination = pyro.sample(
                "discrimination",
                dist.LogNormal(torch.zeros(n_items, device=device), 0.5).to_event(1),
            )
            logit = discrimination[item_idx] * logit

        # 3PL: add guessing prior
        if model_type == "3pl":
            guessing = pyro.sample(
                "guessing",
                dist.Beta(
                    torch.ones(n_items, device=device),
                    4.0 * torch.ones(n_items, device=device),
                ).to_event(1),
            )
            prob = guessing[item_idx] + (1 - guessing[item_idx]) * torch.sigmoid(logit)
        else:
            prob = torch.sigmoid(logit)

        # Likelihood
        with pyro.plate("obs", len(obs)):
            if beta_model:
                mu = prob.clamp(1e-6, 1 - 1e-6)
                a = mu * phi
                b = (1.0 - mu) * phi
                pyro.sample("response", dist.Beta(a, b), obs=obs)
            else:
                pyro.sample("response", dist.Bernoulli(probs=prob), obs=obs)

    guide = AutoNormal(pyro_model)
    optimizer = ClippedAdam({"lr": lr})
    svi = SVI(pyro_model, guide, optimizer, loss=Trace_ELBO())

    history = {"losses": []}

    iterator = range(max_epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="SVI fitting")
        except ImportError:
            pass

    for _ in iterator:
        loss = svi.step(subject_idx, item_idx, observed_responses)
        history["losses"].append(loss)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"ELBO": f"{loss:.2f}"})

    # Extract posterior means and update model parameters
    with torch.no_grad():
        ability_loc = pyro.param("AutoNormal.locs.ability")
        difficulty_loc = pyro.param("AutoNormal.locs.difficulty")
        if hasattr(model, "ability") and isinstance(model.ability, torch.nn.Parameter):
            model.ability.copy_(ability_loc)
        if hasattr(model, "difficulty") and isinstance(model.difficulty, torch.nn.Parameter):
            model.difficulty.copy_(difficulty_loc)

        if model_type in ("2pl", "3pl"):
            disc_loc = pyro.param("AutoNormal.locs.discrimination")
            if hasattr(model, "_discrimination_raw") and isinstance(model._discrimination_raw, torch.nn.Parameter):
                # AutoNormal stores the unconstrained value; LogNormal's loc is the log-mean
                # The posterior mean in the constrained space is exp(loc + scale^2/2),
                # but the raw parameter is log(discrimination), so we store loc directly.
                model._discrimination_raw.copy_(disc_loc)

        if model_type == "3pl":
            guess_loc = pyro.param("AutoNormal.locs.guessing")
            if hasattr(model, "_guessing_raw") and isinstance(model._guessing_raw, torch.nn.Parameter):
                # AutoNormal stores unconstrained value; convert back to logit space
                # The guide samples in unconstrained space, so this is already logit-like
                model._guessing_raw.copy_(guess_loc)

    return history
