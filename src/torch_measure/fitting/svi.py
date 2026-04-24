# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Stochastic Variational Inference fitting via Pyro (optional dependency).

Consolidated from safety-irt/model/irt.py.
Requires: pip install torch_measure[bayesian]
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


def _is_testlet_model(model):
    """Check if the model has testlet random effects."""
    return hasattr(model, "testlet_effect") and hasattr(model, "testlet_map")


def svi_fit(
    model,
    subject_idx: torch.Tensor,
    item_idx: torch.Tensor,
    response: torch.Tensor,
    max_epochs: int = 4000,
    lr: float = 0.01,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Fit an IRT model via Stochastic Variational Inference using Pyro.

    Supports Rasch (1PL), 2PL, 3PL, their Beta variants, and Testlet models.
    Uses N(0,1) priors on ability and difficulty, LogNormal(0,0.5) on
    discrimination, Beta(1,4) on guessing, and a hierarchical
    LogNormal/Normal prior on testlet effects.

    Operates natively on long-form observations: each row k contributes
    one likelihood term ``P(response[k] | ability[subject_idx[k]],
    difficulty[item_idx[k]], ...)``.

    Parameters
    ----------
    model : IRTModel
        The IRT model to fit (Rasch, TwoPL, ThreePL, BetaRasch, BetaTwoPL,
        TestletRasch). Binary-IRT models use Bernoulli likelihood;
        :attr:`model.phi` marks Beta-IRT variants.
    subject_idx : torch.LongTensor
        Integer subject indices, shape ``(n_obs,)``.
    item_idx : torch.LongTensor
        Integer item indices, shape ``(n_obs,)``.
    response : torch.Tensor
        Observed responses, shape ``(n_obs,)``. Binary for standard IRT,
        continuous in ``(0,1)`` for Beta IRT.
    max_epochs : int
        Number of SVI steps.
    lr : float
        Learning rate for ClippedAdam.
    verbose : bool
        Show progress.

    Returns
    -------
    dict
        Training history with 'losses' key (ELBO values) and 'posterior' key
        containing variational posterior parameters. The posterior dict maps
        parameter names to dicts with 'loc' and 'scale' tensors:

        - ``"ability"``: loc (N,), scale (N,)
        - ``"difficulty"``: loc (M,), scale (M,)
        - ``"discrimination"`` (2PL/3PL only): loc (M,), scale (M,) — in log-space
        - ``"guessing"`` (3PL only): loc (M,), scale (M,) — in logit-space
        - ``"testlet_effect"`` (testlet only): loc (N, T), scale (N, T)
        - ``"testlet_scale"`` (testlet only): loc (T,), scale (T,) — in log-space

        Scales are obtained by applying ``softplus`` to AutoNormal's unconstrained
        scale parameters.
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
            "Install with: pip install torch_measure[bayesian]"
        ) from err

    device = response.device
    observed_responses = response.float()
    n_subjects = model.n_subjects
    n_items = model.n_items

    model_type = _detect_model_type(model)
    beta_model = _is_beta_model(model)
    phi = getattr(model, "phi", 10.0) if beta_model else None
    testlet_model = _is_testlet_model(model)
    testlet_map = getattr(model, "testlet_map", None)
    n_testlets = getattr(model, "n_testlets", 0) if testlet_model else 0

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

        # Testlet random effects (hierarchical prior)
        if testlet_model:
            testlet_scale = pyro.sample(
                "testlet_scale",
                dist.LogNormal(
                    torch.full((n_testlets,), -0.7, device=device),
                    0.5 * torch.ones(n_testlets, device=device),
                ).to_event(1),
            )
            testlet_effect = pyro.sample(
                "testlet_effect",
                dist.Normal(
                    torch.zeros(n_subjects, n_testlets, device=device),
                    testlet_scale.unsqueeze(0).expand(n_subjects, -1),
                ).to_event(2),
            )
            logit = logit + testlet_effect[subject_idx, testlet_map[item_idx]]

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

        if testlet_model:
            te_loc = pyro.param("AutoNormal.locs.testlet_effect")
            if hasattr(model, "testlet_effect") and isinstance(model.testlet_effect, torch.nn.Parameter):
                model.testlet_effect.copy_(te_loc)

    # Extract posterior distributions (loc + scale) from AutoNormal guide
    with torch.no_grad():
        posterior = {}

        ability_scale = torch.nn.functional.softplus(pyro.param("AutoNormal.scales.ability"))
        posterior["ability"] = {
            "loc": ability_loc.detach().clone(),
            "scale": ability_scale.detach().clone(),
        }

        difficulty_scale = torch.nn.functional.softplus(pyro.param("AutoNormal.scales.difficulty"))
        posterior["difficulty"] = {
            "loc": difficulty_loc.detach().clone(),
            "scale": difficulty_scale.detach().clone(),
        }

        if model_type in ("2pl", "3pl"):
            disc_scale = torch.nn.functional.softplus(pyro.param("AutoNormal.scales.discrimination"))
            posterior["discrimination"] = {
                "loc": disc_loc.detach().clone(),
                "scale": disc_scale.detach().clone(),
            }

        if model_type == "3pl":
            guess_scale = torch.nn.functional.softplus(pyro.param("AutoNormal.scales.guessing"))
            posterior["guessing"] = {
                "loc": guess_loc.detach().clone(),
                "scale": guess_scale.detach().clone(),
            }

        if testlet_model:
            te_scale = torch.nn.functional.softplus(pyro.param("AutoNormal.scales.testlet_effect"))
            posterior["testlet_effect"] = {
                "loc": te_loc.detach().clone(),
                "scale": te_scale.detach().clone(),
            }
            ts_loc = pyro.param("AutoNormal.locs.testlet_scale")
            ts_scale = torch.nn.functional.softplus(pyro.param("AutoNormal.scales.testlet_scale"))
            posterior["testlet_scale"] = {
                "loc": ts_loc.detach().clone(),
                "scale": ts_scale.detach().clone(),
            }

        history["posterior"] = posterior

    return history
