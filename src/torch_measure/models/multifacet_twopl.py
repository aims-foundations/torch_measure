# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Many-Facet 2PL IRT model with anchoring (requires pyro-ppl).

Extends :class:`MultiFacetRasch` with per-item discrimination (alpha) and a
sparse Student-t prior on the item-by-facet interaction (tau). Supports
anchor items whose tau is forced near zero — useful for cross-lingual safety
analysis where a set of "ground truth" items have known invariant difficulty
across languages.

P(correct) = sigmoid(alpha_i * ((theta_n + delta_nl) - (beta_i + gamma_l + tau_il)))

where:
- theta_n: subject ability
- delta_nl: subject-facet aptitude (e.g., model-language ability)
- alpha_i: item discrimination
- beta_i: base item difficulty (reference facet level)
- gamma_l: global facet shift (reference level = 0)
- tau_il: item-facet interaction (reference level = 0; anchors near 0)
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class MultiFacet2PL(IRTModel):
    """Many-Facet 2PL IRT Model with anchoring (Bayesian SVI only).

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    n_facet_levels : int
        Number of levels in the additional facet (e.g., number of languages).
    device : str
        Device to place parameters on.

    Notes
    -----
    Estimation is Bayesian SVI via Pyro. Install with:
    ``pip install torch_measure[bayesian]``
    """

    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        n_facet_levels: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.n_facet_levels = n_facet_levels

        # Posterior-mean storage (filled by .fit(); also used by .predict())
        self.ability = nn.Parameter(torch.zeros(n_subjects, device=self._device))
        self.difficulty = nn.Parameter(torch.zeros(n_items, device=self._device))
        self._discrimination_raw = nn.Parameter(torch.zeros(n_items, device=self._device))
        self.gamma = nn.Parameter(torch.zeros(n_facet_levels, device=self._device))
        self.tau = nn.Parameter(torch.zeros(n_items, n_facet_levels, device=self._device))
        self.delta = nn.Parameter(torch.zeros(n_subjects, n_facet_levels, device=self._device))

        # Reference-level + anchor masks
        self.register_buffer("gamma_mask", torch.ones(n_facet_levels, device=self._device))
        self.register_buffer("tau_mask", torch.ones(n_items, n_facet_levels, device=self._device))
        self.register_buffer("anchor_mask", torch.zeros(n_items, n_facet_levels, device=self._device))

    @property
    def discrimination(self) -> torch.Tensor:
        """Per-item discrimination, constrained positive via ``exp``."""
        return torch.exp(self._discrimination_raw)

    def set_reference_level(self, level_idx: int) -> None:
        """Anchor a facet level to zero (e.g., English baseline).

        Forces ``gamma[level_idx] = 0`` and ``tau[:, level_idx] = 0`` at both
        fit and predict time. Also zeros ``delta[:, level_idx]`` (the subject
        intercept under the reference facet is absorbed by ``ability``).
        """
        self.gamma_mask[level_idx] = 0.0
        self.tau_mask[:, level_idx] = 0.0

    def set_anchor_items(self, item_indices: Sequence[int] | torch.Tensor) -> None:
        """Mark items whose tau should be near zero across all facet levels.

        Anchor items get a tight (sd=0.01) Student-t prior on tau, encoding
        the assumption that their difficulty is invariant across the facet.
        """
        idx = torch.as_tensor(item_indices, dtype=torch.long, device=self._device)
        self.anchor_mask[idx, :] = 1.0

    def predict(self, facet_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Compute response probabilities for one facet level.

        Parameters
        ----------
        facet_indices : torch.Tensor | None
            Single facet level index. If None, uses level 0.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape ``(n_subjects, n_items)``.
        """
        if facet_indices is None:
            facet_indices = torch.zeros(1, dtype=torch.long, device=self._device)

        gamma = self.gamma * self.gamma_mask
        tau = self.tau * self.tau_mask
        delta = self.delta * self.gamma_mask.unsqueeze(0)

        if facet_indices.numel() == 1:
            fl = int(facet_indices.item())
            difficulty_l = self.difficulty + gamma[fl] + tau[:, fl]
            ability_l = self.ability + delta[:, fl]
            logit = self.discrimination.unsqueeze(0) * (
                ability_l.unsqueeze(1) - difficulty_l.unsqueeze(0)
            )
            return torch.sigmoid(logit)

        raise NotImplementedError(
            "Batch facet indices not yet supported. Pass a single facet level."
        )

    def fit(
        self,
        subject_idx: torch.Tensor,
        item_idx: torch.Tensor,
        facet_idx: torch.Tensor,
        response: torch.Tensor,
        max_epochs: int = 4000,
        lr: float = 0.01,
        clip_norm: float = 10.0,
        verbose: bool = True,
        num_posterior_samples: int = 500,
    ) -> dict:
        """Fit via Bayesian SVI (Pyro).

        Long-form quadruple input: each row is one observation
        ``(subject_idx[k], item_idx[k], facet_idx[k]) -> response[k]``.

        Priors:

        - ``ability ~ Normal(0, 1)``
        - ``difficulty ~ Normal(0, 1)``
        - ``discrimination ~ LogNormal(0.5, 0.5)``  (positive)
        - ``gamma_raw ~ Normal(0, 1)``, then ``gamma = gamma_raw * gamma_mask``
        - ``tau_scale ~ HalfCauchy(1)``;
          ``tau_raw ~ StudentT(1, 0, scale)`` with scale=0.01 at anchor cells,
          ``tau_scale`` elsewhere; then ``tau = tau_raw * tau_mask``
        - ``delta_raw ~ Normal(0, 0.5)``, then ``delta = delta_raw * gamma_mask``

        Parameters
        ----------
        subject_idx, item_idx, facet_idx : torch.LongTensor
            Long-form indices, each shape ``(n_obs,)``.
        response : torch.Tensor
            Binary observations, shape ``(n_obs,)``.
        max_epochs : int
            Number of SVI steps.
        lr : float
            Learning rate for ClippedAdam.
        clip_norm : float
            Gradient clipping norm.
        verbose : bool
            Show tqdm progress bar if available.
        num_posterior_samples : int
            Posterior samples for parameter extraction.

        Returns
        -------
        dict
            ``{"losses": list[float], "posterior": {param_name: Tensor}}``
            where ``posterior`` holds the posterior means used to populate the
            model's parameter slots.
        """
        try:
            import pyro
            import pyro.distributions as dist
            import pyro.poutine
            from pyro.infer import SVI, Predictive, Trace_ELBO
            from pyro.infer.autoguide import AutoNormal
            from pyro.optim import ClippedAdam
        except ImportError as err:
            raise ImportError(
                "Bayesian SVI fitting requires pyro-ppl. "
                "Install with: pip install torch_measure[bayesian]"
            ) from err

        device = self._device
        n_subjects = self.n_subjects
        n_items = self.n_items
        n_facets = self.n_facet_levels

        subject_idx = subject_idx.to(device=device, dtype=torch.long)
        item_idx = item_idx.to(device=device, dtype=torch.long)
        facet_idx = facet_idx.to(device=device, dtype=torch.long)
        response = response.to(device=device, dtype=torch.float32)

        gamma_mask = self.gamma_mask
        tau_mask = self.tau_mask
        anchor_mask = self.anchor_mask

        def pyro_model(s_idx, i_idx, f_idx, obs):
            theta = pyro.sample(
                "ability",
                dist.Normal(torch.zeros(n_subjects, device=device), 1.0).to_event(1),
            )
            beta = pyro.sample(
                "difficulty",
                dist.Normal(torch.zeros(n_items, device=device), 1.0).to_event(1),
            )
            alpha = pyro.sample(
                "discrimination",
                dist.LogNormal(
                    torch.full((n_items,), 0.5, device=device),
                    torch.full((n_items,), 0.5, device=device),
                ).to_event(1),
            )

            gamma_raw = pyro.sample(
                "gamma_raw",
                dist.Normal(torch.zeros(n_facets, device=device), 1.0).to_event(1),
            )
            gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)

            tau_scale = pyro.sample(
                "tau_scale",
                dist.HalfCauchy(torch.ones(1, device=device)).to_event(1),
            )
            tau_scale_per = torch.where(
                anchor_mask > 0.5,
                torch.full((n_items, n_facets), 0.01, device=device),
                tau_scale.expand(n_items, n_facets),
            )
            tau_raw = pyro.sample(
                "tau_raw",
                dist.StudentT(
                    1.0,
                    torch.zeros(n_items, n_facets, device=device),
                    tau_scale_per,
                ).to_event(2),
            )
            tau = pyro.deterministic("tau", tau_raw * tau_mask)

            delta_raw = pyro.sample(
                "delta_raw",
                dist.Normal(
                    torch.zeros(n_subjects, n_facets, device=device), 0.5
                ).to_event(2),
            )
            delta_mask = gamma_mask.unsqueeze(0).expand(n_subjects, -1)
            delta = pyro.deterministic("delta", delta_raw * delta_mask)

            with pyro.plate("data", s_idx.shape[0]):
                ability_eff = theta[s_idx] + delta[s_idx, f_idx]
                difficulty_eff = beta[i_idx] + gamma[f_idx] + tau[i_idx, f_idx]
                logit = alpha[i_idx] * (ability_eff - difficulty_eff)
                pyro.sample("response", dist.Bernoulli(logits=logit), obs=obs)

        pyro.clear_param_store()
        guide = AutoNormal(
            pyro.poutine.block(pyro_model, hide=["response", "tau", "gamma", "delta"])
        )
        optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm})
        svi = SVI(pyro_model, guide, optimizer, loss=Trace_ELBO())

        history: dict = {"losses": []}

        iterator = range(max_epochs)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="SVI fitting (MultiFacet2PL)")
            except ImportError:
                pass

        for _ in iterator:
            loss = svi.step(subject_idx, item_idx, facet_idx, response)
            history["losses"].append(loss)
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"ELBO": f"{loss:.2f}"})

        predictive = Predictive(
            pyro_model,
            guide=guide,
            num_samples=num_posterior_samples,
            return_sites=["ability", "difficulty", "discrimination", "gamma", "tau", "delta"],
        )
        samples = predictive(subject_idx, item_idx, facet_idx, None)

        with torch.no_grad():
            ability_mean = samples["ability"].mean(dim=0).reshape(n_subjects)
            difficulty_mean = samples["difficulty"].mean(dim=0).reshape(n_items)
            disc_mean = samples["discrimination"].mean(dim=0).reshape(n_items)
            gamma_mean = samples["gamma"].mean(dim=0).reshape(n_facets)
            tau_mean = samples["tau"].mean(dim=0).reshape(n_items, n_facets)
            delta_mean = samples["delta"].mean(dim=0).reshape(n_subjects, n_facets)

            self.ability.copy_(ability_mean)
            self.difficulty.copy_(difficulty_mean)
            self._discrimination_raw.copy_(torch.log(disc_mean.clamp_min(1e-8)))
            self.gamma.copy_(gamma_mean)
            self.tau.copy_(tau_mean)
            self.delta.copy_(delta_mean)

            history["posterior"] = {
                "ability": ability_mean.detach().clone(),
                "difficulty": difficulty_mean.detach().clone(),
                "discrimination": disc_mean.detach().clone(),
                "gamma": gamma_mean.detach().clone(),
                "tau": tau_mean.detach().clone(),
                "delta": delta_mean.detach().clone(),
            }

        return history
