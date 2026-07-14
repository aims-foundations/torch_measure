# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Doubly robust predictor: learns a bias-correction on top of a frozen base model.

The correction is trained with inverse-propensity-weighted (IPW) loss so that
the combined predictor remains consistent under informative missingness (MNAR)
in sparse benchmark matrices.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from torch_measure.models._base import IRTModel


class DoublyRobustModel(IRTModel):
    """Residual IRT model trained with propensity-weighted loss.

    Wraps a pre-trained base model and learns an additive correction:

        P(correct | i, j) = clamp( base(i, j) + correction(i, j) )

    where ``correction(i, j) = sigmoid(alpha_i - beta_j) - 0.5``, a centered
    residual Rasch layer. During fitting, the loss for each observed cell is
    weighted by ``1 / e(i, j)`` where ``e`` is the estimated propensity
    (probability of observation), making the estimator consistent even when
    missingness depends on unobserved outcomes.

    Parameters
    ----------
    base_model : IRTModel
        A fitted IRT model whose parameters will be frozen.
    clip_propensity : tuple[float, float]
        Clamp range for propensity scores to avoid extreme weights.
    """

    def __init__(
        self,
        base_model: IRTModel,
        clip_propensity: tuple[float, float] = (0.05, 0.95),
    ) -> None:
        n_subjects = base_model.n_subjects
        n_items = base_model.n_items
        super().__init__(n_subjects, n_items, device=str(base_model.device))

        self._base = base_model
        for p in self._base.parameters():
            p.requires_grad_(False)

        self._clip_propensity = clip_propensity

        self.correction_ability = nn.Parameter(torch.zeros(n_subjects, device=self._device))
        self.correction_difficulty = nn.Parameter(torch.zeros(n_items, device=self._device))

        self._propensity_weights: torch.Tensor | None = None

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """P(correct) = clamp(base + correction)."""
        s = query["subject_idx"]
        i = query["item_idx"]

        base_prob = self._base.predict(query).detach()
        correction = torch.sigmoid(self.correction_ability[s] - self.correction_difficulty[i]) - 0.5

        return (base_prob + correction).clamp(1e-7, 1 - 1e-7)

    def fit(
        self,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        method: str = "mle",
        max_epochs: int = 500,
        lr: float = 0.01,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Fit the correction layer with IPW-weighted loss.

        Before running the optimizer, estimates propensity scores from the
        observation pattern via logistic regression, then passes per-observation
        weights (1/propensity) into the fitting loop.

        Parameters
        ----------
        data : torch.Tensor
            Wide-form response matrix (n_subjects, n_items). NaN = unobserved.
        mask : torch.Tensor | None
            Boolean observation mask. Inferred from NaN if None.
        method : str
            Fitting backend (default ``"mle"``).
        max_epochs : int
            Optimization epochs for the correction layer.
        lr : float
            Learning rate.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history.
        """
        if mask is None:
            mask = ~torch.isnan(data)

        self._estimate_propensity(data, mask)

        subject_idx, item_idx, response = self._normalize_fit_inputs(data, mask)

        weights = self._get_observation_weights(subject_idx, item_idx)

        def ipw_loss(predicted_probs: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
            per_obs_nll = -observed * torch.log(predicted_probs) - (1 - observed) * torch.log(1 - predicted_probs)
            return (per_obs_nll * weights).mean()

        from torch_measure.fitting.mle import mle_fit

        return mle_fit(
            self,
            subject_idx,
            item_idx,
            response,
            max_epochs=max_epochs,
            lr=lr,
            verbose=verbose,
            loss_fn=ipw_loss,
            **kwargs,
        )

    def _estimate_propensity(self, data: torch.Tensor, mask: torch.Tensor) -> None:
        """Fit a logistic regression on observation indicators."""
        n_s, n_i = data.shape
        obs = mask.float()

        row_rate = obs.mean(dim=1)
        col_rate = obs.mean(dim=0)

        features = torch.stack(
            [
                row_rate.repeat_interleave(n_i),
                col_rate.repeat(n_s),
            ],
            dim=1,
        ).numpy()

        if hasattr(self._base, "ability") and hasattr(self._base, "difficulty"):
            ability = self._base.ability.detach().cpu()
            difficulty = self._base.difficulty.detach().cpu()
            features = np.hstack(
                [
                    features,
                    ability.repeat_interleave(n_i).numpy()[:, None],
                    difficulty.repeat(n_s).numpy()[:, None],
                ]
            )

        y = mask.reshape(-1).numpy().astype(np.int32)

        if y.all() or not y.any():
            self._propensity_weights = torch.ones(n_s, n_i, device=self._device)
            return

        lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=0)
        lr.fit(features, y)
        prop_flat = lr.predict_proba(features)[:, 1]
        propensity = torch.from_numpy(prop_flat).float().reshape(n_s, n_i)
        propensity = propensity.clamp(self._clip_propensity[0], self._clip_propensity[1])

        self._propensity_weights = (1.0 / propensity).to(self._device)

    def _get_observation_weights(self, subject_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Look up per-observation IPW weights."""
        if self._propensity_weights is None:
            return torch.ones(subject_idx.shape[0], device=self._device)
        return self._propensity_weights[subject_idx, item_idx]
