# Copyright (c) 2026 AIMS Foundations. MIT License.

"""ARD-regularised training loop for the ARAF amortized factor model.

Ported from ``agent-eval/model/amortized_irt.py:train_amortized_irt`` and
``adapt_amortized_irt_users``. Two optimizer groups (AdamW for ``theta``,
``theta_bias``, ``W``, ``global_bias``, ``difficulty_proj``; SGD on
``tau_raw``), an ARD warmup → linear ramp → full-λ schedule, periodic
"snapping" of inactive dims to a frozen dead-zone value, and a likelihood
switch between Bernoulli (binary responses) and Beta (continuous in (0, 1)).
"""

from __future__ import annotations

import torch

# ARAF defaults — agent-eval/model/amortized_irt.py constants (see paper).
TAU_INIT: float = 0.5
TAU_WARMUP: int = 100
RAMP_EPOCHS: int = 400
SNAPPING_THRESHOLD: float = 1e-3
DEAD_ZONE_VALUE: float = -0.1
LR_THETA: float = 0.01
LR_GLOBAL: float = 0.002
LR_TAU: float = 0.05
WD_THETA: float = 5.0
WD_W: float = 0.1


def _masked_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``values`` with per-row ``weights`` (already gathered)."""
    total_weight = weights.sum().clamp_min(1e-12)
    return (values * weights).sum() / total_weight


def _likelihood_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    likelihood: str,
    beta_phi: float,
) -> torch.Tensor:
    """Per-cell negative log-likelihood, vector-valued for downstream weighting."""
    if likelihood == "beta":
        eps = 1e-6
        y = targets.clamp(eps, 1 - eps)
        dist = torch.distributions.Beta(probs * beta_phi, (1 - probs) * beta_phi)
        return -dist.log_prob(y)
    if likelihood == "bernoulli":
        dist = torch.distributions.Bernoulli(probs=probs)
        return -dist.log_prob(targets)
    raise ValueError(f"Unknown likelihood {likelihood!r}; choose 'bernoulli' or 'beta'.")


def araf_fit(
    model,
    response_matrix: torch.Tensor,
    mask: torch.Tensor,
    likelihood: str = "bernoulli",
    beta_phi: float = 10.0,
    epochs: int = 1000,
    lambda_tau: float = 1.38,
    lr_theta: float = LR_THETA,
    lr_global: float = LR_GLOBAL,
    lr_tau: float = LR_TAU,
    wd_theta: float = WD_THETA,
    wd_w: float = WD_W,
    tau_warmup: int = TAU_WARMUP,
    ramp_epochs: int = RAMP_EPOCHS,
    snapping_threshold: float = SNAPPING_THRESHOLD,
    dead_zone_value: float = DEAD_ZONE_VALUE,
    row_weights: torch.Tensor | None = None,
    verbose: bool = True,
) -> dict:
    """Fit an :class:`~torch_measure.models.ARAF` model with ARD sparsity.

    Parameters
    ----------
    model : ARAF
        Model with item embeddings already registered via
        :meth:`~torch_measure.models.ARAF.set_embeddings`.
    response_matrix : torch.Tensor
        Observed responses, shape ``(n_subjects, n_items)``. Cells where
        ``mask`` is ``False`` are ignored.
    mask : torch.Tensor
        Boolean mask of observed cells, shape ``(n_subjects, n_items)``.
    likelihood : str
        ``"bernoulli"`` or ``"beta"``.
    beta_phi : float
        Beta precision (only used when ``likelihood="beta"``).
    epochs : int
        Number of training epochs.
    lambda_tau : float
        Final L1 weight on ``τ``. The schedule starts at 0 for
        ``tau_warmup`` epochs, then ramps linearly to ``lambda_tau`` over
        ``ramp_epochs``.
    lr_theta, lr_global, lr_tau : float
        Learning rates for the {θ, θ_bias} group, the {W, global_bias,
        difficulty_proj} group, and the SGD optimizer on ``tau_raw``.
    wd_theta, wd_w : float
        Weight decay for the θ and W groups.
    tau_warmup : int
        Epochs of zero λ before the ramp begins.
    ramp_epochs : int
        Linear-ramp length.
    snapping_threshold : float
        Below this τ value a dimension is snapped to ``dead_zone_value``
        every 10 epochs after ``tau_warmup + 50``.
    dead_zone_value : float
        Value written into ``tau_raw`` for snapped (inactive) dims.
    row_weights : torch.Tensor | None
        Per-subject sample weights, shape ``(n_subjects,)``. ``None`` ⇒ ones.
    verbose : bool
        Show a progress bar.

    Returns
    -------
    dict
        Training history with ``"losses"`` and ``"active_dims"`` per epoch.
    """
    device = model.device
    response_matrix = response_matrix.to(device).float()
    mask = mask.to(device).bool()

    if row_weights is None:
        row_weights = torch.ones(response_matrix.shape[0], device=device)
    else:
        row_weights = row_weights.to(device).float()

    optimizer = torch.optim.AdamW(
        [
            {"params": [model.theta, model.theta_bias], "lr": lr_theta, "weight_decay": wd_theta},
            {"params": [model.W, model.global_bias], "lr": lr_global, "weight_decay": wd_w},
            {"params": list(model.difficulty_proj.parameters()), "lr": lr_global},
        ]
    )
    optimizer_tau = torch.optim.SGD([model.tau_raw], lr=lr_tau)

    eps = 1e-6
    history: dict = {"losses": [], "active_dims": []}

    iterator = range(epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="ARAF fitting")
        except ImportError:
            pass

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        optimizer_tau.zero_grad()

        probs_full = model.dense_predict()
        p = probs_full[mask].clamp(eps, 1 - eps)
        targets = response_matrix[mask]
        observed_weights = row_weights.unsqueeze(1).expand_as(response_matrix)[mask]

        per_cell_nll = _likelihood_loss(p, targets, likelihood=likelihood, beta_phi=beta_phi)
        loss_fit = _masked_weighted_mean(per_cell_nll, observed_weights)

        if epoch < tau_warmup:
            current_lambda = 0.0
        elif epoch < tau_warmup + ramp_epochs:
            current_lambda = lambda_tau * (epoch - tau_warmup) / max(ramp_epochs, 1)
        else:
            current_lambda = lambda_tau

        tau = model.get_tau()
        loss_sparsity = current_lambda * tau.sum()
        total = loss_fit + loss_sparsity
        total.backward()

        optimizer.step()
        optimizer_tau.step()

        if epoch > tau_warmup + 50 and epoch % 10 == 0:
            with torch.no_grad():
                inactive = model.get_tau() <= snapping_threshold
                if inactive.any():
                    model.tau_raw.data[inactive] = dead_zone_value

        loss_val = float(loss_fit.detach().item())
        n_active = int((model.get_tau() > snapping_threshold).sum().item())
        history["losses"].append(loss_val)
        history["active_dims"].append(n_active)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.4f}", "active": n_active})

    model.eval()
    return history


def araf_adapt(
    model,
    response_matrix: torch.Tensor,
    mask: torch.Tensor,
    likelihood: str = "bernoulli",
    beta_phi: float = 10.0,
    epochs: int | None = None,
    lr_theta: float = LR_THETA,
    wd_theta: float = WD_THETA,
    row_weights: torch.Tensor | None = None,
    verbose: bool = True,
) -> dict:
    """Refit a trained ARAF model's ``θ`` for new agents (item params frozen).

    Sets ``requires_grad=False`` on every parameter except ``theta`` and
    ``theta_bias``, then runs AdamW for ``epochs`` steps. The model is
    expected to have been resized so that ``theta``/``theta_bias`` have the
    new ``n_subjects`` first dimension.

    Parameters
    ----------
    model : ARAF
        Already trained ARAF model with embeddings registered.
    response_matrix, mask : torch.Tensor
        Support responses and observed-cell mask, shape
        ``(n_support_subjects, n_items)``.
    likelihood, beta_phi : see :func:`araf_fit`.
    epochs : int | None
        Default ``max(200, 500)``.
    lr_theta, wd_theta : float
        AdamW config for the unfrozen group.
    row_weights : torch.Tensor | None
        Per-subject support weights.
    verbose : bool
        Show progress bar.

    Returns
    -------
    dict
        Training history.
    """
    device = model.device
    response_matrix = response_matrix.to(device).float()
    mask = mask.to(device).bool()
    if epochs is None:
        epochs = max(200, 500)

    for name, param in model.named_parameters():
        param.requires_grad = name in {"theta", "theta_bias"}

    optimizer = torch.optim.AdamW(
        [{"params": [model.theta, model.theta_bias], "lr": lr_theta, "weight_decay": wd_theta}]
    )

    if row_weights is None:
        row_weights = torch.ones(response_matrix.shape[0], device=device)
    else:
        row_weights = row_weights.to(device).float()

    eps = 1e-6
    history: dict = {"losses": []}

    iterator = range(epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="ARAF adapting")
        except ImportError:
            pass

    for _epoch in iterator:
        model.train()
        optimizer.zero_grad()
        probs_full = model.dense_predict()
        p = probs_full[mask].clamp(eps, 1 - eps)
        targets = response_matrix[mask]
        observed_weights = row_weights.unsqueeze(1).expand_as(response_matrix)[mask]
        per_cell_nll = _likelihood_loss(p, targets, likelihood=likelihood, beta_phi=beta_phi)
        loss = _masked_weighted_mean(per_cell_nll, observed_weights)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        history["losses"].append(loss_val)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.4f}"})

    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    return history
