# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Expectation-Maximization fitting for IRT models.

Consolidated from predictive-eval/train/amortized_irt/irt.py em() method.
"""

from __future__ import annotations

import numpy as np
import torch

from torch_measure.fitting._losses import bernoulli_nll


def em_fit(
    model,
    response_matrix: torch.Tensor,
    mask: torch.Tensor,
    max_epochs: int = 500,
    lr: float = 0.01,
    n_quadrature: int = 31,
    verbose: bool = True,
    loss_fn=None,
    **kwargs,
) -> dict:
    """Fit an IRT model via Expectation-Maximization.

    E-step: Integrate over latent abilities using Gauss-Hermite quadrature.
    M-step: Maximize likelihood over item parameters given expected abilities.
    Then estimate abilities given fixed item parameters.

    Parameters
    ----------
    model : IRTModel
        The IRT model to fit. Must have 'ability' and 'difficulty' parameters.
    response_matrix : torch.Tensor
        Response matrix (n_subjects, n_items).
    mask : torch.Tensor
        Boolean mask of observed entries.
    max_epochs : int
        Maximum epochs per phase (item params, then abilities).
    lr : float
        Learning rate for Adam optimizer.
    n_quadrature : int
        Number of Gauss-Hermite quadrature points.
    verbose : bool
        Show progress.

    Returns
    -------
    dict
        Training history with 'losses_item' and 'losses_ability' keys.
    """
    if loss_fn is None:
        loss_fn = bernoulli_nll

    device = response_matrix.device
    history = {"losses_item": [], "losses_ability": []}

    # Phase 1: Estimate item parameters by marginalizing over abilities
    theta_nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
    theta_nodes = torch.tensor(theta_nodes, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()

    # Freeze ability, optimize item params
    item_params = [p for name, p in model.named_parameters() if "ability" not in name]
    if item_params:
        optimizer_item = torch.optim.Adam(item_params, lr=lr)

        iterator = range(max_epochs)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="EM: item params")
            except ImportError:
                pass

        for _ in iterator:
            optimizer_item.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for q in range(n_quadrature):
                # Set all abilities to this quadrature point
                with torch.no_grad():
                    model.ability.fill_(theta_nodes[q].item())

                probs = model.predict()
                masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
                nll = loss_fn(masked_probs, response_matrix[mask].float())
                total_loss = total_loss + weights[q] * nll

            total_loss.backward()
            optimizer_item.step()
            history["losses_item"].append(total_loss.item())

            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{total_loss.item():.6f}"})

    # Phase 2: Estimate abilities given fixed item parameters
    for p in item_params:
        p.requires_grad_(False)
    model.ability.requires_grad_(True)

    # Re-initialize abilities
    with torch.no_grad():
        model.ability.zero_()

    optimizer_ability = torch.optim.Adam([model.ability], lr=lr)

    iterator2 = range(max_epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator2 = tqdm(iterator2, desc="EM: abilities")
        except ImportError:
            pass

    for _ in iterator2:
        optimizer_ability.zero_grad()
        probs = model.predict()
        masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
        loss = loss_fn(masked_probs, response_matrix[mask].float())

        # Regularize abilities toward N(0,1)
        loss = loss + 0.01 * (model.ability.mean().abs() + (model.ability.std() - 1).abs())

        loss.backward()
        optimizer_ability.step()
        history["losses_ability"].append(loss.item())

        if verbose and hasattr(iterator2, "set_postfix"):
            iterator2.set_postfix({"loss": f"{loss.item():.6f}"})

    # Restore gradients
    for p in item_params:
        p.requires_grad_(True)

    return history
