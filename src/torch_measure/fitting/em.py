# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Expectation-Maximization fitting for IRT models.

Legacy fitter: E-step marginalises over latent abilities via Gauss-Hermite
quadrature, which genuinely wants a dense ``(n_subjects, n_items)`` matrix
at each quadrature node. Long-form observations are pivoted internally at
``fit()`` entry; the matrix representation is never exposed.

Consolidated from predictive-eval/train/amortized_irt/irt.py em() method.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from torch_measure.fitting._losses import bernoulli_nll
from torch_measure.models._predictor import predict_dense


def _resolve_ability(model: nn.Module) -> tuple[nn.Parameter, str]:
    """Return the ``nn.Parameter`` that holds subject abilities and its name.

    Standard IRT models register ``ability`` as a direct ``nn.Parameter``.
    Models like ``LogisticFM`` store abilities under a different attribute
    (``U``) and declare ``_ability_param_name`` so this helper can find it.
    """
    if isinstance(getattr(model, "ability", None), nn.Parameter):
        return model.ability, "ability"

    attr = getattr(model, "_ability_param_name", None)
    if attr is not None:
        param = getattr(model, attr, None)
        if isinstance(param, nn.Parameter):
            return param, attr

    raise TypeError(
        f"{type(model).__name__} has no 'ability' nn.Parameter and does not "
        f"declare '_ability_param_name'. Cannot use em_fit."
    )


def _pivot_long_to_matrix(
    n_subjects: int,
    n_items: int,
    subject_idx: torch.Tensor,
    item_idx: torch.Tensor,
    response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense ``(n_subjects, n_items)`` matrix + boolean mask from long-form.

    Repeated ``(s, i)`` pairs are averaged (matches the semantics of
    :meth:`torch_measure.data.response_matrix.ResponseMatrix.from_long`).
    """
    device = response.device
    matrix = torch.zeros((n_subjects, n_items), dtype=torch.float32, device=device)
    counts = torch.zeros((n_subjects, n_items), dtype=torch.float32, device=device)
    matrix.index_put_((subject_idx, item_idx), response.float(), accumulate=True)
    counts.index_put_(
        (subject_idx, item_idx),
        torch.ones_like(response, dtype=torch.float32),
        accumulate=True,
    )
    mask = counts > 0
    matrix[mask] = matrix[mask] / counts[mask]
    return matrix, mask


def em_fit(
    model,
    subject_idx: torch.Tensor,
    item_idx: torch.Tensor,
    response: torch.Tensor,
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

    Long-form observations are pivoted to a dense matrix internally — the
    quadrature E-step needs the matrix at every ability node, so the pivot
    is amortised over many epochs.

    Parameters
    ----------
    model : IRTModel
        The IRT model to fit. Must have ``ability`` and ``difficulty``
        parameters.
    subject_idx : torch.LongTensor
        Integer subject indices, shape ``(n_obs,)``.
    item_idx : torch.LongTensor
        Integer item indices, shape ``(n_obs,)``.
    response : torch.Tensor
        Observed responses, shape ``(n_obs,)``, dtype float.
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

    device = response.device
    history = {"losses_item": [], "losses_ability": []}

    response_matrix, mask = _pivot_long_to_matrix(model.n_subjects, model.n_items, subject_idx, item_idx, response)

    # Multiply mean NLL by ``n_obs`` to get summed ``log p`` at each theta node before
    # log-sum-exp.
    n_obs = mask.sum().to(dtype=response_matrix.dtype)

    # Phase 1: Estimate item parameters by marginalizing over abilities
    theta_nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
    theta_nodes = torch.tensor(theta_nodes, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    log_weights = torch.log(weights)

    # Resolve the actual nn.Parameter for abilities
    ability_param, ability_name = _resolve_ability(model)
    if ability_param.dim() > 1 and ability_param.shape[-1] > 1:
        raise ValueError(
            f"em_fit requires 1-dimensional abilities but {type(model).__name__} "
            f"has ability parameter '{ability_name}' with shape "
            f"{tuple(ability_param.shape)}. Use n_factors=1 or method='mle'."
        )

    # Freeze ability, optimize item params
    item_params = [p for name, p in model.named_parameters() if name != ability_name]
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

            quad_log_terms = []
            for q in range(n_quadrature):
                # Set all abilities to this quadrature point
                with torch.no_grad():
                    ability_param.fill_(theta_nodes[q].item())

                probs = predict_dense(model)
                masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
                nll = loss_fn(masked_probs, response_matrix[mask].float())
                quad_log_terms.append(log_weights[q] - n_obs * nll)

            # MML quadrature: ``log(sum_q w_q p(X | theta_q))`` with stable ``log-sum-exp``.
            # With mean NLL each term contributes ``log w_q - n_obs * nll_q``.
            neg_marginal_ll = torch.logsumexp(torch.stack(quad_log_terms), dim=0)
            total_loss = -neg_marginal_ll

            total_loss.backward()
            optimizer_item.step()
            history["losses_item"].append(total_loss.item())

            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{total_loss.item():.6f}"})

    # Phase 2: Estimate abilities given fixed item parameters
    for p in item_params:
        p.requires_grad_(False)
    ability_param.requires_grad_(True)

    # Re-initialize abilities
    with torch.no_grad():
        ability_param.zero_()

    optimizer_ability = torch.optim.Adam([ability_param], lr=lr)

    iterator2 = range(max_epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator2 = tqdm(iterator2, desc="EM: abilities")
        except ImportError:
            pass

    for _ in iterator2:
        optimizer_ability.zero_grad()
        probs = predict_dense(model)
        masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
        loss = loss_fn(masked_probs, response_matrix[mask].float())

        # Regularize abilities toward N(0,1)
        loss = loss + 0.01 * (ability_param.mean().abs() + (ability_param.std() - 1).abs())

        loss.backward()
        optimizer_ability.step()
        history["losses_ability"].append(loss.item())

        if verbose and hasattr(iterator2, "set_postfix"):
            iterator2.set_postfix({"loss": f"{loss.item():.6f}"})

    # Restore gradients
    for p in item_params:
        p.requires_grad_(True)

    return history
