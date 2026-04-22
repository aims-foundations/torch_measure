# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Joint Maximum Likelihood estimation for factor models.

Consolidated from factor-model/calibration/model.py JML_trainer.
"""

from __future__ import annotations

import torch

from torch_measure.fitting._losses import bernoulli_nll


def jml_fit(
    model,
    response_matrix: torch.Tensor,
    mask: torch.Tensor,
    max_epochs: int = 500,
    lr: float = 0.1,
    regularization: float = 0.01,
    convergence_tol: float = 1e-6,
    verbose: bool = True,
    loss_fn=None,
    **kwargs,
) -> dict:
    """Fit a model via Joint Maximum Likelihood with LBFGS.

    JML jointly estimates all parameters (abilities and item params)
    simultaneously. Uses LBFGS optimizer with optional L2 regularization.

    Parameters
    ----------
    model : IRTModel or LogisticFM
        Model to fit.
    response_matrix : torch.Tensor
        Response matrix (n_subjects, n_items).
    mask : torch.Tensor
        Boolean mask of observed entries.
    max_epochs : int
        Maximum LBFGS iterations.
    lr : float
        LBFGS learning rate.
    regularization : float
        L2 regularization strength (Lambda).
    convergence_tol : float
        Stop when loss change is below this.
    verbose : bool
        Show progress.

    Returns
    -------
    dict
        Training history.
    """
    if loss_fn is None:
        loss_fn = bernoulli_nll

    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    masked_responses = response_matrix[mask].float()
    history = {"losses": []}
    prev_loss = float("inf")

    iterator = range(max_epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="JML fitting")
        except ImportError:
            pass

    for _ in iterator:

        def closure():
            optimizer.zero_grad()
            probs = model.predict()
            masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
            nll = loss_fn(masked_probs, masked_responses)

            # L2 regularization on all parameters
            reg = sum(p.pow(2).sum() for p in model.parameters())
            total_params = sum(p.numel() for p in model.parameters())
            loss = nll + regularization * reg / max(total_params, 1)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_val = loss.item()
        history["losses"].append(loss_val)

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.6f}"})

        if abs(prev_loss - loss_val) < convergence_tol:
            break
        prev_loss = loss_val

    return history
