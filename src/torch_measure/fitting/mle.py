# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Maximum Likelihood Estimation for IRT models."""

from __future__ import annotations

import torch

from torch_measure.fitting._losses import bernoulli_nll


def mle_fit(
    model,
    response_matrix: torch.Tensor,
    mask: torch.Tensor,
    max_epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    convergence_tol: float = 1e-6,
    verbose: bool = True,
    optimizer_cls: str = "adam",
    loss_fn=None,
) -> dict:
    """Fit an IRT model via maximum likelihood estimation.

    Minimizes the negative log-likelihood of observed responses under
    a Bernoulli model: L = -sum log P(Y_ij | params) over observed entries.

    Parameters
    ----------
    model : IRTModel
        The IRT model to fit.
    response_matrix : torch.Tensor
        Response matrix (n_subjects, n_items).
    mask : torch.Tensor
        Boolean mask of entries to use for fitting.
    max_epochs : int
        Maximum optimization epochs.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization weight.
    convergence_tol : float
        Stop if loss change is below this threshold.
    verbose : bool
        Show progress bar.
    optimizer_cls : str
        Optimizer: "adam" or "lbfgs".

    Returns
    -------
    dict
        Training history with 'losses' key.
    """
    if loss_fn is None:
        loss_fn = bernoulli_nll

    if optimizer_cls == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    masked_responses = response_matrix[mask].float()
    history = {"losses": []}

    iterator = range(max_epochs)
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="MLE fitting")
        except ImportError:
            pass

    prev_loss = float("inf")

    for _epoch in iterator:
        if optimizer_cls == "lbfgs":

            def closure():
                optimizer.zero_grad()
                probs = model.predict()
                masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
                loss = loss_fn(masked_probs, masked_responses)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = loss.item()
        else:
            optimizer.zero_grad()
            probs = model.predict()
            masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
            loss = loss_fn(masked_probs, masked_responses)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        history["losses"].append(loss_val)

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.6f}"})

        if abs(prev_loss - loss_val) < convergence_tol:
            break
        prev_loss = loss_val

    return history
