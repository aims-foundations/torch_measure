# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Joint Maximum Likelihood estimation for factor models (long-form)."""

from __future__ import annotations

import torch

from torch_measure.fitting._losses import bernoulli_nll


def jml_fit(
    model,
    subject_idx: torch.Tensor,
    item_idx: torch.Tensor,
    response: torch.Tensor,
    max_epochs: int = 500,
    lr: float = 0.1,
    regularization: float = 0.01,
    convergence_tol: float = 1e-6,
    verbose: bool = True,
    loss_fn=None,
    **kwargs,
) -> dict:
    """Fit a model via Joint Maximum Likelihood with LBFGS.

    Jointly estimates all parameters (abilities and item params) by
    minimizing the Bernoulli NLL plus an L2 regularization term scaled
    by the parameter count.

    Parameters
    ----------
    model : IRTModel or LogisticFM
        Model to fit. Must expose ``predict_at(s_idx, i_idx)``.
    subject_idx : torch.LongTensor
        Integer subject indices, shape ``(n_obs,)``.
    item_idx : torch.LongTensor
        Integer item indices, shape ``(n_obs,)``.
    response : torch.Tensor
        Observed responses, shape ``(n_obs,)``, dtype float.
    max_epochs : int
        Maximum LBFGS iterations.
    lr : float
        LBFGS learning rate.
    regularization : float
        L2 regularization strength (lambda).
    convergence_tol : float
        Stop when loss change is below this.
    verbose : bool
        Show progress.

    Returns
    -------
    dict
        Training history with ``"losses"`` key.
    """
    if loss_fn is None:
        loss_fn = bernoulli_nll

    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    response = response.float()
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
            probs = model.predict_at(subject_idx, item_idx).clamp(1e-7, 1 - 1e-7)
            nll = loss_fn(probs, response)

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
