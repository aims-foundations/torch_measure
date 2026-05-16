# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Maximum Likelihood Estimation for IRT models (long-form)."""

from __future__ import annotations

import torch

from torch_measure.fitting._losses import bernoulli_nll, cross_entropy_nll


def _is_continuous(response: torch.Tensor) -> bool:
    """True when *response* contains values other than 0 and 1."""
    return not torch.all((response == 0) | (response == 1))


def mle_fit(
    model,
    subject_idx: torch.Tensor,
    item_idx: torch.Tensor,
    response: torch.Tensor,
    max_epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    convergence_tol: float = 1e-6,
    verbose: bool = True,
    optimizer_cls: str = "adam",
    loss_fn=None,
) -> dict:
    """Fit an IRT model via maximum likelihood on long-form observations.

    Minimises the mean negative log-likelihood of observed responses::

        L = -mean log P(Y_k | params)  over observed rows k

    Supports both binary ``{0, 1}`` and continuous ``[0, 1]`` responses.
    When ``loss_fn`` is ``None`` the function chooses between
    :func:`bernoulli_nll` (binary data) and :func:`cross_entropy_nll`
    (continuous data) automatically.

    Parameters
    ----------
    model : IRTModel
        The IRT model to fit. Must expose ``predict(query)`` (see :class:`Predictor`).
    subject_idx : torch.LongTensor
        Integer subject indices, shape ``(n_obs,)``.
    item_idx : torch.LongTensor
        Integer item indices, shape ``(n_obs,)``.
    response : torch.Tensor
        Observed responses, shape ``(n_obs,)``, dtype float. Values may be
        binary ``{0, 1}`` or continuous ``[0, 1]``.
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
        Optimizer: ``"adam"`` or ``"lbfgs"``.
    loss_fn : callable | None
        Loss function ``(predicted, observed) -> scalar``.
        ``None`` (default) auto-selects based on the response values.

    Returns
    -------
    dict
        Training history with ``"losses"`` key.
    """
    if loss_fn is None:
        loss_fn = cross_entropy_nll if _is_continuous(response) else bernoulli_nll

    if optimizer_cls == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    response = response.float()
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
                probs = model.predict({"subject_idx": subject_idx, "item_idx": item_idx}).clamp(1e-7, 1 - 1e-7)
                loss = loss_fn(probs, response)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = loss.item()
        else:
            optimizer.zero_grad()
            probs = model.predict({"subject_idx": subject_idx, "item_idx": item_idx}).clamp(1e-7, 1 - 1e-7)
            loss = loss_fn(probs, response)
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
