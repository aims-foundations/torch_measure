# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Bradley-Terry model for pairwise comparison data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torch_measure.fitting._losses import bernoulli_nll
from torch_measure.models._predictor import Predictor

if TYPE_CHECKING:
    from torch_measure.data.pairwise import PairwiseComparisons


class BradleyTerry(Predictor):
    """Bradley-Terry model for pairwise comparison data.

    Models the probability that subject *a* beats subject *b* as:

    .. math::

        P(a > b) = \\sigma(\\theta_a - \\theta_b)

    Mathematically equivalent to Rasch, but the "item" axis is itself a
    subject — so ``predict(query)`` consumes ``subject_idx`` (the A-side)
    and ``item_idx`` (the B-side).

    Parameters
    ----------
    n_subjects : int
        Number of subjects (e.g., LLMs).
    device : str
        Device to place parameters on.

    Examples
    --------
    >>> from torch_measure.models import BradleyTerry
    >>> from torch_measure.models._predictor import predict_dense
    >>> model = BradleyTerry(n_subjects=3)
    >>> predict_dense(model)  # (3, 3) win probability matrix
    """

    def __init__(self, n_subjects: int, device: str = "cpu") -> None:
        # Both axes of the prediction are subjects; pass n_subjects twice.
        super().__init__(n_subjects, n_subjects, device)
        self.ability = nn.Parameter(torch.zeros(n_subjects, device=self._device))

    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute P(a beats b) at query rows.

        ``query["subject_idx"]`` is the A-side; ``query["item_idx"]`` is
        the B-side (also a subject index).
        """
        a = query["subject_idx"]
        b = query["item_idx"]
        return torch.sigmoid(self.ability[a] - self.ability[b])

    def predict_pairwise(self, subject_a: torch.Tensor, subject_b: torch.Tensor) -> torch.Tensor:
        """Domain-named convenience: ``P(a beats b)`` for explicit pair tensors.

        Equivalent to ``self.predict({"subject_idx": subject_a, "item_idx": subject_b})``.
        """
        return self.predict({"subject_idx": subject_a, "item_idx": subject_b})

    def fit(
        self,
        comparisons: PairwiseComparisons,
        method: str = "mle",
        max_epochs: int = 1000,
        lr: float = 0.01,
        regularization: float = 0.01,
        convergence_tol: float = 1e-6,
        verbose: bool = True,
    ) -> dict:
        """Fit the model to pairwise comparison data.

        Parameters
        ----------
        comparisons : PairwiseComparisons
            Pairwise comparison data with ``subject_a``, ``subject_b``,
            and ``outcome`` tensors.
        method : str
            Fitting method: ``"mle"`` (Adam optimizer) or
            ``"jml"`` (LBFGS with L2 regularization).
        max_epochs : int
            Maximum number of optimization epochs.
        lr : float
            Learning rate.
        regularization : float
            L2 regularization weight (only used for ``method="jml"``).
        convergence_tol : float
            Stop if loss change is below this threshold.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history with ``'losses'`` key.
        """
        subject_a = comparisons.subject_a.to(self._device)
        subject_b = comparisons.subject_b.to(self._device)
        outcome = comparisons.outcome.to(self._device)

        if method == "jml":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20)
        elif method == "mle":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'mle' or 'jml'.")

        history: dict[str, list] = {"losses": []}

        iterator = range(max_epochs)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc=f"BT {method.upper()} fitting")
            except ImportError:
                pass

        prev_loss = float("inf")

        for _epoch in iterator:
            if method == "jml":

                def closure():
                    optimizer.zero_grad()
                    probs = self.predict_pairwise(subject_a, subject_b).clamp(1e-7, 1 - 1e-7)
                    loss = bernoulli_nll(probs, outcome)
                    loss = loss + regularization * self.ability.pow(2).mean()
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                loss_val = loss.item()
            else:
                optimizer.zero_grad()
                probs = self.predict_pairwise(subject_a, subject_b).clamp(1e-7, 1 - 1e-7)
                loss = bernoulli_nll(probs, outcome)
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

    def __repr__(self) -> str:
        return f"BradleyTerry(n_subjects={self._n_subjects})"
