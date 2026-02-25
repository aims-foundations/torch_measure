# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Bradley-Terry model for pairwise comparison data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torch_measure.fitting._losses import bernoulli_nll

if TYPE_CHECKING:
    from torch_measure.data.pairwise import PairwiseComparisons


class BradleyTerry(nn.Module):
    """Bradley-Terry model for pairwise comparison data.

    Models the probability that subject *a* beats subject *b* as:

    .. math::

        P(a > b) = \\sigma(\\theta_a - \\theta_b)

    This is mathematically equivalent to the Rasch IRT model, but applied
    to paired comparisons rather than subject × item responses.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (e.g., LLMs).
    device : str
        Device to place parameters on.

    Examples
    --------
    >>> from torch_measure.models import BradleyTerry
    >>> from torch_measure.data import PairwiseComparisons
    >>> model = BradleyTerry(n_subjects=3)
    >>> model.predict()  # (3, 3) win probability matrix
    """

    def __init__(self, n_subjects: int, device: str = "cpu") -> None:
        super().__init__()
        self._n_subjects = n_subjects
        self._device = torch.device(device)
        self.ability = nn.Parameter(torch.zeros(n_subjects, device=self._device))

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return self._n_subjects

    def predict(self) -> torch.Tensor:
        """Compute full pairwise win probability matrix.

        Returns
        -------
        torch.Tensor
            Matrix of shape ``(n_subjects, n_subjects)`` where entry ``(i, j)``
            is ``P(i beats j) = σ(θ_i - θ_j)``. The diagonal is 0.5.
        """
        diff = self.ability.unsqueeze(1) - self.ability.unsqueeze(0)
        return torch.sigmoid(diff)

    def predict_pairwise(self, subject_a: torch.Tensor, subject_b: torch.Tensor) -> torch.Tensor:
        """Compute win probabilities for specific pairs.

        Parameters
        ----------
        subject_a : torch.LongTensor
            Indices of the first subject in each pair, shape ``(n_pairs,)``.
        subject_b : torch.LongTensor
            Indices of the second subject in each pair, shape ``(n_pairs,)``.

        Returns
        -------
        torch.Tensor
            ``P(a beats b)`` for each pair, shape ``(n_pairs,)``.
        """
        return torch.sigmoid(self.ability[subject_a] - self.ability[subject_b])

    def forward(self) -> torch.Tensor:
        """Forward pass returns the full win probability matrix."""
        return self.predict()

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
