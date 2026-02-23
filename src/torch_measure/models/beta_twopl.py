# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Beta-2PL IRT model for continuous (0, 1) responses."""

from __future__ import annotations

from functools import partial

import torch

from torch_measure.models.twopl import TwoPL


class BetaTwoPL(TwoPL):
    """Beta-2PL IRT model.

    Identical to TwoPL in prediction: ``mu = sigmoid(a * (theta - b))``.
    Uses Beta NLL loss instead of Bernoulli NLL for fitting, allowing
    continuous responses in (0, 1) such as empirical probabilities.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (test-takers / models).
    n_items : int
        Number of items (test questions / benchmark tasks).
    phi : float
        Beta distribution precision parameter. Higher values mean
        tighter concentration around the predicted mean. Default 10.0.
    device : str
        Device to place parameters on.

    References
    ----------
    .. [1] Item Response Scaling Laws (ICML 2026).
    """

    def __init__(
        self, n_subjects: int, n_items: int, phi: float = 10.0, device: str = "cpu"
    ) -> None:
        super().__init__(n_subjects, n_items, device)
        self.phi = phi

    def fit(
        self,
        response_matrix: torch.Tensor,
        mask: torch.Tensor | None = None,
        method: str = "mle",
        max_epochs: int = 1000,
        lr: float = 0.01,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Fit the Beta-2PL model using Beta NLL loss.

        Parameters
        ----------
        response_matrix : torch.Tensor
            Continuous response matrix with values in (0, 1),
            shape (n_subjects, n_items). Values must be strictly
            between 0 and 1 (exclusive).
        mask : torch.Tensor | None
            Boolean mask of entries to use. If None, uses all non-NaN entries.
        method : str
            Fitting method: "mle", "em", or "jml".
        max_epochs : int
            Maximum optimization epochs.
        lr : float
            Learning rate.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history.
        """
        from torch_measure.fitting._losses import beta_nll

        loss_fn = partial(beta_nll, phi=self.phi)
        return super().fit(
            response_matrix,
            mask=mask,
            method=method,
            max_epochs=max_epochs,
            lr=lr,
            verbose=verbose,
            loss_fn=loss_fn,
            **kwargs,
        )
