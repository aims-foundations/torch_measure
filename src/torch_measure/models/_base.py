# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Base class for all IRT models."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class IRTModel(nn.Module):
    """Abstract base class for Item Response Theory models.

    All IRT models share the interface:
    - `.fit(response_matrix, mask, ...)` to estimate parameters
    - `.predict()` to compute response probabilities
    - `.ability` to access subject ability parameters
    - `.difficulty` to access item difficulty parameters
    """

    def __init__(self, n_subjects: int, n_items: int, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self._n_subjects = n_subjects
        self._n_items = n_items
        self._device = torch.device(device)

    @property
    def n_subjects(self) -> int:
        return self._n_subjects

    @property
    def n_items(self) -> int:
        return self._n_items

    @abstractmethod
    def predict(self) -> torch.Tensor:
        """Compute predicted response probabilities.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        ...

    def forward(self) -> torch.Tensor:
        """Forward pass returns predicted probabilities."""
        return self.predict()

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
        """Fit the model to a response matrix.

        Parameters
        ----------
        response_matrix : torch.Tensor
            Binary response matrix of shape (n_subjects, n_items).
            Use NaN or -1 for missing data.
        mask : torch.Tensor | None
            Boolean mask of entries to use for fitting. If None, uses all
            non-NaN entries.
        method : str
            Fitting method: "mle", "em", "jml", or "svi" (requires pyro-ppl).
        max_epochs : int
            Maximum number of optimization epochs.
        lr : float
            Learning rate.
        verbose : bool
            Whether to show a progress bar.

        Returns
        -------
        dict
            Training history with loss values.
        """
        from torch_measure.fitting.mle import mle_fit

        response_matrix = response_matrix.to(self._device)
        if mask is None:
            mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
        mask = mask.to(self._device)

        if method == "mle":
            return mle_fit(self, response_matrix, mask, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        elif method == "em":
            from torch_measure.fitting.em import em_fit

            return em_fit(self, response_matrix, mask, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        elif method == "jml":
            from torch_measure.fitting.jml import jml_fit

            return jml_fit(self, response_matrix, mask, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        elif method == "svi":
            from torch_measure.fitting.svi import svi_fit

            return svi_fit(self, response_matrix, mask, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unknown fitting method: {method!r}. Use 'mle', 'em', 'jml', or 'svi'.")

    @staticmethod
    def _irt_probability(
        ability: torch.Tensor,
        difficulty: torch.Tensor,
        discrimination: torch.Tensor | None = None,
        guessing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute IRT probability P(correct | ability, item_params).

        Implements the general IRT formula:
            P = c + (1 - c) * sigmoid(a * (theta - b))

        where theta=ability, b=difficulty, a=discrimination, c=guessing.

        Parameters
        ----------
        ability : torch.Tensor
            Subject abilities of shape (N,) or (N, D).
        difficulty : torch.Tensor
            Item difficulties of shape (M,).
        discrimination : torch.Tensor | None
            Item discriminations of shape (M,). Defaults to 1.
        guessing : torch.Tensor | None
            Item guessing parameters of shape (M,). Defaults to 0.

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (N, M).
        """
        # ability: (N,) or (N, D) -> (N, 1)
        if ability.ndim == 1:
            ability = ability.unsqueeze(1)
        # difficulty: (M,) -> (1, M)
        difficulty = difficulty.unsqueeze(0)

        logit = ability - difficulty  # (N, M)

        if discrimination is not None:
            logit = discrimination.unsqueeze(0) * logit

        prob = torch.sigmoid(logit)

        if guessing is not None:
            prob = guessing.unsqueeze(0) + (1 - guessing.unsqueeze(0)) * prob

        return prob
