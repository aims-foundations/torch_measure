# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Three-Parameter Logistic (3PL) IRT model."""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class ThreePL(IRTModel):
    """3-Parameter Logistic IRT model.

    P(correct) = c + (1 - c) * sigmoid(a * (theta - b)) where:
    - theta: subject ability
    - b: item difficulty
    - a: item discrimination
    - c: guessing parameter (lower asymptote)

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    device : str
        Device to place parameters on.
    """

    def __init__(self, n_subjects: int, n_items: int, device: str = "cpu") -> None:
        super().__init__(n_subjects, n_items, device)
        self.ability = nn.Parameter(torch.randn(n_subjects, device=self._device))
        self.difficulty = nn.Parameter(torch.randn(n_items, device=self._device))
        self._discrimination_raw = nn.Parameter(torch.randn(n_items, device=self._device))
        self._guessing_raw = nn.Parameter(torch.full((n_items,), -2.0, device=self._device))

    @property
    def discrimination(self) -> torch.Tensor:
        """Item discrimination parameters (constrained positive)."""
        return torch.exp(self._discrimination_raw)

    @property
    def guessing(self) -> torch.Tensor:
        """Item guessing parameters (constrained to [0, 1])."""
        return torch.sigmoid(self._guessing_raw)

    def predict(self) -> torch.Tensor:
        """Compute P(correct) = c + (1-c) * sigmoid(a * (theta - b)).

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        return self._irt_probability(
            self.ability,
            self.difficulty,
            discrimination=self.discrimination,
            guessing=self.guessing,
        )
