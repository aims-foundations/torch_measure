# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Two-Parameter Logistic (2PL) IRT model."""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class TwoPL(IRTModel):
    """2-Parameter Logistic IRT model.

    P(correct) = sigmoid(a * (theta - b)) where:
    - theta: subject ability
    - b: item difficulty
    - a: item discrimination (how well the item differentiates abilities)

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

    @property
    def discrimination(self) -> torch.Tensor:
        """Item discrimination parameters (constrained to be positive)."""
        return torch.exp(self._discrimination_raw)

    def predict(self) -> torch.Tensor:
        """Compute P(correct) = sigmoid(a * (theta - b)).

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        return self._irt_probability(self.ability, self.difficulty, discrimination=self.discrimination)
