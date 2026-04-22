# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Rasch (1PL) Item Response Theory model."""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class Rasch(IRTModel):
    """Rasch (1-Parameter Logistic) IRT model.

    The simplest IRT model where P(correct) = sigmoid(theta - b):
    - theta: subject ability (one per subject)
    - b: item difficulty (one per item)

    No discrimination or guessing parameters.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (test-takers / models).
    n_items : int
        Number of items (test questions / benchmark tasks).
    device : str
        Device to place parameters on.
    """

    def __init__(self, n_subjects: int, n_items: int, device: str = "cpu") -> None:
        super().__init__(n_subjects, n_items, device)
        self.ability = nn.Parameter(torch.randn(n_subjects, device=self._device))
        self.difficulty = nn.Parameter(torch.randn(n_items, device=self._device))

    def predict(self) -> torch.Tensor:
        """Compute P(correct) = sigmoid(ability - difficulty).

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        return self._irt_probability(self.ability, self.difficulty)
