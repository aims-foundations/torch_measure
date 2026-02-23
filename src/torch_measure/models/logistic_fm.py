# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Logistic Factor Model for measurement.

Consolidated from factor-model/calibration/model.py LogisticFM.
"""

from __future__ import annotations

import torch
from torch import nn

from torch_measure.models._base import IRTModel


class LogisticFM(IRTModel):
    """K-factor Logistic Factor Model.

    P(correct) = sigmoid(U @ V^T + Z^T) where:
    - U: (n_subjects, K) latent ability factors
    - V: (n_items, K) item loadings on factors
    - Z: (n_items,) item intercepts (easiness)

    When K=1, this is equivalent to the Rasch model.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_items : int
        Number of items.
    n_factors : int
        Number of latent factors (K).
    device : str
        Device to place parameters on.
    """

    def __init__(self, n_subjects: int, n_items: int, n_factors: int = 2, device: str = "cpu") -> None:
        super().__init__(n_subjects, n_items, device)
        self.n_factors = n_factors

        self.U = nn.Parameter(torch.randn(n_subjects, n_factors, device=self._device))
        self.V = nn.Parameter(torch.randn(n_items, n_factors, device=self._device))
        self.Z = nn.Parameter(torch.randn(n_items, device=self._device))

    @property
    def ability(self) -> torch.Tensor:
        """Subject ability factors (n_subjects, K)."""
        return self.U.detach()

    @property
    def difficulty(self) -> torch.Tensor:
        """Item intercepts (n_items,). Negative Z = harder items."""
        return -self.Z.detach()

    @property
    def loadings(self) -> torch.Tensor:
        """Item factor loadings (n_items, K)."""
        return self.V.detach()

    def predict(self) -> torch.Tensor:
        """Compute P(correct) = sigmoid(U @ V^T + Z^T).

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_subjects, n_items).
        """
        return torch.sigmoid(self.U @ self.V.T + self.Z.unsqueeze(0))
