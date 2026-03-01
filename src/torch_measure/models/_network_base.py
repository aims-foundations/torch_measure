# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Abstract base class for network psychometric models."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class NetworkModel(nn.Module):
    """Abstract base class for network psychometric models.

    Network models characterize the conditional dependence structure among
    items rather than estimating per-subject latent traits. They expose:

    - `.fit(response_matrix, ...)` to estimate network parameters
    - `.adjacency` to access the estimated edge weight matrix
    - `.centrality(measure)` for common node centrality metrics

    Unlike :class:`~torch_measure.models.IRTModel`, there is no notion of
    subjects or per-subject ability — the model is defined entirely over items.
    """

    def __init__(self, n_items: int, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self._n_items = n_items
        self._device = torch.device(device)

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    @abstractmethod
    def adjacency(self) -> torch.Tensor:
        """Edge weight matrix of shape (n_items, n_items).

        Symmetric with zero diagonal. Positive values indicate positive
        conditional dependence; negative values indicate negative dependence.

        Returns
        -------
        torch.Tensor
            Detached weight matrix, shape (n_items, n_items).
        """
        ...

    @abstractmethod
    def fit(
        self,
        response_matrix: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 0.01,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Estimate network parameters from a response matrix.

        Parameters
        ----------
        response_matrix : torch.Tensor
            Response matrix of shape (n_subjects, n_items).
            Use NaN or -1 for missing entries.
        mask : torch.Tensor | None
            Boolean mask of entries to use. Inferred from NaNs if None.
        max_epochs : int
            Maximum optimisation epochs.
        lr : float
            Learning rate.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Training history with ``"losses"`` key.
        """
        ...

    def centrality(self, measure: str = "strength") -> torch.Tensor:
        """Compute node centrality from the estimated adjacency matrix.

        Parameters
        ----------
        measure : str
            One of ``"strength"``, ``"expected_influence"``,
            ``"closeness"``, or ``"betweenness"``.

        Returns
        -------
        torch.Tensor
            Centrality scores per item, shape (n_items,).
        """
        from torch_measure.metrics.network import (
            betweenness_centrality,
            closeness_centrality,
            expected_influence,
            strength_centrality,
        )

        A = self.adjacency
        if measure == "strength":
            return strength_centrality(A)
        elif measure == "expected_influence":
            return expected_influence(A)
        elif measure == "closeness":
            return closeness_centrality(A)
        elif measure == "betweenness":
            return betweenness_centrality(A)
        else:
            raise ValueError(
                f"Unknown centrality measure: {measure!r}. "
                "Choose from 'strength', 'expected_influence', 'closeness', 'betweenness'."
            )
