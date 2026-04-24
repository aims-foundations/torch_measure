# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Abstract base class for network psychometric models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch_measure.datasets._long_form import LongFormData


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
        data,
        mask: torch.Tensor | None = None,
        max_epochs: int = 1000,
        lr: float = 0.01,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Estimate network parameters.

        Parameters
        ----------
        data : LongFormData | torch.Tensor
            Long-form dataset (preferred) or wide-form response tensor of
            shape ``(n_subjects, n_items)``. For wide-form, missing entries
            may be encoded as NaN or -1.
        mask : torch.Tensor | None
            Only used with wide-form input — boolean mask of entries to use.
            Inferred from NaNs if None.
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

    def _normalize_fit_inputs_to_matrix(
        self,
        data,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coerce ``data`` to a dense ``(n_subjects, n_items)`` matrix + mask.

        Network models genuinely want the matrix (for item-item covariance,
        conditional probability via matrix multiply, etc.), so long-form
        inputs are pivoted internally. Repeated ``(s, i)`` pairs are
        averaged.
        """
        from torch_measure.datasets._long_form import LongFormData

        if isinstance(data, LongFormData):
            fit_inputs = data.to_fit_tensors(device=str(self._device))
            subject_idx = fit_inputs["subject_idx"]
            item_idx = fit_inputs["item_idx"]
            response = fit_inputs["response"]
            n_subjects = len(fit_inputs["subject_ids"])
            n_items = len(fit_inputs["item_ids"])

            matrix = torch.zeros((n_subjects, n_items), dtype=torch.float32,
                                 device=self._device)
            counts = torch.zeros((n_subjects, n_items), dtype=torch.float32,
                                 device=self._device)
            matrix.index_put_((subject_idx, item_idx), response.float(),
                              accumulate=True)
            counts.index_put_(
                (subject_idx, item_idx),
                torch.ones_like(response, dtype=torch.float32),
                accumulate=True,
            )
            built_mask = counts > 0
            matrix[built_mask] = matrix[built_mask] / counts[built_mask]
            return matrix, built_mask

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"fit() expected LongFormData or torch.Tensor, got {type(data).__name__}"
            )

        response_matrix = data.to(self._device)
        if mask is None:
            mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
        mask = mask.to(self._device)
        return response_matrix, mask

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
