# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Abstract base for any model that predicts P(correct) over long-form observations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import nn


class Predictor(nn.Module, ABC):
    """Base class for any model producing P(correct) over (subject, item) cells.

    Subclasses implement :meth:`predict`, which accepts a long-form query
    (a dict of 1-D index tensors) and returns one probability per row.
    :meth:`forward` is a thin wrapper that delegates to :meth:`predict`,
    so ``model(query)`` works via :meth:`nn.Module.__call__`.

    Each subclass declares the keys it consumes via :attr:`expected_keys`.
    The default ``("subject_idx", "item_idx")`` covers every IRT-style
    model; condition-aware or trial-aware models extend it.
    """

    expected_keys: ClassVar[tuple[str, ...]] = ("subject_idx", "item_idx")

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

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def predict(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict P(correct) for each row of ``query``.

        Parameters
        ----------
        query : dict[str, torch.Tensor]
            Must contain a 1-D tensor for each name in :attr:`expected_keys`,
            all of equal length ``N``. Extra keys are ignored.

        Returns
        -------
        torch.Tensor
            Predicted probabilities, shape ``(N,)`` on the model's device.
        """
        ...

    def forward(self, query: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.predict(query)


def cartesian_query(
    n_subjects: int,
    n_items: int,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Build the (subject, item) Cartesian-product query of size ``n_subjects * n_items``.

    Useful when a caller wants the dense matrix of predictions; see
    :func:`predict_dense` for the common shortcut.

    Parameters
    ----------
    n_subjects, n_items : int
        Universe sizes.
    device : str or torch.device or None
        Device for the returned tensors. ``None`` uses the torch default.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"subject_idx": LongTensor (n_subjects*n_items,), "item_idx":
        LongTensor (n_subjects*n_items,)}``. Row order is subject-major:
        all of subject 0's items first, then subject 1's items, etc.
    """
    s = torch.arange(n_subjects, device=device).repeat_interleave(n_items)
    i = torch.arange(n_items, device=device).repeat(n_subjects)
    return {"subject_idx": s, "item_idx": i}


def predict_dense(model: Predictor, **extra_keys: torch.Tensor) -> torch.Tensor:
    """Predict over the full ``(n_subjects, n_items)`` Cartesian grid.

    Convenience wrapper around :func:`cartesian_query` + ``model.predict``,
    reshaped back to a ``(n_subjects, n_items)`` matrix. Use this for
    visualization, EM quadrature, and other callers that genuinely want
    the dense view.

    Parameters
    ----------
    model : Predictor
        Any predictor with a ``(n_subjects, n_items)`` universe.
    **extra_keys : torch.Tensor
        Additional query columns required by the model's :attr:`expected_keys`
        beyond ``subject_idx`` / ``item_idx``. Each must be 1-D of length
        ``n_subjects * n_items``.

    Returns
    -------
    torch.Tensor
        Probability matrix of shape ``(n_subjects, n_items)``.
    """
    query = cartesian_query(model.n_subjects, model.n_items, device=model.device)
    query.update(extra_keys)
    return model.predict(query).view(model.n_subjects, model.n_items)
