# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Base class for all IRT models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Union

import torch
from torch import nn

if TYPE_CHECKING:
    from torch_measure.datasets._long_form import LongFormData


class IRTModel(nn.Module):
    """Abstract base class for Item Response Theory models.

    All IRT models share the interface:

    - ``.fit(data, ...)`` to estimate parameters from a
      :class:`~torch_measure.datasets.LongFormData` (preferred) or a
      wide-form response tensor.
    - ``.predict()`` to compute the full ``(n_subjects, n_items)`` probability
      matrix; ``.predict_at(s_idx, i_idx)`` for sparse evaluation.
    - ``.ability`` to access subject ability parameters.
    - ``.difficulty`` to access item difficulty parameters.
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

    def predict_at(self, subject_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Predict response probabilities at specified ``(subject, item)`` cells.

        Default implementation materialises the full ``(n_subjects, n_items)``
        probability matrix via :meth:`predict` and indexes into it. Subclasses
        should override this when the full matrix is expensive to compute
        (e.g., :class:`AmortizedIRT` with many items).

        Parameters
        ----------
        subject_idx : torch.LongTensor
            Integer subject indices, shape ``(n_obs,)``.
        item_idx : torch.LongTensor
            Integer item indices, shape ``(n_obs,)``.

        Returns
        -------
        torch.Tensor
            Probabilities at the requested cells, shape ``(n_obs,)``.
        """
        return self.predict()[subject_idx, item_idx]

    def forward(self) -> torch.Tensor:
        """Forward pass returns predicted probabilities."""
        return self.predict()

    def fit(
        self,
        data: "Union[LongFormData, torch.Tensor]",
        mask: torch.Tensor | None = None,
        method: str = "mle",
        max_epochs: int = 1000,
        lr: float = 0.01,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Fit the model.

        Parameters
        ----------
        data : LongFormData | torch.Tensor
            Either a :class:`~torch_measure.datasets.LongFormData` (canonical
            long-form input — every observation is one row) or a wide-form
            response tensor of shape ``(n_subjects, n_items)``. For wide-form,
            missing entries may be encoded as ``NaN`` or ``-1``.
        mask : torch.Tensor | None
            Only used when ``data`` is a wide-form tensor — boolean mask of
            entries to use for fitting. Inferred from NaN/-1 when ``None``.
            Ignored for long-form input (absent rows are absent observations).
        method : str
            Fitting method: ``"mle"``, ``"em"``, ``"jml"``, or ``"svi"``
            (requires pyro-ppl).
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
        subject_idx, item_idx, response = self._normalize_fit_inputs(data, mask)

        if method == "mle":
            from torch_measure.fitting.mle import mle_fit

            return mle_fit(self, subject_idx, item_idx, response,
                           max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        elif method == "em":
            from torch_measure.fitting.em import em_fit

            return em_fit(self, subject_idx, item_idx, response,
                          max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        elif method == "jml":
            from torch_measure.fitting.jml import jml_fit

            return jml_fit(self, subject_idx, item_idx, response,
                           max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        elif method == "svi":
            from torch_measure.fitting.svi import svi_fit

            return svi_fit(self, subject_idx, item_idx, response,
                           max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unknown fitting method: {method!r}. Use 'mle', 'em', 'jml', or 'svi'.")

    def _normalize_fit_inputs(
        self,
        data,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coerce ``data`` (LongFormData or wide-form tensor) to the long-form triple.

        Returns ``(subject_idx, item_idx, response)`` on ``self._device``.
        """
        from torch_measure.datasets._long_form import LongFormData

        if isinstance(data, LongFormData):
            fit_inputs = data.to_fit_tensors(device=str(self._device))
            return (
                fit_inputs["subject_idx"],
                fit_inputs["item_idx"],
                fit_inputs["response"],
            )

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"fit() expected LongFormData or torch.Tensor, got {type(data).__name__}"
            )

        response_matrix = data.to(self._device)
        if mask is None:
            mask = ~torch.isnan(response_matrix) & (response_matrix != -1)
        mask = mask.to(self._device)

        obs_indices = mask.nonzero(as_tuple=False)
        subject_idx = obs_indices[:, 0].to(self._device)
        item_idx = obs_indices[:, 1].to(self._device)
        response = response_matrix[mask].float().to(self._device)
        return subject_idx, item_idx, response

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
