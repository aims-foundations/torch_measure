# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Base class for all IRT models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_measure.models._predictor import Predictor

if TYPE_CHECKING:
    from torch_measure.datasets._long_form import LongFormData


class IRTModel(Predictor):
    """Abstract base for factor-based Item Response Theory models.

    Specialises :class:`Predictor` for models with explicit ``ability`` and
    ``difficulty`` parameters that compose into a per-cell probability via
    a logistic link. Subclasses implement :meth:`predict` (inherited from
    :class:`Predictor`) by gathering parameters at the query indices and
    applying the IRT formula — see :meth:`_irt_probability`.

    For non-factor predictors (TabPFN-style, neural baselines), inherit
    :class:`Predictor` directly instead.
    """

    def fit(
        self,
        data: LongFormData | torch.Tensor,
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
            Fitting method: ``"mle"``, ``"em"``, ``"jml"``, or ``"svi"``.
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

            return mle_fit(
                self, subject_idx, item_idx, response, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs
            )
        elif method == "em":
            from torch_measure.fitting.em import em_fit

            return em_fit(
                self, subject_idx, item_idx, response, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs
            )
        elif method == "jml":
            from torch_measure.fitting.jml import jml_fit

            return jml_fit(
                self, subject_idx, item_idx, response, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs
            )
        elif method == "svi":
            from torch_measure.fitting.svi import svi_fit

            return svi_fit(
                self, subject_idx, item_idx, response, max_epochs=max_epochs, lr=lr, verbose=verbose, **kwargs
            )
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
            raise TypeError(f"fit() expected LongFormData or torch.Tensor, got {type(data).__name__}")

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
        """Element-wise IRT probability ``P(correct | params)``.

        All inputs must be 1-D tensors of equal length ``N`` (already
        gathered at the query indices). Implements::

            P = c + (1 - c) * sigmoid(a * (theta - b))

        where ``theta=ability``, ``b=difficulty``, ``a=discrimination``,
        ``c=guessing``.

        Parameters
        ----------
        ability : torch.Tensor
            Subject abilities at query rows, shape ``(N,)``.
        difficulty : torch.Tensor
            Item difficulties at query rows, shape ``(N,)``.
        discrimination : torch.Tensor | None
            Item discriminations at query rows, shape ``(N,)``. Defaults to 1.
        guessing : torch.Tensor | None
            Item guessing parameters at query rows, shape ``(N,)``. Defaults to 0.

        Returns
        -------
        torch.Tensor
            Probabilities, shape ``(N,)``.
        """
        logit = ability - difficulty
        if discrimination is not None:
            logit = discrimination * logit
        prob = torch.sigmoid(logit)
        if guessing is not None:
            prob = guessing + (1 - guessing) * prob
        return prob
