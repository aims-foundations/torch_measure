# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Pluggable LLM-judge protocol for second-pass review of flagged items.

The diagnostics ensemble surfaces items that are statistically anomalous,
but does not interpret the content of those items. Callers can attach any
callable matching :class:`ItemJudge` to :func:`flag_items`; flagged items
will be passed to the callable and its short justification stored in the
returned DataFrame.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ItemJudge"]


@runtime_checkable
class ItemJudge(Protocol):
    """Callable contract for LLM-based review of a flagged benchmark item.

    Implementations should be side-effect free and return a short string
    explaining why the item may be invalid. No specific provider, prompt,
    or output schema is assumed.
    """

    def __call__(
        self,
        item_text: str,
        item_idx: int,
        anomaly_score: float,
    ) -> str:
        """Review one flagged item.

        Parameters
        ----------
        item_text : str
            The text of the item being reviewed.
        item_idx : int
            Position of the item in the original response matrix.
        anomaly_score : float
            Ensemble anomaly score, on the standard-normal scale produced
            by the Gaussian rank transform.

        Returns
        -------
        str
            Short free-text justification.
        """
        ...
