# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Core ResponseMatrix data structure for measurement analysis."""

from __future__ import annotations

import torch


class ResponseMatrix:
    """A binary or continuous response matrix (subjects x items).

    Parameters
    ----------
    data : torch.Tensor
        Response matrix of shape (n_subjects, n_items). Values can be:
        - Binary (0/1) for correct/incorrect responses
        - Continuous [0, 1] for probability responses
        - NaN for missing data
    subject_ids : list[str] | None
        Optional identifiers for subjects (rows).
    item_ids : list[str] | None
        Optional identifiers for items (columns).
    """

    def __init__(
        self,
        data: torch.Tensor,
        subject_ids: list[str] | None = None,
        item_ids: list[str] | None = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {data.ndim}D")
        self.data = data.float()
        self.subject_ids = subject_ids
        self.item_ids = item_ids

    @property
    def n_rows(self) -> int:
        """Number of subjects (rows)."""
        return self.data.shape[0]

    @property
    def n_cols(self) -> int:
        """Number of items (columns)."""
        return self.data.shape[1]

    @property
    def n_subjects(self) -> int:
        """Number of subjects (rows)."""
        return self.data.shape[0]

    @property
    def n_items(self) -> int:
        """Number of items (columns)."""
        return self.data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the response matrix."""
        return (self.n_rows, self.n_cols)

    @property
    def observed_mask(self) -> torch.Tensor:
        """Boolean mask of observed (non-NaN) entries."""
        return ~torch.isnan(self.data)

    @property
    def density(self) -> float:
        """Fraction of observed (non-missing) entries."""
        return self.observed_mask.float().mean().item()

    @property
    def subject_means(self) -> torch.Tensor:
        """Mean response per subject (ignoring NaN)."""
        data = self.data.clone()
        data[~self.observed_mask] = 0.0
        counts = self.observed_mask.float().sum(dim=1)
        return data.sum(dim=1) / counts.clamp(min=1)

    @property
    def item_means(self) -> torch.Tensor:
        """Mean response per item (ignoring NaN), i.e., item easiness/facility."""
        data = self.data.clone()
        data[~self.observed_mask] = 0.0
        counts = self.observed_mask.float().sum(dim=0)
        return data.sum(dim=0) / counts.clamp(min=1)

    def to(self, device: torch.device | str) -> ResponseMatrix:
        """Move response matrix to a device."""
        return ResponseMatrix(
            data=self.data.to(device),
            subject_ids=self.subject_ids,
            item_ids=self.item_ids,
        )

    def binarize(self, threshold: float = 0.5) -> ResponseMatrix:
        """Convert continuous responses to binary using a threshold."""
        binary = (self.data >= threshold).float()
        binary[~self.observed_mask] = float("nan")
        return ResponseMatrix(binary, self.subject_ids, self.item_ids)

    @classmethod
    def from_numpy(cls, array, **kwargs) -> ResponseMatrix:
        """Create from a numpy array."""
        return cls(torch.from_numpy(array).float(), **kwargs)

    @classmethod
    def from_dataframe(cls, df) -> ResponseMatrix:
        """Create from a pandas DataFrame."""
        return cls(
            torch.tensor(df.values, dtype=torch.float32),
            subject_ids=list(df.index.astype(str)),
            item_ids=list(df.columns.astype(str)),
        )

    def __repr__(self) -> str:
        return (
            f"ResponseMatrix(n_subjects={self.n_subjects}, n_items={self.n_items}, "
            f"density={self.density:.2%})"
        )
