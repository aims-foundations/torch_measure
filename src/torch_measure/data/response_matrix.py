# Copyright (c) 2026 AIMS Foundations. MIT License.

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
    item_contents : list[str] | None
        Optional text content for each item (e.g., question text).
    subject_metadata : list[dict[str, str | int | float | bool | None]] | None
        Optional structured metadata for each subject (one dict per row).
        For HELM datasets, each dict has keys: ``org``, ``model``,
        ``param_count``, ``is_instruct``.
    info : dict | None
        Optional dataset-level metadata (interpretation notes, paper URL,
        data source URL, license, etc.). Usually loaded from
        ``data/<benchmark>/info.yaml``. Common keys include:
        ``description``, ``testing_condition``, ``paper_url``,
        ``data_source_url``, ``subject_type``, ``item_type``, ``license``,
        ``citation``, ``tags``.
    """

    def __init__(
        self,
        data: torch.Tensor,
        subject_ids: list[str] | None = None,
        item_ids: list[str] | None = None,
        item_contents: list[str] | None = None,
        subject_metadata: list[dict] | None = None,
        info: dict | None = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {data.ndim}D")
        self.data = data.float()
        self.subject_ids = subject_ids
        self.item_ids = item_ids
        self.item_contents = item_contents
        self.subject_metadata = subject_metadata
        self.info = info

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
            item_contents=self.item_contents,
            subject_metadata=self.subject_metadata,
            info=self.info,
        )

    def binarize(self, threshold: float = 0.5) -> ResponseMatrix:
        """Convert continuous responses to binary using a threshold."""
        binary = (self.data >= threshold).float()
        binary[~self.observed_mask] = float("nan")
        return ResponseMatrix(
            binary, self.subject_ids, self.item_ids,
            self.item_contents, self.subject_metadata, self.info,
        )

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

    @classmethod
    def from_long(cls, data) -> ResponseMatrix:
        """Pivot a :class:`LongFormData` into a wide :class:`ResponseMatrix`.

        When multiple trials or non-null ``test_condition`` values exist per
        (subject, item) cell, the response is averaged across those
        dimensions. The legacy ``load()`` path used to do this automatically;
        consumers who want polytomous / per-trial / multi-condition analysis
        should work with the :class:`LongFormData` directly.

        Parameters
        ----------
        data : LongFormData
            The long-form dataset returned by
            :func:`torch_measure.datasets.load`.

        Returns
        -------
        ResponseMatrix
            Subject-by-item matrix with subjects rendered as their
            ``display_name`` (when the subjects registry is populated) and
            items keyed by ``item_id``. ``item_contents`` carries the item
            ``content`` strings from the items registry.
        """
        import pandas as pd  # noqa: F401 — required for pivot

        responses = data.responses
        items = data.items
        subjects = data.subjects
        name = data.name

        if "benchmark_id" in items.columns:
            items_bench = items[items["benchmark_id"] == name]
        else:
            items_bench = items
        items_bench = items_bench.set_index("item_id")
        present_items = set(responses["item_id"].unique())
        items_bench = items_bench[items_bench.index.isin(present_items)]

        needs_agg = (
            responses["trial"].nunique() > 1
            or (
                "test_condition" in responses.columns
                and responses["test_condition"].notna().any()
            )
        )
        if needs_agg:
            agg = (
                responses
                .groupby(["subject_id", "item_id"], as_index=False)["response"]
                .mean()
            )
        else:
            agg = responses[["subject_id", "item_id", "response"]]

        matrix = agg.pivot(index="subject_id", columns="item_id", values="response")

        ordered_item_ids = [iid for iid in items_bench.index if iid in matrix.columns]
        matrix = matrix.reindex(columns=ordered_item_ids)

        subjects_by_id = (
            subjects.set_index("subject_id")
            if "subject_id" in subjects.columns
            else subjects
        )
        subject_ids = list(matrix.index)
        display_names = [
            str(subjects_by_id.at[sid, "display_name"])
            if (
                hasattr(subjects_by_id, "index")
                and sid in subjects_by_id.index
                and "display_name" in getattr(subjects_by_id, "columns", [])
            )
            else sid
            for sid in subject_ids
        ]
        item_contents = [
            (
                items_bench.at[iid, "content"]
                if (iid in items_bench.index and "content" in items_bench.columns)
                else None
            )
            or ""
            for iid in ordered_item_ids
        ]
        item_contents = [str(c) for c in item_contents]

        tensor = torch.tensor(matrix.values, dtype=torch.float32)

        info = dict(data.info) if data.info is not None else {}
        info.setdefault("benchmark_id", name)

        return cls(
            data=tensor,
            subject_ids=display_names,
            item_ids=ordered_item_ids,
            item_contents=item_contents,
            subject_metadata=None,
            info=info,
        )

    def __repr__(self) -> str:
        return (
            f"ResponseMatrix(n_subjects={self.n_subjects}, n_items={self.n_items}, "
            f"density={self.density:.2%})"
        )
