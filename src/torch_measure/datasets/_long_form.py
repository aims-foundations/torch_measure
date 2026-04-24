# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Long-form dataset container — the native unit of measurement-db."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from torch_measure.data.response_matrix import ResponseMatrix


@dataclass
class LongFormData:
    """One dataset's long-form observations + registries, as returned by ``load()``.

    Attributes
    ----------
    name : str
        The benchmark id, e.g. ``"mtbench"``.
    responses : pandas.DataFrame
        Long-form response rows with columns
        ``subject_id, item_id, benchmark_id, trial, test_condition,
        response, correct_answer, trace``. Primary key
        ``(subject_id, item_id, trial, test_condition)``.
    items : pandas.DataFrame
        Item registry filtered to this benchmark — columns
        ``item_id, benchmark_id, raw_item_id, content, correct_answer,
        content_hash``.
    subjects : pandas.DataFrame
        Subject registry filtered to subjects that appear in ``responses`` —
        columns ``subject_id, display_name, provider, hub_repo, revision,
        params, release_date, raw_labels_seen, notes``.
    traces : pandas.DataFrame | None
        Optional long-form traces (same PK as responses) — only present for
        benchmarks that publish a ``{name}_traces.parquet`` on HF.
    info : dict
        The benchmarks-registry row for this dataset as a dict (``name``,
        ``license``, ``source_url``, ``description``, ``modality``,
        ``domain``, ``response_type``, ``response_scale``, ``categorical``,
        ``paper_url``, ``release_date``, ...).
    """

    name: str
    responses: "pd.DataFrame"
    items: "pd.DataFrame"
    subjects: "pd.DataFrame"
    traces: "pd.DataFrame | None"
    info: dict

    def to_response_matrix(self) -> "ResponseMatrix":
        """Opt-in pivot to the legacy wide-form :class:`ResponseMatrix`."""
        from torch_measure.data.response_matrix import ResponseMatrix

        return ResponseMatrix.from_long(self)

    def to_fit_tensors(self, device: str = "cpu"):
        """Extract long-form fit inputs for IRT/factor models.

        Returns integer index tensors plus the ordered ID lists that define
        the mapping: ``subject_idx[k] == i`` means row k's subject is
        ``subject_ids[i]``. Models are expected to be constructed with
        ``n_subjects == len(subject_ids)`` and ``n_items == len(item_ids)``.

        When multiple trials or test conditions exist for a (subject, item)
        cell, each observation becomes its own row — the caller doesn't
        lose information here (unlike ``to_response_matrix()``, which
        averages).

        Parameters
        ----------
        device : str
            Target device for the returned tensors.

        Returns
        -------
        dict
            ``{"subject_idx": LongTensor (n_obs,), "item_idx": LongTensor
            (n_obs,), "response": FloatTensor (n_obs,), "subject_ids":
            list[str], "item_ids": list[str]}``.
        """
        import torch

        responses = self.responses
        if responses.empty:
            raise ValueError(f"LongFormData({self.name!r}) has no responses")

        subject_ids = sorted(responses["subject_id"].unique().tolist())
        item_ids = sorted(responses["item_id"].unique().tolist())
        subj_to_idx = {s: i for i, s in enumerate(subject_ids)}
        item_to_idx = {it: i for i, it in enumerate(item_ids)}

        subject_idx = torch.tensor(
            [subj_to_idx[s] for s in responses["subject_id"]],
            dtype=torch.long,
            device=device,
        )
        item_idx = torch.tensor(
            [item_to_idx[it] for it in responses["item_id"]],
            dtype=torch.long,
            device=device,
        )
        response = torch.tensor(
            responses["response"].to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=device,
        )
        return {
            "subject_idx": subject_idx,
            "item_idx": item_idx,
            "response": response,
            "subject_ids": subject_ids,
            "item_ids": item_ids,
        }
