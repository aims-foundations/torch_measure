# Copyright (c) 2026 AIMS Foundations. MIT License.

"""DatasetInfo metadata class for the dataset registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a dataset in the registry.

    Mirrors a row of ``benchmarks.parquet`` on the measurement-db HF bucket
    plus a small number of legacy ``torch_measure``-specific convenience
    fields. All fields default to an empty value so manifest rows with
    missing columns still yield a valid :class:`DatasetInfo`.

    Parameters
    ----------
    name : str
        Canonical benchmark id (e.g., ``"mtbench"``). Used as the key in
        :func:`torch_measure.datasets.load`.
    description : str
        One-line description of the dataset.
    response_type : str
        How the grader emits the response. Controlled vocabulary in
        measurement-db: ``"binary"``, ``"likert_5"``, ``"likert_10"``,
        ``"win_rate"``, ``"ordinal"``, ``"fraction"``,
        ``"continuous_bounded"``, ``"continuous_unbounded"``,
        ``"error_presence"``, ``"mixed"``.
    response_scale : str
        Free-form description of the response value set, e.g. ``"{0, 1}"``,
        ``"{1, 2, 3, 4, 5}"``, ``"[-18, 18] continuous"``.
    categorical : bool
        ``True`` if the response set is finitely enumerable in a small way
        (binary, likert, ordinal, win-rate, error-presence). ``False`` for
        truly continuous responses or variable-denominator fractions.
    modality : list[str]
        Input modalities required to solve items (``"text"``, ``"image"``,
        ``"grid"``, ``"gui_screenshot"``, ``"audio"``).
    domain : list[str]
        Subject areas (e.g., ``"software_engineering"``, ``"mathematics"``,
        ``"preference"``).
    release_date : str
        Benchmark release date as ``YYYY-MM``.
    paper_url : str
        URL of the benchmark's originating paper.
    license : str
        SPDX license identifier (or free-form if unknown).
    source_url : str
        URL to the benchmark's data repo / upstream source.
    version : str
        Version string, if the benchmark has one.
    repo_id : str
        HuggingFace Hub repository id that hosts the long-form parquet.
    filename : str
        Filename of the responses parquet within the HF repo.
    citation : str
        BibTeX citation string.
    tags : list[str]
        Searchable tags — union of ``modality`` and ``domain`` by default.
    family : str
        Legacy grouping key. Kept for backwards compatibility with the
        previous per-family registry layout; new code should filter on
        ``domain`` or ``tags`` instead.
    n_subjects : int
        Number of subjects in the long-form responses (0 if not populated
        yet; the loader fills this in after downloading).
    n_items : int
        Number of items in the long-form responses (0 if not populated yet).
    subject_entity : str
        What subjects represent (e.g., ``"LLM"``, ``"agent"``, ``"judge"``).
    item_entity : str
        What items represent (e.g., ``"question"``, ``"task"``).
    url : str
        Backwards-compatible alias for ``source_url``; populated to the
        same value.
    n_comparisons : int
        Number of pairwise comparisons, for pairwise benchmarks. ``0`` for
        point-response benchmarks.
    """

    name: str
    description: str = ""
    response_type: str = ""
    response_scale: str = ""
    categorical: bool = True
    modality: list[str] = field(default_factory=list)
    domain: list[str] = field(default_factory=list)
    release_date: str = ""
    paper_url: str = ""
    license: str = ""
    source_url: str = ""
    version: str = ""
    repo_id: str = "aims-foundations/measurement-db"
    filename: str = ""
    citation: str = ""
    tags: list[str] = field(default_factory=list)
    family: str = "misc"
    n_subjects: int = 0
    n_items: int = 0
    subject_entity: str = "LLM"
    item_entity: str = "question"
    url: str = ""
    n_comparisons: int = 0
