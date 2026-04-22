# Copyright (c) 2026 AIMS Foundations. MIT License.

"""DatasetInfo metadata class for the dataset registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a dataset in the registry.

    Parameters
    ----------
    name : str
        Canonical name (e.g., ``"swebench"``). Used as the key in :func:`load`.
    family : str
        Dataset family (e.g., ``"helm"``).
    description : str
        One-line description of the dataset.
    response_type : str
        One of ``"binary"`` or ``"continuous"``.
    n_subjects : int
        Number of subjects (rows) in the response matrix.
    n_items : int
        Number of items (columns) in the response matrix.
    subject_entity : str
        What subjects represent (e.g., ``"LLM"``, ``"human"``).
    item_entity : str
        What items represent (e.g., ``"question"``, ``"task"``).
    repo_id : str
        HuggingFace Hub repository ID.
    filename : str
        Filename within the HF repo (e.g., ``"swebench.pt"``).
    citation : str
        BibTeX citation string for the source data.
    url : str
        URL to the original benchmark or paper.
    license : str
        License of the source data.
    tags : list[str]
        Searchable tags (e.g., ``["nlp", "multiple-choice"]``).
    """

    name: str
    family: str
    description: str
    response_type: str
    n_subjects: int
    n_items: int
    subject_entity: str = "LLM"
    item_entity: str = "question"
    repo_id: str = "aims-foundations/measurement-db"
    filename: str = ""
    citation: str = ""
    url: str = ""
    license: str = ""
    n_comparisons: int = 0
    tags: list[str] = field(default_factory=list)
