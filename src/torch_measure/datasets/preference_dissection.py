# Copyright (c) 2026 AIMS Foundation. MIT License.

"""GAIR Preference Dissection dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PD_URL = "https://huggingface.co/datasets/GAIR/preference-dissection"

_PD_CITATION = (
    "@inproceedings{li2024dissecting,\n"
    "  title={Dissecting Human and LLM Preferences},\n"
    "  author={Li, Junlong and Fan, Shichao and Shao, Yilong and "
    "Peng, Marco and Yang, Gao and Liu, Zekun and Liu, Pengfei},\n"
    "  booktitle={Proceedings of the 62nd Annual Meeting of the "
    "Association for Computational Linguistics (ACL)},\n"
    "  year={2024},\n"
    "  url={https://arxiv.org/abs/2402.11296}\n"
    "}"
)


def _register_preference_dissection_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for GAIR Preference Dissection response matrices.

    GAIR Preference Dissection evaluates pairwise preferences of 33 judges
    (32 LLM judges + 1 human judge) on 5,240 Chatbot Arena conversation
    pairs.  Each response matrix has:

    - **all_judges**: (33 judges x 5,240 pairs) — judges as subjects,
      pairs as items.  Binary 0/1 encoding: 0 = prefers response_1,
      1 = prefers response_2.
    - **crossed**: (5,240 pairs x 33 judges) — transposed for
      G-theory facet analysis.

    Source data: ``GAIR/preference-dissection`` on HuggingFace (CC-BY-NC-4.0).
    """
    datasets: dict[str, DatasetInfo] = {}

    datasets["all_judges"] = DatasetInfo(
        name="all_judges",
        family="preference_dissection",
        description=(
            "GAIR Preference Dissection — 33 judges x 5,240 Chatbot Arena "
            "pairs, binary pairwise preferences"
        ),
        response_type="binary",
        n_subjects=33,
        n_items=5240,
        subject_entity="judge",
        item_entity="pair",
        filename="all_judges.pt",
        citation=_PD_CITATION,
        url=_PD_URL,
        license="CC-BY-NC-4.0",
        tags=["preference", "pairwise", "multi-judge", "chatbot-arena"],
    )

    datasets["crossed"] = DatasetInfo(
        name="crossed",
        family="preference_dissection",
        description=(
            "GAIR Preference Dissection — 5,240 pairs x 33 judges, "
            "transposed for G-theory facet analysis"
        ),
        response_type="binary",
        n_subjects=5240,
        n_items=33,
        subject_entity="pair",
        item_entity="judge",
        filename="crossed.pt",
        citation=_PD_CITATION,
        url=_PD_URL,
        license="CC-BY-NC-4.0",
        tags=["preference", "pairwise", "multi-judge", "g-theory", "crossed"],
    )

    return datasets
