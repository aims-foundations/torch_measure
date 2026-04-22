# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Indeterminacy Experiments dataset definitions.

Multi-judge LLM evaluation data from "Validating LLM-as-a-Judge under
Rating Indeterminacy" (NeurIPS 2025).  9 LLM judges evaluate 200 items
across 4 task groups using forced-choice and multi-label rating schemes.

Source: ``lguerdan/indeterminacy-experiments`` on HuggingFace.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_INDETERMINACY_URL = "https://huggingface.co/datasets/lguerdan/indeterminacy-experiments"

_INDETERMINACY_CITATION = (
    "@inproceedings{guerdan2025indeterminacy,\n"
    "  title={Validating LLM-as-a-Judge under Rating Indeterminacy},\n"
    "  author={Guerdan, Luke and others},\n"
    "  booktitle={NeurIPS},\n"
    "  year={2025},\n"
    "  url={https://huggingface.co/datasets/lguerdan/indeterminacy-experiments}\n"
    "}"
)


def _register_indeterminacy_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Indeterminacy Experiments response matrices.

    Each response matrix has:

    - **Rows**: 9 LLM judges (claude-3-5-sonnet, claude-3-haiku, deepseek-chat,
      Llama-3.3-70B, mistral-large, mistral-small, gpt-3.5-turbo,
      gpt-4o-mini, o3-mini).
    - **Columns**: 200 evaluation items (or 800 for the combined matrix).
    - **Values**: P(category 0) averaged across repetitions, continuous [0, 1].
      NaN for items without responses in a given task group.

    There are 4 task groups (different rating configurations applied to the
    same 200 items) and 7 meaningful (group, scale) rating tasks total.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Combined (all groups) ---

    datasets["all"] = DatasetInfo(
        name="all",
        family="indeterminacy",
        description=("All task groups combined, P(cat 0) across reps (9 judges x 800 items)"),
        response_type="continuous",
        n_subjects=9,
        n_items=800,
        subject_entity="LLM",
        item_entity="item",
        filename="all.pt",
        citation=_INDETERMINACY_CITATION,
        url=_INDETERMINACY_URL,
        license="MIT",
        tags=["llm-as-judge", "multi-judge", "indeterminacy", "rating"],
    )

    # --- Per task group ---

    _group_descriptions = {
        0: "Task group 0: 2-category forced-choice (9 judges x 200 items)",
        1: "Task group 1: 2-category forced-choice (9 judges x 200 items)",
        2: "Task group 2: 2-category forced-choice (9 judges x 200 items)",
        3: "Task group 3: 2/3-category mixed rating (9 judges x 200 items)",
    }

    for g in range(4):
        name = f"group_{g}"
        datasets[name] = DatasetInfo(
            name=name,
            family="indeterminacy",
            description=_group_descriptions[g],
            response_type="continuous",
            n_subjects=9,
            n_items=200,
            subject_entity="LLM",
            item_entity="item",
            filename=f"group_{g}.pt",
            citation=_INDETERMINACY_CITATION,
            url=_INDETERMINACY_URL,
            license="MIT",
            tags=["llm-as-judge", "multi-judge", "indeterminacy", "rating"],
        )

    return datasets
