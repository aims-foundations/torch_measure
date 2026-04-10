# Copyright (c) 2026 AIMS Foundation. MIT License.

"""OpenAssistant OASST1 dataset definitions.

This module registers response matrices from the OpenAssistant OASST1 dataset,
which contains ~161K messages in conversation trees with ~461K quality ratings
from human volunteers. Multiple assistant responses per prompt are ranked by
human annotators.

Each response matrix follows the standard torch_measure convention:

- **Rows (subjects)**: Rank tiers (rank_0 = best, rank_1 = second best, ...).
- **Columns (items)**: Prompts (identified by parent message_id).
- **Values**: Normalized human-assigned rank scores in [0, 1], where
  1.0 = best (rank 0) and 0.0 = worst.  NaN for missing entries
  (not every rank tier is present for every prompt).

Data files live on HuggingFace Hub at ``aims-foundation/torch-measure-data``
under the ``oasst/`` prefix.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_OASST_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1"

_OASST_CITATION = (
    "@inproceedings{kopf2023openassistant,\n"
    "  title={OpenAssistant Conversations -- Democratizing Large Language Model Alignment},\n"
    "  author={K{\\\"o}pf, Andreas and Kilcher, Yannic and von R{\\\"u}tte, Dimitri and "
    "Anagnostidis, Sotiris and Tam, Zhi-Rui and Stevens, Keith and "
    "Barhoum, Abdullah and Nguyen, Duc and Stanley, Oliver and "
    "Nagyfi, Rich{\\'a}rd and others},\n"
    "  booktitle={Advances in Neural Information Processing Systems},\n"
    "  year={2023}\n"
    "}"
)


def _register_oasst_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for OASST1 response matrices.

    OpenAssistant OASST1 provides human-ranked assistant responses to
    conversation prompts.  Each response matrix has:

    - **Rows**: Rank tiers (rank_0 = best response, rank_1 = second best, ...).
    - **Columns**: Prompts with multiple ranked assistant alternatives.
    - **Values**: Normalized rank score [0, 1] where 1.0 = rank 0 (best),
      0.0 = worst rank.  NaN for rank tiers not present for a prompt.

    Two versions are provided:
    - ``oasst/ranked``: All prompts with >= 2 ranked alternatives.
    - ``oasst/ranked_rich``: Prompts with >= 3 ranked alternatives (richer signal).

    Source data: ``OpenAssistant/oasst1`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- All prompts with >= 2 ranked alternatives ---
    datasets["ranked"] = DatasetInfo(
        name="ranked",
        family="oasst",
        description=(
            "OASST1 -- human-ranked assistant responses, normalized rank scores, "
            "prompts with >= 2 alternatives (16 rank tiers x 18,922 prompts)"
        ),
        response_type="continuous",
        n_subjects=16,
        n_items=18922,
        subject_entity="rank_tier",
        item_entity="prompt",
        filename="ranked.pt",
        citation=_OASST_CITATION,
        url=_OASST_URL,
        license="Apache-2.0",
        tags=["ranking", "preference", "human-rated", "conversation", "multilingual"],
    )

    # --- Rich subset: prompts with >= 3 ranked alternatives ---
    datasets["ranked_rich"] = DatasetInfo(
        name="ranked_rich",
        family="oasst",
        description=(
            "OASST1 -- human-ranked assistant responses, normalized rank scores, "
            "prompts with >= 3 alternatives (16 rank tiers x 12,058 prompts)"
        ),
        response_type="continuous",
        n_subjects=16,
        n_items=12058,
        subject_entity="rank_tier",
        item_entity="prompt",
        filename="ranked_rich.pt",
        citation=_OASST_CITATION,
        url=_OASST_URL,
        license="Apache-2.0",
        tags=["ranking", "preference", "human-rated", "conversation", "multilingual", "subset"],
    )

    return datasets
