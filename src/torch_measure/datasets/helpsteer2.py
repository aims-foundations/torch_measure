# Copyright (c) 2026 AIMS Foundation. MIT License.

"""HelpSteer2 dataset definitions.

This module registers response matrices from the NVIDIA HelpSteer2 dataset,
which contains ~10K prompts each with 2 anonymous responses, rated by human
annotators on 5 attributes (helpfulness, correctness, coherence, complexity,
verbosity) on a 0-4 scale.

Each response matrix follows the standard torch_measure convention:

- **Rows (subjects)**: 2 anonymous responses per prompt (response_0, response_1).
- **Columns (items)**: Individual prompts (~10K items).
- **Values**: Human ratings normalized to [0, 1] (from original 0-4 scale).

Data files live on HuggingFace Hub at ``aims-foundation/torch-measure-data``
under the ``helpsteer2/`` prefix.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_HS2_URL = "https://huggingface.co/datasets/nvidia/HelpSteer2"

_HS2_CITATION = (
    "@article{wang2024helpsteer2,\n"
    "  title={HelpSteer2: Open-source dataset for training top-performing reward models},\n"
    "  author={Wang, Zhilin and Dong, Yi and Delalleau, Olivier and "
    "Zeng, Jiaqi and Shen, Gerald and Zhang, Daniel and Kuchaiev, Oleksii and "
    "Zeng, Jimmy},\n"
    "  journal={arXiv preprint arXiv:2406.08673},\n"
    "  year={2024}\n"
    "}"
)

# Dimensions will be filled after running migration (placeholders updated below).
_N_SUBJECTS = 2
_N_ITEMS = 10679


def _register_helpsteer2_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for HelpSteer2 response matrices.

    NVIDIA HelpSteer2 provides human multi-attribute preference ratings for
    paired responses to ~10K prompts.  Each prompt has exactly 2 anonymous
    responses rated on 5 attributes (helpfulness, correctness, coherence,
    complexity, verbosity) on a 0-4 scale, normalized to [0, 1].
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Overall (mean across all 5 attributes) ---
    datasets["overall"] = DatasetInfo(
        name="overall",
        family="helpsteer2",
        description=(
            "HelpSteer2 -- mean human rating across all attributes, "
            f"normalized to [0,1] ({_N_SUBJECTS} responses x {_N_ITEMS} prompts)"
        ),
        response_type="continuous",
        n_subjects=_N_SUBJECTS,
        n_items=_N_ITEMS,
        subject_entity="response",
        item_entity="prompt",
        filename="overall.pt",
        citation=_HS2_CITATION,
        url=_HS2_URL,
        license="CC-BY-4.0",
        tags=["nlp", "preference", "human-rated", "multi-attribute"],
    )

    # --- Helpfulness ---
    datasets["helpfulness"] = DatasetInfo(
        name="helpfulness",
        family="helpsteer2",
        description=(
            "HelpSteer2 -- human helpfulness rating, "
            f"normalized to [0,1] ({_N_SUBJECTS} responses x {_N_ITEMS} prompts)"
        ),
        response_type="continuous",
        n_subjects=_N_SUBJECTS,
        n_items=_N_ITEMS,
        subject_entity="response",
        item_entity="prompt",
        filename="helpfulness.pt",
        citation=_HS2_CITATION,
        url=_HS2_URL,
        license="CC-BY-4.0",
        tags=["nlp", "helpfulness", "human-rated"],
    )

    # --- Correctness ---
    datasets["correctness"] = DatasetInfo(
        name="correctness",
        family="helpsteer2",
        description=(
            "HelpSteer2 -- human correctness rating, "
            f"normalized to [0,1] ({_N_SUBJECTS} responses x {_N_ITEMS} prompts)"
        ),
        response_type="continuous",
        n_subjects=_N_SUBJECTS,
        n_items=_N_ITEMS,
        subject_entity="response",
        item_entity="prompt",
        filename="correctness.pt",
        citation=_HS2_CITATION,
        url=_HS2_URL,
        license="CC-BY-4.0",
        tags=["nlp", "correctness", "human-rated"],
    )

    # --- Coherence ---
    datasets["coherence"] = DatasetInfo(
        name="coherence",
        family="helpsteer2",
        description=(
            "HelpSteer2 -- human coherence rating, "
            f"normalized to [0,1] ({_N_SUBJECTS} responses x {_N_ITEMS} prompts)"
        ),
        response_type="continuous",
        n_subjects=_N_SUBJECTS,
        n_items=_N_ITEMS,
        subject_entity="response",
        item_entity="prompt",
        filename="coherence.pt",
        citation=_HS2_CITATION,
        url=_HS2_URL,
        license="CC-BY-4.0",
        tags=["nlp", "coherence", "human-rated"],
    )

    # --- Complexity ---
    datasets["complexity"] = DatasetInfo(
        name="complexity",
        family="helpsteer2",
        description=(
            "HelpSteer2 -- human complexity rating, "
            f"normalized to [0,1] ({_N_SUBJECTS} responses x {_N_ITEMS} prompts)"
        ),
        response_type="continuous",
        n_subjects=_N_SUBJECTS,
        n_items=_N_ITEMS,
        subject_entity="response",
        item_entity="prompt",
        filename="complexity.pt",
        citation=_HS2_CITATION,
        url=_HS2_URL,
        license="CC-BY-4.0",
        tags=["nlp", "complexity", "human-rated"],
    )

    # --- Verbosity ---
    datasets["verbosity"] = DatasetInfo(
        name="verbosity",
        family="helpsteer2",
        description=(
            "HelpSteer2 -- human verbosity rating, "
            f"normalized to [0,1] ({_N_SUBJECTS} responses x {_N_ITEMS} prompts)"
        ),
        response_type="continuous",
        n_subjects=_N_SUBJECTS,
        n_items=_N_ITEMS,
        subject_entity="response",
        item_entity="prompt",
        filename="verbosity.pt",
        citation=_HS2_CITATION,
        url=_HS2_URL,
        license="CC-BY-4.0",
        tags=["nlp", "verbosity", "human-rated"],
    )

    return datasets
