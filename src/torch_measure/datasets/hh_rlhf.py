# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Anthropic HH-RLHF dataset definitions.

Anthropic HH-RLHF contains ~161K human preference pairs for RLHF training.
Each sample is a (chosen, rejected) conversation pair where a human annotator
selected the preferred response.  The data is split into "helpful" and
"harmless" subsets:

- **helpful**: Combinations of helpful-base, helpful-online, and
  helpful-rejection-sampled configs.
- **harmless**: The harmless-base config.

Each pairwise comparison payload has:

- **subject_ids**: ``["chosen", "rejected"]`` (2 subjects).
- **Comparisons**: One per preference pair; chosen always wins (outcome=1.0).
- **item_contents**: The chosen conversation text for each pair.
- **comparison_metadata**: Contains the rejected conversation text.

Source data: ``Anthropic/hh-rlhf`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_HH_RLHF_URL = "https://huggingface.co/datasets/Anthropic/hh-rlhf"

_HH_RLHF_CITATION = (
    "@article{bai2022training,\n"
    "  title={Training a Helpful and Harmless Assistant with Reinforcement "
    "Learning from Human Feedback},\n"
    "  author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda "
    "and Chen, Anna and DaSilva, Nova and Drain, Dawn and Fort, Stanislav and "
    "Ganguli, Deep and Henighan, Tom and others},\n"
    "  journal={arXiv preprint arXiv:2204.05862},\n"
    "  year={2022}\n"
    "}"
)


def _register_hh_rlhf_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Anthropic HH-RLHF pairwise preference data.

    Anthropic HH-RLHF provides human preference pairs where annotators chose
    between two assistant responses.  Each dataset has:

    - **Subjects**: ``["chosen", "rejected"]`` (2).
    - **Comparisons**: One per preference pair (chosen always wins).
    - **Items**: Each preference pair is a unique item with conversation text.

    Source data: ``Anthropic/hh-rlhf`` on HuggingFace Hub (MIT license).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Helpful subset ---
    datasets["helpful"] = DatasetInfo(
        name="helpful",
        family="hh_rlhf",
        description=(
            "Anthropic HH-RLHF helpful subset — human preference pairs "
            "from helpful-base, helpful-online, helpful-rejection-sampled"
        ),
        response_type="pairwise",
        n_subjects=2,
        n_items=124503,
        n_comparisons=124503,
        subject_entity="response",
        item_entity="conversation_pair",
        filename="helpful.pt",
        citation=_HH_RLHF_CITATION,
        url=_HH_RLHF_URL,
        license="MIT",
        tags=["rlhf", "preference", "pairwise", "helpful", "human-evaluation"],
    )

    # --- Harmless subset ---
    datasets["harmless"] = DatasetInfo(
        name="harmless",
        family="hh_rlhf",
        description=("Anthropic HH-RLHF harmless subset — human preference pairs from harmless-base"),
        response_type="pairwise",
        n_subjects=2,
        n_items=44849,
        n_comparisons=44849,
        subject_entity="response",
        item_entity="conversation_pair",
        filename="harmless.pt",
        citation=_HH_RLHF_CITATION,
        url=_HH_RLHF_URL,
        license="MIT",
        tags=["rlhf", "preference", "pairwise", "harmless", "human-evaluation"],
    )

    return datasets
