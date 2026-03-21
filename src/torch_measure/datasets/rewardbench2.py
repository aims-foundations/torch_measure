# Copyright (c) 2026 AIMS Foundation. MIT License.

"""RewardBench 2 dataset definitions.

RewardBench 2 (Allen AI) evaluates reward models on harder preference triples
across 6 domains: Factuality, Focus, Math, Precise IF, Safety, Ties.

Each response matrix has:
- **Rows (subjects)**: Reward models / judges (sequence classifiers,
  generative RMs, LLM-as-judge, etc.).
- **Columns (items)**: Individual (prompt, chosen, rejected) evaluation triples.
- **Values**: Binary {0, 1} indicating whether the judge correctly ranked
  the chosen response above the rejected response.

Source data:
    - ``allenai/reward-bench-2`` on HuggingFace (1,865 evaluation items).
    - ``allenai/reward-bench-2-results`` on HuggingFace (per-judge results).
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_RB2_URL = "https://huggingface.co/datasets/allenai/reward-bench-2"

_RB2_CITATION = (
    "@article{rewardbench2,\n"
    "  title={RewardBench 2: A Harder Evaluation for Reward Models},\n"
    "  author={Lambert, Nathan and Pyatkin, Valentina and Hayase, Jonathan "
    "and Liu, Bill Yuchen and Soldaini, Luca},\n"
    "  year={2025},\n"
    "  eprint={2506.01937},\n"
    "  archivePrefix={arXiv},\n"
    "  url={https://arxiv.org/abs/2506.01937}\n"
    "}"
)


def _register_rewardbench2_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for RewardBench 2 response matrices.

    RewardBench 2 evaluates reward models on harder preference triples.
    Each matrix is judges x items with binary correctness values.
    """
    d: dict[str, DatasetInfo] = {}

    d["rewardbench2/all"] = DatasetInfo(
        name="rewardbench2/all",
        family="rewardbench2",
        description=(
            "RewardBench 2 all domains — reward model evaluation on harder "
            "preference triples (188 judges x 1865 items)"
        ),
        response_type="binary",
        n_subjects=188,
        n_items=1865,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/all.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "binary"],
    )

    d["rewardbench2/factuality"] = DatasetInfo(
        name="rewardbench2/factuality",
        family="rewardbench2",
        description="RewardBench 2 Factuality domain (188 judges x 445 items)",
        response_type="binary",
        n_subjects=188,
        n_items=445,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/factuality.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "factuality"],
    )

    d["rewardbench2/focus"] = DatasetInfo(
        name="rewardbench2/focus",
        family="rewardbench2",
        description="RewardBench 2 Focus domain (188 judges x 505 items)",
        response_type="binary",
        n_subjects=188,
        n_items=505,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/focus.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "focus"],
    )

    d["rewardbench2/math"] = DatasetInfo(
        name="rewardbench2/math",
        family="rewardbench2",
        description="RewardBench 2 Math domain (188 judges x 193 items)",
        response_type="binary",
        n_subjects=188,
        n_items=193,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/math.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "math"],
    )

    d["rewardbench2/precise_if"] = DatasetInfo(
        name="rewardbench2/precise_if",
        family="rewardbench2",
        description="RewardBench 2 Precise Instruction Following domain (188 judges x 160 items)",
        response_type="binary",
        n_subjects=188,
        n_items=160,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/precise_if.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "instruction-following"],
    )

    d["rewardbench2/safety"] = DatasetInfo(
        name="rewardbench2/safety",
        family="rewardbench2",
        description="RewardBench 2 Safety domain (188 judges x 460 items)",
        response_type="binary",
        n_subjects=188,
        n_items=460,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/safety.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "safety"],
    )

    d["rewardbench2/ties"] = DatasetInfo(
        name="rewardbench2/ties",
        family="rewardbench2",
        description="RewardBench 2 Ties domain (188 judges x 102 items)",
        response_type="binary",
        n_subjects=188,
        n_items=102,
        subject_entity="reward_model",
        item_entity="preference_triple",
        filename="rewardbench2/ties.pt",
        citation=_RB2_CITATION,
        url=_RB2_URL,
        license="ODC-BY",
        tags=["reward-model", "preference", "ties"],
    )

    return d
