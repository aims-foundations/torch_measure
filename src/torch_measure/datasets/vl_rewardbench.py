# Copyright (c) 2026 AIMS Foundation. MIT License.

"""VL-RewardBench dataset definitions.

VL-RewardBench evaluates vision-language generative reward models (VL-GenRMs)
on preference judgments across three domains: General, Hallucination, and
Reasoning.

Each response matrix has:
- **Rows (subjects)**: VLM judges evaluated as reward models.
- **Columns (items)**: Image-text preference pairs with human rankings.
- **Values**: Binary {0, 1} indicating whether the judge correctly identified
  the human-preferred response.

Source data:
    - ``MMInstruction/VL-RewardBench`` on HuggingFace (1,247 evaluation pairs).
    - Paper: arXiv:2411.17451 (CVPR 2025).
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_VLR_URL = "https://huggingface.co/datasets/MMInstruction/VL-RewardBench"

_VLR_CITATION = (
    "@article{VLRewardBench,\n"
    "  title={VL-RewardBench: A Challenging Benchmark for Vision-Language "
    "Generative Reward Models},\n"
    "  author={Li, Lei and Wei, Yuancheng and Xie, Zhihui and Yang, Xuqing "
    "and Song, Yifan and Wang, Peiyi and An, Chenxin and Liu, Tianyu "
    "and Li, Sujian and Lin, Bill Yuchen and Kong, Lingpeng and Liu, Qi},\n"
    "  year={2024},\n"
    "  eprint={2411.17451},\n"
    "  archivePrefix={arXiv},\n"
    "  url={https://arxiv.org/abs/2411.17451}\n"
    "}"
)


def _register_vl_rewardbench_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for VL-RewardBench response matrices.

    VL-RewardBench evaluates vision-language reward models on preference
    judgments.  Each matrix is judges x pairs with binary correctness values.

    Categories (derived from item ID prefixes):
    - **General**: WildVision, VLFeedback items.
    - **Hallucination**: RLAIF-V, RLHF-V, POVID, LRVInstruction items.
    - **Reasoning**: MathVerse, MMMU-Pro items.
    """
    d: dict[str, DatasetInfo] = {}

    d["all"] = DatasetInfo(
        name="all",
        family="vl_rewardbench",
        description=(
            "VL-RewardBench all categories — vision-language reward model "
            "evaluation on preference pairs (16 judges x 1247 pairs)"
        ),
        response_type="binary",
        n_subjects=16,
        n_items=1247,
        subject_entity="judge",
        item_entity="pair",
        filename="all.pt",
        citation=_VLR_CITATION,
        url=_VLR_URL,
        license="MIT",
        tags=["reward-model", "preference", "vision-language", "multimodal", "binary"],
    )

    d["general"] = DatasetInfo(
        name="general",
        family="vl_rewardbench",
        description=(
            "VL-RewardBench General category — general multimodal instruction "
            "pairs (16 judges x 183 pairs)"
        ),
        response_type="binary",
        n_subjects=16,
        n_items=183,
        subject_entity="judge",
        item_entity="pair",
        filename="general.pt",
        citation=_VLR_CITATION,
        url=_VLR_URL,
        license="MIT",
        tags=["reward-model", "preference", "vision-language", "general"],
    )

    d["hallucination"] = DatasetInfo(
        name="hallucination",
        family="vl_rewardbench",
        description=(
            "VL-RewardBench Hallucination category — visual hallucination "
            "detection pairs (16 judges x 746 pairs)"
        ),
        response_type="binary",
        n_subjects=16,
        n_items=746,
        subject_entity="judge",
        item_entity="pair",
        filename="hallucination.pt",
        citation=_VLR_CITATION,
        url=_VLR_URL,
        license="MIT",
        tags=["reward-model", "preference", "vision-language", "hallucination"],
    )

    d["reasoning"] = DatasetInfo(
        name="reasoning",
        family="vl_rewardbench",
        description=(
            "VL-RewardBench Reasoning category — multimodal reasoning "
            "pairs (16 judges x 318 pairs)"
        ),
        response_type="binary",
        n_subjects=16,
        n_items=318,
        subject_entity="judge",
        item_entity="pair",
        filename="reasoning.pt",
        citation=_VLR_CITATION,
        url=_VLR_URL,
        license="MIT",
        tags=["reward-model", "preference", "vision-language", "reasoning"],
    )

    return d
