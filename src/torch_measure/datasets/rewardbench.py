# Copyright (c) 2026 AIMS Foundations. MIT License.

"""RewardBench dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_REWARDBENCH_URL = "https://huggingface.co/datasets/allenai/reward-bench"

_REWARDBENCH_CITATION = (
    "@inproceedings{lambert2024rewardbench,\n"
    "  title={RewardBench: Evaluating Reward Models for Language Modeling},\n"
    "  author={Lambert, Nathan and Pyatkin, Valentina and Morrison, Jacob and\n"
    "          Miranda, LJ and Lin, Bill Yuchen and Chandu, Khyathi and\n"
    "          Dziri, Nouha and Kumar, Sachin and Zick, Tom and Choi, Yejin and\n"
    "          Smith, Noah A. and Hajishirzi, Hannaneh},\n"
    "  booktitle={arXiv preprint arXiv:2403.13787},\n"
    "  year={2024},\n"
    "  url={https://huggingface.co/datasets/allenai/reward-bench}\n"
    "}"
)


def _register_rewardbench_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for RewardBench response matrices.

    RewardBench evaluates reward models on 2,985 (prompt, chosen, rejected)
    trios.  Each reward model produces a binary correct/incorrect decision
    per item (whether it ranked chosen > rejected).

    - **Rows**: Reward models (classifiers, generative, DPO, etc.).
    - **Columns**: Individual (prompt, chosen, rejected) items.
    - **Values**: Binary 0/1 (incorrect/correct preference ranking).

    Items span 23 subsets grouped into 4 categories:
    chat, chat_hard, safety, reasoning.

    Source data: ``allenai/reward-bench-results`` on HuggingFace
    (``eval-set-scores/`` directory).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- All items ---

    datasets["results"] = DatasetInfo(
        name="results",
        family="rewardbench",
        description=("RewardBench full eval set, binary correct/incorrect per reward model"),
        response_type="binary",
        n_subjects=151,
        n_items=2985,
        subject_entity="reward_model",
        item_entity="preference_pair",
        filename="results.pt",
        citation=_REWARDBENCH_CITATION,
        url=_REWARDBENCH_URL,
        license="Apache-2.0",
        tags=["reward-model", "preference", "binary"],
    )

    # --- Per-category splits ---

    datasets["chat"] = DatasetInfo(
        name="chat",
        family="rewardbench",
        description="RewardBench chat subset (alpacaeval, mt-bench easy/med)",
        response_type="binary",
        n_subjects=151,
        n_items=358,
        subject_entity="reward_model",
        item_entity="preference_pair",
        filename="chat.pt",
        citation=_REWARDBENCH_CITATION,
        url=_REWARDBENCH_URL,
        license="Apache-2.0",
        tags=["reward-model", "preference", "binary", "chat"],
    )

    datasets["chat_hard"] = DatasetInfo(
        name="chat_hard",
        family="rewardbench",
        description="RewardBench chat-hard subset (mt-bench hard, LLMBar adversarial)",
        response_type="binary",
        n_subjects=151,
        n_items=456,
        subject_entity="reward_model",
        item_entity="preference_pair",
        filename="chat_hard.pt",
        citation=_REWARDBENCH_CITATION,
        url=_REWARDBENCH_URL,
        license="Apache-2.0",
        tags=["reward-model", "preference", "binary", "chat-hard"],
    )

    datasets["safety"] = DatasetInfo(
        name="safety",
        family="rewardbench",
        description="RewardBench safety subset (refusals, xstest, donotanswer)",
        response_type="binary",
        n_subjects=151,
        n_items=740,
        subject_entity="reward_model",
        item_entity="preference_pair",
        filename="safety.pt",
        citation=_REWARDBENCH_CITATION,
        url=_REWARDBENCH_URL,
        license="Apache-2.0",
        tags=["reward-model", "preference", "binary", "safety"],
    )

    datasets["reasoning"] = DatasetInfo(
        name="reasoning",
        family="rewardbench",
        description="RewardBench reasoning subset (math-prm, HumanEval code)",
        response_type="binary",
        n_subjects=151,
        n_items=1431,
        subject_entity="reward_model",
        item_entity="preference_pair",
        filename="reasoning.pt",
        citation=_REWARDBENCH_CITATION,
        url=_REWARDBENCH_URL,
        license="Apache-2.0",
        tags=["reward-model", "preference", "binary", "reasoning"],
    )

    return datasets
