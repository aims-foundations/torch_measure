# Copyright (c) 2026 AIMS Foundation. MIT License.

"""MT-Bench GPT-4 judgment dataset definitions.

MT-Bench (LMSYS) evaluates LLMs on 80 multi-turn questions across 8 categories
(writing, roleplay, extraction, reasoning, math, coding, STEM, humanities),
scored 1-10 by GPT-4.  Each response matrix has:

- **Rows (subjects)**: LLMs evaluated on the benchmark.
- **Columns (items)**: 80 questions labeled ``q{id}_{category}``.
- **Values**: Continuous [0, 1] scores (original 1-10 divided by 10), or
  binary {0, 1} where score >= 0.5 maps to 1.

Source data: ``lmsys/mt-bench`` HuggingFace Space (GPT-4 single-answer grading).
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_MTBENCH_URL = "https://huggingface.co/spaces/lmsys/mt-bench"

_MTBENCH_CITATION = (
    "@article{zheng2023judging,\n"
    "  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},\n"
    "  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and "
    "Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and "
    "Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and "
    "Gonzalez, Joseph E. and Stoica, Ion},\n"
    "  journal={NeurIPS},\n"
    "  year={2023},\n"
    "  url={https://arxiv.org/abs/2306.05685}\n"
    "}"
)


def _register_mtbench_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for MT-Bench GPT-4 judgment response matrices."""
    datasets: dict[str, DatasetInfo] = {}

    # --- Both turns averaged (continuous) ---
    datasets["score"] = DatasetInfo(
        name="score",
        family="mtbench",
        description="MT-Bench GPT-4 scores, both turns averaged, normalized to [0,1]",
        response_type="continuous",
        n_subjects=34,
        n_items=80,
        subject_entity="LLM",
        item_entity="question",
        filename="score.pt",
        citation=_MTBENCH_CITATION,
        url=_MTBENCH_URL,
        license="CC-BY-4.0",
        tags=["nlp", "multi-turn", "llm-judge", "gpt4-graded"],
    )

    # --- Both turns averaged (binary) ---
    datasets["binary"] = DatasetInfo(
        name="binary",
        family="mtbench",
        description="MT-Bench GPT-4 scores, both turns, binary (score >= 0.5)",
        response_type="binary",
        n_subjects=34,
        n_items=80,
        subject_entity="LLM",
        item_entity="question",
        filename="binary.pt",
        citation=_MTBENCH_CITATION,
        url=_MTBENCH_URL,
        license="CC-BY-4.0",
        tags=["nlp", "multi-turn", "llm-judge", "gpt4-graded"],
    )

    # --- Turn 1 only (continuous) ---
    datasets["score_t1"] = DatasetInfo(
        name="score_t1",
        family="mtbench",
        description="MT-Bench GPT-4 scores, turn 1 only, normalized to [0,1]",
        response_type="continuous",
        n_subjects=34,
        n_items=80,
        subject_entity="LLM",
        item_entity="question",
        filename="score_t1.pt",
        citation=_MTBENCH_CITATION,
        url=_MTBENCH_URL,
        license="CC-BY-4.0",
        tags=["nlp", "single-turn", "llm-judge", "gpt4-graded"],
    )

    # --- Turn 1 only (binary) ---
    datasets["binary_t1"] = DatasetInfo(
        name="binary_t1",
        family="mtbench",
        description="MT-Bench GPT-4 scores, turn 1, binary (score >= 0.5)",
        response_type="binary",
        n_subjects=34,
        n_items=80,
        subject_entity="LLM",
        item_entity="question",
        filename="binary_t1.pt",
        citation=_MTBENCH_CITATION,
        url=_MTBENCH_URL,
        license="CC-BY-4.0",
        tags=["nlp", "single-turn", "llm-judge", "gpt4-graded"],
    )

    # --- Turn 2 only (continuous) ---
    datasets["score_t2"] = DatasetInfo(
        name="score_t2",
        family="mtbench",
        description="MT-Bench GPT-4 scores, turn 2 only, normalized to [0,1]",
        response_type="continuous",
        n_subjects=34,
        n_items=80,
        subject_entity="LLM",
        item_entity="question",
        filename="score_t2.pt",
        citation=_MTBENCH_CITATION,
        url=_MTBENCH_URL,
        license="CC-BY-4.0",
        tags=["nlp", "multi-turn", "llm-judge", "gpt4-graded"],
    )

    # --- Turn 2 only (binary) ---
    datasets["binary_t2"] = DatasetInfo(
        name="binary_t2",
        family="mtbench",
        description="MT-Bench GPT-4 scores, turn 2, binary (score >= 0.5)",
        response_type="binary",
        n_subjects=34,
        n_items=80,
        subject_entity="LLM",
        item_entity="question",
        filename="binary_t2.pt",
        citation=_MTBENCH_CITATION,
        url=_MTBENCH_URL,
        license="CC-BY-4.0",
        tags=["nlp", "multi-turn", "llm-judge", "gpt4-graded"],
    )

    return datasets
