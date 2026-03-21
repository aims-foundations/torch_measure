# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Chatbot Arena pairwise comparison dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_ARENA_URL = "https://chat.lmsys.org/"

_ARENA_CITATION = (
    "@article{chiang2024chatbot,\n"
    "  title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},\n"
    "  author={Chiang, Wei-Lin and Zheng, Lianmin and Sheng, Ying and Angelopoulos, Anastasios Nikolas "
    "and Li, Tianle and Li, Dacheng and Zhang, Hao and Zhu, Banghua and Jordan, Michael and "
    "Gonzalez, Joseph E and Stoica, Ion},\n"
    "  journal={Proceedings of ICML},\n"
    "  year={2024}\n"
    "}"
)


def _register_arena_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Chatbot Arena pairwise comparison data.

    Chatbot Arena (LMSYS) collects human preference judgments between pairs
    of LLMs in blind A/B tests.  Each comparison records which model's
    response a human judge preferred, or whether it was a tie.

    Source data: ``stair-lab/chatbot_arena`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    datasets["arena/chatbot_arena"] = DatasetInfo(
        name="arena/chatbot_arena",
        family="arena",
        description="LMSYS Chatbot Arena human preference comparisons (pairwise A/B tests)",
        response_type="pairwise",
        n_subjects=20,
        n_items=0,
        n_comparisons=23294,
        subject_entity="LLM",
        item_entity="prompt",
        filename="arena/chatbot_arena.pt",
        citation=_ARENA_CITATION,
        url=_ARENA_URL,
        license="CC-BY-4.0",
        tags=["nlp", "pairwise", "preference", "human-evaluation", "chatbot"],
    )

    datasets["arena/chatbot_arena_140k"] = DatasetInfo(
        name="arena/chatbot_arena_140k",
        family="arena",
        description="LMSYS Chatbot Arena 140K human preference comparisons (pairwise A/B tests, 53 models)",
        response_type="pairwise",
        n_subjects=53,
        n_items=0,
        n_comparisons=135634,
        subject_entity="LLM",
        item_entity="prompt",
        filename="arena/chatbot_arena_140k.pt",
        citation=_ARENA_CITATION,
        url=_ARENA_URL,
        license="CC-BY-4.0",
        tags=["nlp", "pairwise", "preference", "human-evaluation", "chatbot"],
    )

    return datasets
