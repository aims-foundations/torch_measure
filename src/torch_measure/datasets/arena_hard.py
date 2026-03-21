# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Arena-Hard-Auto dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_ARENA_HARD_URL = "https://github.com/lm-sys/arena-hard-auto"

_ARENA_HARD_CITATION = (
    "@article{li2024crowdsourced,\n"
    "  title={From Crowdsourced Data to High-Quality Benchmarks: "
    "Arena Hard and BenchBuilder Pipeline},\n"
    "  author={Li, Tianle and Chiang, Wei-Lin and Frick, Evan and "
    "Dunlap, Lisa and Wu, Tianhao and Zhu, Banghua and Gonzalez, Joseph E "
    "and Stoica, Ion},\n"
    "  journal={arXiv preprint arXiv:2406.11939},\n"
    "  year={2024}\n"
    "}"
)


def _register_arena_hard_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Arena-Hard-Auto response matrices.

    Arena-Hard-Auto (LMSYS) evaluates LLMs by having a judge model
    (GPT-4-Turbo) compare each model's response against a fixed baseline
    (GPT-4-0314) on 500 challenging user prompts.

    Each judgment is a 5-level outcome (A>>B, A>B, A=B, B>A, B>>A) mapped
    to numeric scores from the model's perspective:
        A>>B=1.0, A>B=0.75, A=B=0.5, B>A=0.25, B>>A=0.0

    Two games per prompt (with swapped ordering) are averaged to reduce
    position bias.

    - **Rows**: LLMs being evaluated.
    - **Columns**: 500 challenging prompts (question IDs).
    - **Values**: Continuous [0, 1] judgment scores averaged across
      two games (swapped A/B order).

    Source data: ``lmarena-ai/arena-hard-auto`` on HuggingFace.
    """
    datasets: dict[str, DatasetInfo] = {}

    datasets["arena_hard/judgments"] = DatasetInfo(
        name="arena_hard/judgments",
        family="arena_hard",
        description=(
            "Arena-Hard-Auto v0.1 judgments: GPT-4-Turbo judge scores "
            "for 72 models vs GPT-4-0314 baseline on 500 prompts"
        ),
        response_type="continuous",
        n_subjects=72,
        n_items=500,
        subject_entity="LLM",
        item_entity="prompt",
        filename="arena_hard/judgments.pt",
        citation=_ARENA_HARD_CITATION,
        url=_ARENA_HARD_URL,
        license="Apache-2.0",
        tags=["llm-judge", "pairwise-comparison", "arena-hard"],
    )

    return datasets
