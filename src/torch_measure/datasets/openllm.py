# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Open LLM Leaderboard dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_OPENLLM_URL = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
_N_SUBJECTS = 4232  # Number of LLMs evaluated on the Open LLM Leaderboard

_OPENLLM_CITATION = (
    "@misc{open-llm-leaderboard-v2,\n"
    "  author={Fourrier, Clémentine and Habib, Nathan and Wolf, Thomas and Tunstall, Lewis},\n"
    "  title={Open LLM Leaderboard v2},\n"
    "  year={2024},\n"
    "  publisher={Hugging Face},\n"
    "  howpublished={\\url{https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard}}\n"
    "}"
)


def _register_openllm_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Open LLM Leaderboard response matrices.

    The Open LLM Leaderboard (v2) by Hugging Face evaluates open-source LLMs
    on multiple benchmarks.  The source data (``stair-lab/zero_shot_open_llm_leaderboard``)
    contains item-level responses for **bbh** and **mmlu_pro**.

    - **Rows**: 4,232 LLMs submitted to the leaderboard (subjects).
    - **Columns**: Individual questions from each benchmark (items).
    - **Values**: Binary {0, 1} for correct / incorrect (NaN for missing).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Reasoning ---

    datasets["bbh"] = DatasetInfo(
        name="bbh",
        family="openllm",
        description="BIG-Bench Hard (23 challenging BIG-Bench tasks requiring multi-step reasoning)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=5758,
        filename="bbh.pt",
        citation=(
            "@article{suzgun2022challenging,\n"
            "  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},\n"
            "  author={Suzgun, Mirac and Scales, Nathan and Sch{\\\"a}rli, Nathanael and Gehrmann, Sebastian and "
            "Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and "
            "Zhou, Denny and Wei, Jason},\n"
            "  journal={arXiv preprint arXiv:2210.09261},\n"
            "  year={2022}\n"
            "}"
        ),
        url=_OPENLLM_URL,
        license="Apache-2.0",
        tags=["nlp", "reasoning", "multi-step"],
    )

    # --- Knowledge ---

    datasets["mmlu_pro"] = DatasetInfo(
        name="mmlu_pro",
        family="openllm",
        description="MMLU-Pro (harder MMLU variant with 10-choice questions across 14 disciplines)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=11864,
        filename="mmlu_pro.pt",
        citation=(
            "@article{wang2024mmlu,\n"
            "  title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},\n"
            "  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and "
            "Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and "
            "Li, Tianle and Ku, Max and Wang, Kai and Zeng, Alex and Yu, Chunwei and others},\n"
            "  journal={arXiv preprint arXiv:2406.01574},\n"
            "  year={2024}\n"
            "}"
        ),
        url=_OPENLLM_URL,
        license="MIT",
        tags=["nlp", "knowledge", "multiple-choice"],
    )

    # --- Aggregate ---

    datasets["all"] = DatasetInfo(
        name="all",
        family="openllm",
        description="All Open LLM Leaderboard benchmarks concatenated (4,232 models x 17,622 items across 2 benchmarks)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=17622,
        filename="all.pt",
        citation=_OPENLLM_CITATION,
        url=_OPENLLM_URL,
        license="Apache-2.0",
        tags=["nlp", "aggregate", "multi-benchmark"],
    )

    return datasets
