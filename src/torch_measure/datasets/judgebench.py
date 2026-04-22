# Copyright (c) 2026 AIMS Foundations. MIT License.

"""JudgeBench dataset definitions.

JudgeBench evaluates LLM-as-judge capabilities on challenging response pairs
with objectively verifiable correctness labels.  The benchmark covers four
categories: knowledge, reasoning, math, and coding, drawn from MMLU-Pro,
LiveBench, and LiveCodeBench.

Each response matrix has:
- **Rows (subjects)**: Judge models (prompted judges, fine-tuned judges,
  multi-agent judges, and reward models).
- **Columns (items)**: Individual response pairs to be judged.
- **Values**: Binary {0, 1} indicating whether the judge correctly identified
  the objectively better response.

Source data:
    - ``ScalerLab/JudgeBench`` on HuggingFace (350 GPT pairs + 270 Claude pairs).
    - ``ScalerLab/JudgeBench`` on GitHub (per-judge evaluation outputs).

Reference:
    Tan et al., "JudgeBench: A Benchmark for Evaluating LLM-based Judges",
    ICLR 2025.  https://arxiv.org/abs/2410.12784
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_JB_URL = "https://huggingface.co/datasets/ScalerLab/JudgeBench"

_JB_CITATION = (
    "@inproceedings{judgebench2025,\n"
    "  title={JudgeBench: A Benchmark for Evaluating LLM-based Judges},\n"
    "  author={Tan, Sijun and Zhuang, Siyuan and Montgomery, Kyle and "
    "Tang, William Y. and Cuadron, Alejandro and Wang, Chenguang and "
    "Popa, Raluca Ada and Stoica, Ion},\n"
    "  booktitle={International Conference on Learning Representations (ICLR)},\n"
    "  year={2025},\n"
    "  url={https://arxiv.org/abs/2410.12784}\n"
    "}"
)

# ---------------------------------------------------------------------------
# Category mapping: source -> category
# ---------------------------------------------------------------------------

# knowledge: 13 MMLU-Pro subjects (excluding math, which goes in the math category)
_KNOWLEDGE_SOURCES = [
    "mmlu-pro-biology",
    "mmlu-pro-business",
    "mmlu-pro-chemistry",
    "mmlu-pro-computer science",
    "mmlu-pro-economics",
    "mmlu-pro-engineering",
    "mmlu-pro-health",
    "mmlu-pro-history",
    "mmlu-pro-law",
    "mmlu-pro-other",
    "mmlu-pro-philosophy",
    "mmlu-pro-physics",
    "mmlu-pro-psychology",
]

# math: LiveBench math + MMLU-Pro math
_MATH_SOURCES = ["livebench-math", "mmlu-pro-math"]

# reasoning: LiveBench reasoning
_REASONING_SOURCES = ["livebench-reasoning"]

# coding: LiveCodeBench
_CODING_SOURCES = ["livecodebench"]

SOURCE_TO_CATEGORY = {}
for _s in _KNOWLEDGE_SOURCES:
    SOURCE_TO_CATEGORY[_s] = "knowledge"
for _s in _MATH_SOURCES:
    SOURCE_TO_CATEGORY[_s] = "math"
for _s in _REASONING_SOURCES:
    SOURCE_TO_CATEGORY[_s] = "reasoning"
for _s in _CODING_SOURCES:
    SOURCE_TO_CATEGORY[_s] = "coding"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _register_judgebench_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for JudgeBench response matrices.

    JudgeBench evaluates LLM-as-judge capabilities on challenging response
    pairs with objectively verifiable correctness labels.  Each matrix is
    judges x pairs with binary correctness values.

    We build matrices from the GPT split (350 pairs, 33 judges) which has
    the most comprehensive judge coverage.

    Categories:
    - **knowledge**: 143 pairs from 13 MMLU-Pro subjects
    - **reasoning**: 98 pairs from LiveBench reasoning
    - **math**: 67 pairs from LiveBench math + MMLU-Pro math
    - **coding**: 42 pairs from LiveCodeBench
    """
    d: dict[str, DatasetInfo] = {}

    # --- All pairs (GPT split) ---

    d["all"] = DatasetInfo(
        name="all",
        family="judgebench",
        description=(
            "JudgeBench all categories — LLM-as-judge evaluation on "
            "challenging response pairs (33 judges x 350 pairs)"
        ),
        response_type="binary",
        n_subjects=33,
        n_items=350,
        subject_entity="judge",
        item_entity="pair",
        filename="all.pt",
        citation=_JB_CITATION,
        url=_JB_URL,
        license="MIT",
        tags=["llm-as-judge", "preference", "binary"],
    )

    # --- Per-category splits ---

    d["knowledge"] = DatasetInfo(
        name="knowledge",
        family="judgebench",
        description=(
            "JudgeBench knowledge category — MMLU-Pro subjects "
            "(33 judges x 143 pairs)"
        ),
        response_type="binary",
        n_subjects=33,
        n_items=143,
        subject_entity="judge",
        item_entity="pair",
        filename="knowledge.pt",
        citation=_JB_CITATION,
        url=_JB_URL,
        license="MIT",
        tags=["llm-as-judge", "preference", "knowledge"],
    )

    d["reasoning"] = DatasetInfo(
        name="reasoning",
        family="judgebench",
        description=(
            "JudgeBench reasoning category — LiveBench reasoning "
            "(33 judges x 98 pairs)"
        ),
        response_type="binary",
        n_subjects=33,
        n_items=98,
        subject_entity="judge",
        item_entity="pair",
        filename="reasoning.pt",
        citation=_JB_CITATION,
        url=_JB_URL,
        license="MIT",
        tags=["llm-as-judge", "preference", "reasoning"],
    )

    d["math"] = DatasetInfo(
        name="math",
        family="judgebench",
        description=(
            "JudgeBench math category — LiveBench math + MMLU-Pro math "
            "(33 judges x 67 pairs)"
        ),
        response_type="binary",
        n_subjects=33,
        n_items=67,
        subject_entity="judge",
        item_entity="pair",
        filename="math.pt",
        citation=_JB_CITATION,
        url=_JB_URL,
        license="MIT",
        tags=["llm-as-judge", "preference", "math"],
    )

    d["coding"] = DatasetInfo(
        name="coding",
        family="judgebench",
        description=(
            "JudgeBench coding category — LiveCodeBench "
            "(33 judges x 42 pairs)"
        ),
        response_type="binary",
        n_subjects=33,
        n_items=42,
        subject_entity="judge",
        item_entity="pair",
        filename="coding.pt",
        citation=_JB_CITATION,
        url=_JB_URL,
        license="MIT",
        tags=["llm-as-judge", "preference", "coding"],
    )

    return d
