# Copyright (c) 2026 AIMS Foundation. MIT License.

"""FLASK (Fine-grained Language Model Evaluation based on Alignment Skill Sets) dataset definitions.

FLASK evaluates LLMs on fine-grained skill-based criteria using GPT-4 as a judge.
Each response matrix has:

- **Rows (subjects)**: LLMs (15 models including GPT-4, ChatGPT, Claude, LLaMA-2, etc.).
- **Columns (items)**: Instructions drawn from multiple sources (1,700 items).
- **Values**: Continuous scores on a 1-5 scale (NaN where a skill is not applicable
  to an instruction or was marked N/A by the judge).

FLASK defines 4 primary skill categories decomposed into 12 fine-grained skills:

- **Logical Thinking**: Logical Correctness, Logical Robustness, Logical Efficiency
- **Background Knowledge**: Factuality, Commonsense Understanding
- **Problem Handling**: Comprehension, Insightfulness, Completeness, Metacognition
- **User Alignment**: Conciseness, Readability, Harmlessness

Each instruction is evaluated on 2-3 relevant skills (not all 12), so per-skill
response matrices are sparse (NaN where a skill does not apply to an item).

Source data: ``kaistAI/FLASK`` on GitHub (GPT-4 review outputs).
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_FLASK_URL = "https://github.com/kaistAI/FLASK"

_FLASK_CITATION = (
    "@inproceedings{ye2024flask,\n"
    "  title={FLASK: Fine-grained Language Model Evaluation based on "
    "Alignment Skill Sets},\n"
    "  author={Ye, Seonghyeon and Kim, Doyoung and Kim, Sungdong and "
    "Hwang, Hyeonbin and Kim, Seungone and Jo, Yongrae and Thorne, James "
    "and Kim, Juho and Seo, Minjoon},\n"
    "  booktitle={Proceedings of ICLR},\n"
    "  year={2024},\n"
    "  url={https://arxiv.org/abs/2307.10928}\n"
    "}"
)

# 12 fine-grained skills with their item counts (out of 1,700 evaluated items).
_SKILLS: dict[str, int] = {
    "logical_correctness": 467,
    "logical_robustness": 212,
    "logical_efficiency": 174,
    "factuality": 612,
    "commonsense_understanding": 705,
    "comprehension": 1148,
    "insightfulness": 297,
    "completeness": 454,
    "metacognition": 131,
    "conciseness": 325,
    "readability": 448,
    "harmlessness": 127,
}

# Number of evaluated models and items (instructions) in the review data.
_N_MODELS = 15
_N_ITEMS = 1700


def _register_flask_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for FLASK response matrices.

    Registers an overall dataset (mean score across all applicable skills per
    instruction) plus one dataset per fine-grained skill (12 skills, each a
    sparse models x instructions matrix).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Overall (mean across applicable skills) ---

    datasets["overall"] = DatasetInfo(
        name="overall",
        family="flask",
        description=(
            f"FLASK overall mean score across applicable skills "
            f"({_N_MODELS} models x {_N_ITEMS} instructions)"
        ),
        response_type="continuous",
        n_subjects=_N_MODELS,
        n_items=_N_ITEMS,
        subject_entity="model",
        item_entity="instruction",
        filename="overall.pt",
        citation=_FLASK_CITATION,
        url=_FLASK_URL,
        license="Apache-2.0",
        tags=["fine-grained", "skill-based", "llm-as-judge", "gpt4-eval"],
    )

    # --- Per-skill splits ---

    for skill, n_items in _SKILLS.items():
        name = f"{skill}"
        datasets[name] = DatasetInfo(
            name=name,
            family="flask",
            description=(
                f"FLASK {skill.replace('_', ' ')} scores "
                f"({_N_MODELS} models x {n_items} instructions)"
            ),
            response_type="continuous",
            n_subjects=_N_MODELS,
            n_items=n_items,
            subject_entity="model",
            item_entity="instruction",
            filename=f"{skill}.pt",
            citation=_FLASK_CITATION,
            url=_FLASK_URL,
            license="Apache-2.0",
            tags=["fine-grained", "skill-based", "llm-as-judge", "gpt4-eval", skill],
        )

    return datasets
