# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Nectar (Berkeley NEST) dataset definitions.

Nectar contains 182,954 prompts, each with 7 ranked responses from diverse
models (GPT-4, GPT-3.5-turbo, Llama-2-7B-chat, Mistral-7B-instruct, etc.),
ranked 1-7 by GPT-4.

The response matrices use normalized rank as score:
  rank 1 (best)  -> 1.0
  rank 7 (worst) -> 0.0
  formula: (7 - rank) / 6

Each matrix has:
- **Rows (subjects)**: LLM model names (40 unique models).
- **Columns (items)**: Individual prompts.
- **Values**: Normalized rank scores in [0, 1], NaN for missing entries
  (not every model responds to every prompt; ~83.8% missing overall).

Source data: ``berkeley-nest/Nectar`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_NECTAR_URL = "https://huggingface.co/datasets/berkeley-nest/Nectar"

_NECTAR_CITATION = (
    "@article{zhu2023starling,\n"
    "  title={Starling-7B: Improving Helpfulness and Harmlessness with RLAIF},\n"
    "  author={Zhu, Banghua and Frick, Evan and Wu, Tianhao and Zhu, Hanlin\n"
    "          and Ganesan, Karthik and Chiang, Wei-Lin and Zhang, Jian\n"
    "          and Jiao, Jiantao},\n"
    "  year={2023},\n"
    "  url={https://starling.cs.berkeley.edu/}\n"
    "}"
)


def _register_nectar_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Nectar response matrices.

    Nectar (Berkeley NEST lab) provides ranked LLM responses to diverse
    prompts.  Each response matrix has:

    - **Rows**: LLM models (40 unique models across all prompts).
    - **Columns**: Prompts from Nectar.
    - **Values**: Normalized rank [0, 1] where 1.0 = rank 1 (best),
      0.0 = rank 7 (worst).  NaN for models that did not respond to
      a given prompt (~83.8% missing).

    Two versions are provided:
    - ``nectar/all``: All 182,954 prompts.
    - ``nectar/50k``: Random 50,000-prompt subset (seed=42).

    Source data: ``berkeley-nest/Nectar`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- All prompts ---
    datasets["all"] = DatasetInfo(
        name="all",
        family="nectar",
        description=("Nectar — all prompts, normalized rank scores (40 models x 182,954 prompts, 83.8% missing)"),
        response_type="continuous",
        n_subjects=40,
        n_items=182954,
        subject_entity="LLM",
        item_entity="prompt",
        filename="all.pt",
        citation=_NECTAR_CITATION,
        url=_NECTAR_URL,
        license="CC-BY-NC-4.0",
        tags=["ranking", "preference", "multi-model", "gpt4-judge"],
    )

    # --- 50K subset ---
    datasets["50k"] = DatasetInfo(
        name="50k",
        family="nectar",
        description=(
            "Nectar — 50K random prompt subset, normalized rank scores (40 models x 50,000 prompts, 83.8% missing)"
        ),
        response_type="continuous",
        n_subjects=40,
        n_items=50000,
        subject_entity="LLM",
        item_entity="prompt",
        filename="50k.pt",
        citation=_NECTAR_CITATION,
        url=_NECTAR_URL,
        license="CC-BY-NC-4.0",
        tags=["ranking", "preference", "multi-model", "gpt4-judge", "subset"],
    )

    return datasets
