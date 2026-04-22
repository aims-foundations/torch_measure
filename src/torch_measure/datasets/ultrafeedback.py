# Copyright (c) 2026 AIMS Foundations. MIT License.

"""UltraFeedback dataset definitions.

This module registers response matrices from the OpenBMB UltraFeedback dataset,
which contains 64K prompts each with 4 responses from different LLMs, rated by
GPT-4 on four aspects: helpfulness, honesty, instruction-following, and
truthfulness.

Each response matrix follows the standard torch_measure convention:

- **Rows (subjects)**: 17 LLMs that generated responses.
- **Columns (items)**: Individual prompts (63,967 items).
- **Values**: GPT-4 ratings normalized to [0, 1] (from original 1-5 scale).
  NaN for missing entries (each prompt has responses from only 4 of 17 models).

Data files live on HuggingFace Hub at ``aims-foundations/measurement-db``
under the ``ultrafeedback/`` prefix.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_UF_URL = "https://huggingface.co/datasets/openbmb/UltraFeedback"

_UF_CITATION = (
    "@article{cui2023ultrafeedback,\n"
    "  title={UltraFeedback: Boosting Language Models with High-quality Feedback},\n"
    "  author={Cui, Ganqu and Yuan, Lifan and Ding, Ning and Yao, Guanming and "
    "Zhu, Wei and Ni, Yuan and Xie, Guotong and Liu, Zhiyuan and Sun, Maosong},\n"
    "  journal={arXiv preprint arXiv:2310.01377},\n"
    "  year={2023}\n"
    "}"
)


def _register_ultrafeedback_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for UltraFeedback response matrices.

    UltraFeedback evaluates 17 LLMs on 64K diverse prompts.  GPT-4 rates each
    response on four aspects (helpfulness, honesty, instruction-following,
    truthfulness) on a 1-5 scale.  Scores are normalized to [0, 1].

    Each prompt has responses from exactly 4 of the 17 models, resulting in
    ~76.5% missing entries in the response matrix.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Overall (mean across all 4 aspects) ---
    datasets["overall"] = DatasetInfo(
        name="overall",
        family="ultrafeedback",
        description=(
            "UltraFeedback — mean GPT-4 rating across all aspects, normalized to [0,1] (17 models x 63,967 prompts)"
        ),
        response_type="continuous",
        n_subjects=17,
        n_items=63967,
        subject_entity="LLM",
        item_entity="prompt",
        filename="overall.pt",
        citation=_UF_CITATION,
        url=_UF_URL,
        license="MIT",
        tags=["nlp", "instruction-following", "gpt4-rated", "multi-aspect"],
    )

    # --- Helpfulness ---
    datasets["helpfulness"] = DatasetInfo(
        name="helpfulness",
        family="ultrafeedback",
        description=("UltraFeedback — GPT-4 helpfulness rating, normalized to [0,1] (17 models x 63,967 prompts)"),
        response_type="continuous",
        n_subjects=17,
        n_items=63967,
        subject_entity="LLM",
        item_entity="prompt",
        filename="helpfulness.pt",
        citation=_UF_CITATION,
        url=_UF_URL,
        license="MIT",
        tags=["nlp", "helpfulness", "gpt4-rated"],
    )

    # --- Honesty ---
    datasets["honesty"] = DatasetInfo(
        name="honesty",
        family="ultrafeedback",
        description=("UltraFeedback — GPT-4 honesty rating, normalized to [0,1] (17 models x 63,967 prompts)"),
        response_type="continuous",
        n_subjects=17,
        n_items=63967,
        subject_entity="LLM",
        item_entity="prompt",
        filename="honesty.pt",
        citation=_UF_CITATION,
        url=_UF_URL,
        license="MIT",
        tags=["nlp", "honesty", "gpt4-rated"],
    )

    # --- Instruction Following ---
    datasets["instruction_following"] = DatasetInfo(
        name="instruction_following",
        family="ultrafeedback",
        description=(
            "UltraFeedback — GPT-4 instruction-following rating, normalized to [0,1] (17 models x 63,967 prompts)"
        ),
        response_type="continuous",
        n_subjects=17,
        n_items=63967,
        subject_entity="LLM",
        item_entity="prompt",
        filename="instruction_following.pt",
        citation=_UF_CITATION,
        url=_UF_URL,
        license="MIT",
        tags=["nlp", "instruction-following", "gpt4-rated"],
    )

    # --- Truthfulness ---
    datasets["truthfulness"] = DatasetInfo(
        name="truthfulness",
        family="ultrafeedback",
        description=("UltraFeedback — GPT-4 truthfulness rating, normalized to [0,1] (17 models x 63,967 prompts)"),
        response_type="continuous",
        n_subjects=17,
        n_items=63967,
        subject_entity="LLM",
        item_entity="prompt",
        filename="truthfulness.pt",
        citation=_UF_CITATION,
        url=_UF_URL,
        license="MIT",
        tags=["nlp", "truthfulness", "gpt4-rated"],
    )

    return datasets
