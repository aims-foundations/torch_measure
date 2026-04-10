# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Prometheus evaluation dataset definitions.

Prometheus (KAIST) is a fine-grained evaluation framework that uses custom
rubrics to evaluate LLM responses.  GPT-4 scores or compares model responses
across 996 unique evaluation criteria.

Each response matrix has:

- **Rows (subjects)**: Evaluation rubric criteria (996 unique criteria).
- **Columns (items)**: Instance indices within each criterion.
- **Values**: Normalized [0, 1] scores (Feedback-Collection, from 1-5 scale)
  or binary {0, 1} preferences (Preference-Collection).

Source data:
  - ``prometheus-eval/Feedback-Collection`` on HuggingFace (~100K instances)
  - ``prometheus-eval/Preference-Collection`` on HuggingFace (~200K instances)
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_PROMETHEUS_URL = "https://huggingface.co/datasets/prometheus-eval/Feedback-Collection"

_PROMETHEUS_CITATION = (
    "@inproceedings{kim2024prometheus,\n"
    "  title={Prometheus: Inducing Fine-grained Evaluation Capability "
    "in Language Models},\n"
    "  author={Kim, Seungone and Shin, Jamin and Cho, Yejin and Jang, Joel "
    "and Longpre, Shayne and Lee, Hwaran and Yun, Sangdoo and Shin, Seongjin "
    "and Kim, Sungdong and Thorne, James and Seo, Minjoon},\n"
    "  booktitle={Proceedings of ICLR},\n"
    "  year={2024},\n"
    "  url={https://arxiv.org/abs/2310.08491}\n"
    "}"
)

_PROMETHEUS2_CITATION = (
    "@inproceedings{kim2024prometheus2,\n"
    "  title={Prometheus 2: An Open Source Language Model Specialized "
    "in Evaluating Other Language Models},\n"
    "  author={Kim, Seungone and Suk, Juyoung and Longpre, Shayne and "
    "Lin, Bill Yuchen and Shin, Jamin and Welleck, Sean and Neubig, Graham "
    "and Lee, Moontae and Lee, Kyungjae and Seo, Minjoon},\n"
    "  booktitle={Proceedings of EMNLP},\n"
    "  year={2024},\n"
    "  url={https://arxiv.org/abs/2405.01535}\n"
    "}"
)


def _register_prometheus_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for Prometheus evaluation response matrices.

    Registers two datasets:

    - ``prometheus/feedback``: GPT-4 absolute scores (1-5, normalized to [0,1])
      on ~100K evaluation instances across 996 rubric criteria.
    - ``prometheus/preference``: GPT-4 pairwise preferences (binary) on ~200K
      comparison instances across 996 rubric criteria.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Feedback-Collection (absolute scoring) ---

    datasets["feedback"] = DatasetInfo(
        name="feedback",
        family="prometheus",
        description=(
            "Prometheus Feedback-Collection — GPT-4 rubric-based scores, "
            "normalized to [0,1] (996 criteria x 100 instances)"
        ),
        response_type="continuous",
        n_subjects=996,
        n_items=100,
        subject_entity="rubric_criteria",
        item_entity="instance",
        repo_id="aims-foundation/torch-measure-data",
        filename="feedback.pt",
        citation=_PROMETHEUS_CITATION,
        url="https://huggingface.co/datasets/prometheus-eval/Feedback-Collection",
        license="CC-BY-4.0",
        tags=["llm-as-judge", "gpt4-rated", "rubric", "fine-grained-eval"],
    )

    # --- Preference-Collection (pairwise comparison) ---

    datasets["preference"] = DatasetInfo(
        name="preference",
        family="prometheus",
        description=(
            "Prometheus Preference-Collection — GPT-4 pairwise preferences, "
            "binary (996 criteria x 200 instances)"
        ),
        response_type="binary",
        n_subjects=996,
        n_items=200,
        subject_entity="rubric_criteria",
        item_entity="instance",
        repo_id="aims-foundation/torch-measure-data",
        filename="preference.pt",
        citation=_PROMETHEUS2_CITATION,
        url="https://huggingface.co/datasets/prometheus-eval/Preference-Collection",
        license="CC-BY-4.0",
        tags=["llm-as-judge", "gpt4-rated", "preference", "pairwise", "rubric"],
    )

    return datasets
