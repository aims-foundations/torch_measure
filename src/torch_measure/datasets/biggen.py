# Copyright (c) 2026 AIMS Foundations. MIT License.

"""BiGGen-Bench Results dataset definitions.

BiGGen-Bench (NAACL 2025) evaluates frontier LLMs across 70 tasks and 8
capabilities, scored by 5 different LLM judges on a 1-5 scale.  The fully
crossed (examinee x item x judge) design makes it ideal for Generalizability
Theory (G-theory) studies.

Each per-judge response matrix has:

- **Rows (subjects)**: LLMs (99 models).
- **Columns (items)**: Task/instance identifiers (695 items).
- **Values**: Continuous [0, 1] scores normalized from 1-5 via (score-1)/4.
  NaN for missing evaluations.

The combined ``biggen/all_judges`` dataset provides a 3D tensor
(subjects x items x judges) for multi-facet analyses.

Source data: ``prometheus-eval/BiGGen-Bench-Results`` on HuggingFace Hub.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_BIGGEN_URL = "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results"

_BIGGEN_CITATION = (
    "@inproceedings{kim2024biggen,\n"
    "  title={BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation "
    "of Language Models with Language Models},\n"
    "  author={Kim, Seungone and Suk, Juyoung and Longpre, Shayne and Lin, "
    "Bill Yuchen and Shin, Jamin and Welleck, Sean and Neubig, Graham and "
    "Lee, Moontae and Lee, Kyungjae and Seo, Minjoon},\n"
    "  booktitle={Proceedings of NAACL},\n"
    "  year={2025},\n"
    "  url={https://arxiv.org/abs/2406.05761}\n"
    "}"
)


def _register_biggen_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for BiGGen-Bench Results response matrices.

    Registers 5 per-judge 2D matrices and 1 combined 3D tensor (6 total).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Per-judge 2D response matrices (99 models x 695 items) ---

    datasets["gpt4"] = DatasetInfo(
        name="gpt4",
        family="biggen",
        description="BiGGen-Bench scored by GPT-4-1106 (99 models x 695 items)",
        response_type="continuous",
        n_subjects=99,
        n_items=695,
        subject_entity="LLM",
        item_entity="task",
        filename="gpt4.pt",
        citation=_BIGGEN_CITATION,
        url=_BIGGEN_URL,
        license="Apache-2.0",
        tags=["multi-judge", "g-theory", "llm-as-judge", "gpt4"],
    )

    datasets["gpt4_turbo"] = DatasetInfo(
        name="gpt4_turbo",
        family="biggen",
        description="BiGGen-Bench scored by GPT-4-Turbo-2024-04-09 (99 models x 695 items)",
        response_type="continuous",
        n_subjects=99,
        n_items=695,
        subject_entity="LLM",
        item_entity="task",
        filename="gpt4_turbo.pt",
        citation=_BIGGEN_CITATION,
        url=_BIGGEN_URL,
        license="Apache-2.0",
        tags=["multi-judge", "g-theory", "llm-as-judge", "gpt4-turbo"],
    )

    datasets["claude"] = DatasetInfo(
        name="claude",
        family="biggen",
        description="BiGGen-Bench scored by Claude-3-Opus (99 models x 695 items)",
        response_type="continuous",
        n_subjects=99,
        n_items=695,
        subject_entity="LLM",
        item_entity="task",
        filename="claude.pt",
        citation=_BIGGEN_CITATION,
        url=_BIGGEN_URL,
        license="Apache-2.0",
        tags=["multi-judge", "g-theory", "llm-as-judge", "claude"],
    )

    datasets["prometheus"] = DatasetInfo(
        name="prometheus",
        family="biggen",
        description="BiGGen-Bench scored by Prometheus-2-8x7B (99 models x 695 items)",
        response_type="continuous",
        n_subjects=99,
        n_items=695,
        subject_entity="LLM",
        item_entity="task",
        filename="prometheus.pt",
        citation=_BIGGEN_CITATION,
        url=_BIGGEN_URL,
        license="Apache-2.0",
        tags=["multi-judge", "g-theory", "llm-as-judge", "prometheus"],
    )

    datasets["prometheus_bgb"] = DatasetInfo(
        name="prometheus_bgb",
        family="biggen",
        description="BiGGen-Bench scored by Prometheus-2-8x7B-BGB (99 models x 695 items)",
        response_type="continuous",
        n_subjects=99,
        n_items=695,
        subject_entity="LLM",
        item_entity="task",
        filename="prometheus_bgb.pt",
        citation=_BIGGEN_CITATION,
        url=_BIGGEN_URL,
        license="Apache-2.0",
        tags=["multi-judge", "g-theory", "llm-as-judge", "prometheus"],
    )

    # --- Combined 3D tensor (99 models x 695 items x 5 judges) ---

    datasets["all_judges"] = DatasetInfo(
        name="all_judges",
        family="biggen",
        description="BiGGen-Bench all 5 judges combined, 3D tensor (99 models x 695 items x 5 judges)",
        response_type="continuous",
        n_subjects=99,
        n_items=695,
        subject_entity="LLM",
        item_entity="task",
        filename="all_judges.pt",
        citation=_BIGGEN_CITATION,
        url=_BIGGEN_URL,
        license="Apache-2.0",
        tags=["multi-judge", "g-theory", "llm-as-judge", "3d-tensor", "fully-crossed"],
    )

    return datasets
