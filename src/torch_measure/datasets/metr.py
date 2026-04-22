# Copyright (c) 2026 AIMS Foundations. MIT License.

"""METR eval-analysis-public dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_METR_URL = "https://github.com/METR/eval-analysis-public"

_METR_CITATION = (
    "@article{metr2025time,\n"
    "  title={Measuring AI Ability to Complete Long Tasks},\n"
    "  author={{METR}},\n"
    "  year={2025},\n"
    "  url={https://metr.github.io/eval-analysis-public/}\n"
    "}"
)


def _register_metr_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for METR eval-analysis response matrices.

    METR evaluates AI agents on real-world tasks spanning software engineering,
    machine learning, cybersecurity, and more.  Each response matrix has:

    - **Rows**: AI agents (LLM + scaffold combinations).
    - **Columns**: Individual tasks from HCAST, RE-Bench, and SWAA benchmarks.
    - **Values**: Continuous [0, 1] pass rates or scores averaged across
      multiple evaluation runs per (agent, task) pair.

    Source data: ``METR/eval-analysis-public`` on GitHub (``runs.jsonl``).
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- All tasks (pass rate from score_binarized) ---

    datasets["all"] = DatasetInfo(
        name="all",
        family="metr",
        description="All METR tasks, mean pass rate across runs (34 agents x 170 tasks)",
        response_type="continuous",
        n_subjects=34,
        n_items=170,
        subject_entity="LLM",
        item_entity="task",
        filename="all.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "multi-benchmark", "pass-rate"],
    )

    datasets["all_score"] = DatasetInfo(
        name="all_score",
        family="metr",
        description="All METR tasks, mean continuous score across runs (34 agents x 170 tasks)",
        response_type="continuous",
        n_subjects=34,
        n_items=170,
        subject_entity="LLM",
        item_entity="task",
        filename="all_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "multi-benchmark", "continuous"],
    )

    # --- Per task-source splits (pass rate) ---

    datasets["hcast"] = DatasetInfo(
        name="hcast",
        family="metr",
        description="HCAST tasks, mean pass rate across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=97,
        subject_entity="LLM",
        item_entity="task",
        filename="hcast.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "hcast", "pass-rate"],
    )

    datasets["rebench"] = DatasetInfo(
        name="rebench",
        family="metr",
        description="RE-Bench tasks, mean pass rate across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=7,
        subject_entity="LLM",
        item_entity="task",
        filename="rebench.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "rebench", "pass-rate"],
    )

    datasets["swaa"] = DatasetInfo(
        name="swaa",
        family="metr",
        description="SWAA tasks, mean pass rate across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=66,
        subject_entity="LLM",
        item_entity="task",
        filename="swaa.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "swaa", "pass-rate"],
    )

    # --- Per task-source splits (continuous score) ---

    datasets["hcast_score"] = DatasetInfo(
        name="hcast_score",
        family="metr",
        description="HCAST tasks, mean continuous score across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=97,
        subject_entity="LLM",
        item_entity="task",
        filename="hcast_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "hcast", "continuous"],
    )

    datasets["rebench_score"] = DatasetInfo(
        name="rebench_score",
        family="metr",
        description="RE-Bench tasks, mean continuous score across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=7,
        subject_entity="LLM",
        item_entity="task",
        filename="rebench_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "rebench", "continuous"],
    )

    datasets["swaa_score"] = DatasetInfo(
        name="swaa_score",
        family="metr",
        description="SWAA tasks, mean continuous score across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=66,
        subject_entity="LLM",
        item_entity="task",
        filename="swaa_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "swaa", "continuous"],
    )

    return datasets
