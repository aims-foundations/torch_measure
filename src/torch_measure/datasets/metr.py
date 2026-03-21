# Copyright (c) 2026 AIMS Foundation. MIT License.

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

    datasets["metr/all"] = DatasetInfo(
        name="metr/all",
        family="metr",
        description="All METR tasks, mean pass rate across runs (34 agents x 170 tasks)",
        response_type="continuous",
        n_subjects=34,
        n_items=170,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/all.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "multi-benchmark", "pass-rate"],
    )

    datasets["metr/all_score"] = DatasetInfo(
        name="metr/all_score",
        family="metr",
        description="All METR tasks, mean continuous score across runs (34 agents x 170 tasks)",
        response_type="continuous",
        n_subjects=34,
        n_items=170,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/all_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "multi-benchmark", "continuous"],
    )

    # --- Per task-source splits (pass rate) ---

    datasets["metr/hcast"] = DatasetInfo(
        name="metr/hcast",
        family="metr",
        description="HCAST tasks, mean pass rate across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=97,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/hcast.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "hcast", "pass-rate"],
    )

    datasets["metr/rebench"] = DatasetInfo(
        name="metr/rebench",
        family="metr",
        description="RE-Bench tasks, mean pass rate across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=7,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/rebench.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "rebench", "pass-rate"],
    )

    datasets["metr/swaa"] = DatasetInfo(
        name="metr/swaa",
        family="metr",
        description="SWAA tasks, mean pass rate across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=66,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/swaa.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "swaa", "pass-rate"],
    )

    # --- Per task-source splits (continuous score) ---

    datasets["metr/hcast_score"] = DatasetInfo(
        name="metr/hcast_score",
        family="metr",
        description="HCAST tasks, mean continuous score across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=97,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/hcast_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "hcast", "continuous"],
    )

    datasets["metr/rebench_score"] = DatasetInfo(
        name="metr/rebench_score",
        family="metr",
        description="RE-Bench tasks, mean continuous score across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=7,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/rebench_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "rebench", "continuous"],
    )

    datasets["metr/swaa_score"] = DatasetInfo(
        name="metr/swaa_score",
        family="metr",
        description="SWAA tasks, mean continuous score across runs",
        response_type="continuous",
        n_subjects=34,
        n_items=66,
        subject_entity="LLM",
        item_entity="task",
        filename="metr/swaa_score.pt",
        citation=_METR_CITATION,
        url=_METR_URL,
        license="MIT",
        tags=["agentic", "swaa", "continuous"],
    )

    return datasets
