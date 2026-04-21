# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Agentic benchmark dataset definitions (HAL Leaderboard)."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_HAL_URL = "https://hal.cs.princeton.edu/"

_HAL_CITATION = (
    "@article{gu2025hal,\n"
    "  title={HAL: A Holistic Agent Leaderboard for Evaluating AI Agents},\n"
    "  author={Gu, Alex and others},\n"
    "  year={2025}\n"
    "}"
)


def _register_agentic_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for agentic benchmark response matrices.

    These datasets come from the HAL (Holistic Agent Leaderboard) ecosystem,
    evaluating AI agents on real-world tasks spanning software engineering,
    scientific computing, web navigation, and more.

    Each binary response matrix has:

    - **Rows**: AI agents (LLM + scaffolding combinations).
    - **Columns**: Individual tasks from each benchmark.
    - **Values**: Binary {0, 1} for pass/fail (NaN for missing).

    Score variants provide continuous scores (e.g., partial credit, CodeBERT
    similarity) for benchmarks that support graded evaluation.

    Post-revision reruns average multiple evaluation runs per (agent, task)
    pair, producing continuous [0, 1] pass-rate values for reliability analysis.

    Source data: ``aims-foundation/eval_response_matrix`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Pre-revision binary response matrices (10 benchmarks) ---

    datasets["swebench"] = DatasetInfo(
        name="swebench",
        family="agentic",
        description="SWE-bench Verified Mini (real-world GitHub issue resolution)",
        response_type="binary",
        n_subjects=44,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        filename="swebench.pt",
        citation=(
            "@article{jimenez2024swebench,\n"
            "  title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},\n"
            "  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and "
            "Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},\n"
            "  journal={Proceedings of ICLR},\n"
            "  year={2024}\n"
            "}"
        ),
        url="https://www.swebench.com/",
        license="MIT",
        tags=["agentic", "code", "software-engineering", "github"],
    )

    datasets["assistantbench"] = DatasetInfo(
        name="assistantbench",
        family="agentic",
        description="AssistantBench (realistic web assistant tasks)",
        response_type="binary",
        n_subjects=31,
        n_items=33,
        subject_entity="LLM",
        item_entity="task",
        filename="assistantbench.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "web", "assistant"],
    )

    datasets["colbench"] = DatasetInfo(
        name="colbench",
        family="agentic",
        description="ColBench (collaborative backend programming)",
        response_type="binary",
        n_subjects=22,
        n_items=1000,
        subject_entity="LLM",
        item_entity="task",
        filename="colbench.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "collaborative-programming"],
    )

    datasets["corebench_hard"] = DatasetInfo(
        name="corebench_hard",
        family="agentic",
        description="CORE-Bench Hard (computational reproducibility of scientific papers)",
        response_type="binary",
        n_subjects=59,
        n_items=45,
        subject_entity="LLM",
        item_entity="task",
        filename="corebench_hard.pt",
        citation=(
            "@article{siegel2024corebench,\n"
            "  title={CORE-Bench: Fostering the Credibility of Published Research "
            "Through a Computational Reproducibility Agent Benchmark},\n"
            "  author={Siegel, Zachary S and others},\n"
            "  journal={arXiv preprint arXiv:2409.11363},\n"
            "  year={2024}\n"
            "}"
        ),
        url="https://github.com/siegelz/core-bench",
        license="MIT",
        tags=["agentic", "science", "reproducibility"],
    )

    datasets["gaia"] = DatasetInfo(
        name="gaia",
        family="agentic",
        description="GAIA (General AI Assistants benchmark)",
        response_type="binary",
        n_subjects=35,
        n_items=165,
        subject_entity="LLM",
        item_entity="task",
        filename="gaia.pt",
        citation=(
            "@article{mialon2023gaia,\n"
            "  title={GAIA: A Benchmark for General AI Assistants},\n"
            "  author={Mialon, Gr{\\'e}goire and Fourrier, Cl{\\'e}mentine and "
            "Tong, Shengyi and others},\n"
            "  journal={arXiv preprint arXiv:2311.12983},\n"
            "  year={2023}\n"
            "}"
        ),
        url="https://huggingface.co/gaia-benchmark",
        license="MIT",
        tags=["agentic", "general", "assistant", "web"],
    )

    datasets["mind2web"] = DatasetInfo(
        name="mind2web",
        family="agentic",
        description="Mind2Web Online (web navigation and interaction)",
        response_type="binary",
        n_subjects=29,
        n_items=336,
        subject_entity="LLM",
        item_entity="task",
        filename="mind2web.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "web", "navigation"],
    )

    datasets["scicode"] = DatasetInfo(
        name="scicode",
        family="agentic",
        description="SciCode (scientific code generation)",
        response_type="binary",
        n_subjects=37,
        n_items=65,
        subject_entity="LLM",
        item_entity="task",
        filename="scicode.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "science"],
    )

    datasets["scienceagentbench"] = DatasetInfo(
        name="scienceagentbench",
        family="agentic",
        description="ScienceAgentBench (data-driven scientific discovery)",
        response_type="binary",
        n_subjects=25,
        n_items=102,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench.pt",
        citation=(
            "@article{chen2024scienceagentbench,\n"
            "  title={ScienceAgentBench: Toward Rigorous Assessment of "
            "Language Agents for Data-Driven Scientific Discovery},\n"
            "  author={Chen, Ziru and others},\n"
            "  journal={Proceedings of ICLR},\n"
            "  year={2025}\n"
            "}"
        ),
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "data-analysis"],
    )

    datasets["taubench_airline"] = DatasetInfo(
        name="taubench_airline",
        family="agentic",
        description="Tau-bench Airline (customer service agent benchmark)",
        response_type="binary",
        n_subjects=47,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        filename="taubench_airline.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "customer-service", "dialogue"],
    )

    datasets["usaco"] = DatasetInfo(
        name="usaco",
        family="agentic",
        description="USACO (competitive programming problems)",
        response_type="binary",
        n_subjects=14,
        n_items=307,
        subject_entity="LLM",
        item_entity="task",
        filename="usaco.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "competitive-programming"],
    )

    # --- Pre-revision continuous score variants ---

    datasets["colbench_raw_score"] = DatasetInfo(
        name="colbench_raw_score",
        family="agentic",
        description="ColBench raw scores (continuous 0-1, partial credit)",
        response_type="continuous",
        n_subjects=22,
        n_items=1000,
        subject_entity="LLM",
        item_entity="task",
        filename="colbench_raw_score.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "collaborative-programming", "continuous"],
    )

    datasets["corebench_hard_vision_score"] = DatasetInfo(
        name="corebench_hard_vision_score",
        family="agentic",
        description="CORE-Bench Hard vision scores (figure reproduction accuracy)",
        response_type="continuous",
        n_subjects=59,
        n_items=45,
        subject_entity="LLM",
        item_entity="task",
        filename="corebench_hard_vision_score.pt",
        citation=_HAL_CITATION,
        url="https://github.com/siegelz/core-bench",
        license="MIT",
        tags=["agentic", "science", "reproducibility", "continuous"],
    )

    datasets["corebench_hard_written_score"] = DatasetInfo(
        name="corebench_hard_written_score",
        family="agentic",
        description="CORE-Bench Hard written scores (text reproduction accuracy)",
        response_type="continuous",
        n_subjects=59,
        n_items=45,
        subject_entity="LLM",
        item_entity="task",
        filename="corebench_hard_written_score.pt",
        citation=_HAL_CITATION,
        url="https://github.com/siegelz/core-bench",
        license="MIT",
        tags=["agentic", "science", "reproducibility", "continuous"],
    )

    datasets["scienceagentbench_codebert_score"] = DatasetInfo(
        name="scienceagentbench_codebert_score",
        family="agentic",
        description="ScienceAgentBench CodeBERT similarity scores (continuous 0-1)",
        response_type="continuous",
        n_subjects=25,
        n_items=102,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_codebert_score.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "code-similarity", "continuous"],
    )

    datasets["scienceagentbench_success_rate"] = DatasetInfo(
        name="scienceagentbench_success_rate",
        family="agentic",
        description="ScienceAgentBench success rate (binary execution success)",
        response_type="binary",
        n_subjects=25,
        n_items=102,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_success_rate.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "execution"],
    )

    datasets["scienceagentbench_valid_program"] = DatasetInfo(
        name="scienceagentbench_valid_program",
        family="agentic",
        description="ScienceAgentBench valid program (binary program validity)",
        response_type="binary",
        n_subjects=25,
        n_items=102,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_valid_program.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "validation"],
    )

    # --- Post-revision rerun averages (continuous pass rates) ---

    datasets["colbench_rerun"] = DatasetInfo(
        name="colbench_rerun",
        family="agentic",
        description="ColBench rerun pass rates (averaged across 54 runs)",
        response_type="continuous",
        n_subjects=11,
        n_items=1000,
        subject_entity="LLM",
        item_entity="task",
        filename="colbench_rerun.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "rerun", "reliability", "continuous"],
    )

    datasets["colbench_rerun_raw_score"] = DatasetInfo(
        name="colbench_rerun_raw_score",
        family="agentic",
        description="ColBench rerun raw scores (averaged across 54 runs, continuous 0-1)",
        response_type="continuous",
        n_subjects=11,
        n_items=1000,
        subject_entity="LLM",
        item_entity="task",
        filename="colbench_rerun_raw_score.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "rerun", "reliability", "continuous"],
    )

    datasets["corebench_hard_rerun"] = DatasetInfo(
        name="corebench_hard_rerun",
        family="agentic",
        description="CORE-Bench Hard rerun pass rates (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=45,
        subject_entity="LLM",
        item_entity="task",
        filename="corebench_hard_rerun.pt",
        citation=_HAL_CITATION,
        url="https://github.com/siegelz/core-bench",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    datasets["corebench_hard_rerun_vision_score"] = DatasetInfo(
        name="corebench_hard_rerun_vision_score",
        family="agentic",
        description="CORE-Bench Hard rerun vision scores (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=45,
        subject_entity="LLM",
        item_entity="task",
        filename="corebench_hard_rerun_vision_score.pt",
        citation=_HAL_CITATION,
        url="https://github.com/siegelz/core-bench",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    datasets["corebench_hard_rerun_written_score"] = DatasetInfo(
        name="corebench_hard_rerun_written_score",
        family="agentic",
        description="CORE-Bench Hard rerun written scores (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=45,
        subject_entity="LLM",
        item_entity="task",
        filename="corebench_hard_rerun_written_score.pt",
        citation=_HAL_CITATION,
        url="https://github.com/siegelz/core-bench",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    datasets["scicode_rerun"] = DatasetInfo(
        name="scicode_rerun",
        family="agentic",
        description="SciCode rerun pass rates (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=29,
        subject_entity="LLM",
        item_entity="task",
        filename="scicode_rerun.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "science", "rerun", "reliability", "continuous"],
    )

    datasets["scicode_rerun_subtask_score"] = DatasetInfo(
        name="scicode_rerun_subtask_score",
        family="agentic",
        description="SciCode rerun subtask scores (averaged across 11 runs, continuous 0-1)",
        response_type="continuous",
        n_subjects=8,
        n_items=29,
        subject_entity="LLM",
        item_entity="task",
        filename="scicode_rerun_subtask_score.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "code", "science", "rerun", "reliability", "continuous"],
    )

    datasets["scienceagentbench_rerun"] = DatasetInfo(
        name="scienceagentbench_rerun",
        family="agentic",
        description="ScienceAgentBench rerun pass rates (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=1212,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_rerun.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    datasets["scienceagentbench_rerun_codebert_score"] = DatasetInfo(
        name="scienceagentbench_rerun_codebert_score",
        family="agentic",
        description="ScienceAgentBench rerun CodeBERT scores (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=1212,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_rerun_codebert_score.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    datasets["scienceagentbench_rerun_success_rate"] = DatasetInfo(
        name="scienceagentbench_rerun_success_rate",
        family="agentic",
        description="ScienceAgentBench rerun success rates (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=1212,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_rerun_success_rate.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    datasets["scienceagentbench_rerun_valid_program"] = DatasetInfo(
        name="scienceagentbench_rerun_valid_program",
        family="agentic",
        description="ScienceAgentBench rerun valid program rates (averaged across 11 runs)",
        response_type="continuous",
        n_subjects=8,
        n_items=1212,
        subject_entity="LLM",
        item_entity="task",
        filename="scienceagentbench_rerun_valid_program.pt",
        citation=_HAL_CITATION,
        url="https://osu-nlp-group.github.io/ScienceAgentBench/",
        license="MIT",
        tags=["agentic", "science", "rerun", "reliability", "continuous"],
    )

    # --- Aggregate ---

    datasets["all"] = DatasetInfo(
        name="all",
        family="agentic",
        description="All agentic benchmarks concatenated (pre-revision binary, 10 benchmarks)",
        response_type="binary",
        n_subjects=272,
        n_items=2153,
        subject_entity="LLM",
        item_entity="task",
        filename="all.pt",
        citation=_HAL_CITATION,
        url=_HAL_URL,
        license="MIT",
        tags=["agentic", "aggregate", "multi-benchmark"],
    )

    return datasets
