#!/usr/bin/env python3
"""Migrate benchmark CSV response matrices to .pt format on HuggingFace Hub.

Reads CSV response matrices from data/<benchmark>_data/processed/, converts them
to the standardized .pt dict format, and optionally uploads to HuggingFace Hub.

Usage:
    # Convert only (no upload) — prints dimensions for bench.py registry
    python scripts/migrate_bench_data.py --no-upload

    # Convert and upload (uses locally stored HF credentials)
    python scripts/migrate_bench_data.py

    # Use a custom source directory
    python scripts/migrate_bench_data.py --src-dir /path/to/data

Source data:
    data/<benchmark>_data/processed/*.csv

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,       # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],    # model/agent identifiers
        "item_ids": list[str],       # task/item identifiers
        "item_contents": list[str],  # task description/instruction when available, else ""
        "subject_metadata": None,
    }
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent.parent / "data"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path("/tmp/torch_measure_bench_migration")


@dataclass
class BenchmarkSpec:
    """Describes how to convert one CSV file into a .pt payload."""

    benchmark: str        # e.g., "agentdojo" (directory prefix)
    csv_file: str         # filename within processed/
    registry_name: str    # e.g., "bench/agentdojo"
    response_type: str    # "binary" or "continuous"
    orientation: str      # "items_as_rows" or "models_as_rows"
    description: str
    url: str = ""
    tags: list[str] = field(default_factory=list)
    # Columns to strip before building the data tensor (by name).
    metadata_cols: list[str] = field(default_factory=list)
    # Which column to use as the index (item or model IDs). Default: 0.
    index_col: int | str = 0
    # Multiply all data values by this factor (e.g., 0.01 to convert 0-100 → 0-1).
    scale_factor: float = 1.0
    # For clinebench: columns to extract as subjects (model names).
    subject_cols: list[str] | None = None
    # Parse fraction strings like "2/3" → 0.667 and "X" → 0.0.
    parse_fractions: bool = False
    # Item content enrichment: metadata file (relative to processed/), key column, and content column.
    content_file: str | None = None
    content_key: str | None = None
    content_col: str | None = None


# ---------------------------------------------------------------------------
# Complete benchmark manifest
# ---------------------------------------------------------------------------

SPECS: list[BenchmarkSpec] = [
    # ── AgentDojo ──────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="agentdojo",
        csv_file="response_matrix.csv",
        registry_name="bench/agentdojo",
        response_type="binary",
        orientation="items_as_rows",
        description="AgentDojo — tool-use agent utility tasks",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="agentdojo",
        csv_file="response_matrix_security.csv",
        registry_name="bench/agentdojo_security",
        response_type="binary",
        orientation="items_as_rows",
        description="AgentDojo — security evaluation",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],    ),
    BenchmarkSpec(
        benchmark="agentdojo",
        csv_file="response_matrix_utility_under_attack.csv",
        registry_name="bench/agentdojo_utility_attack",
        response_type="binary",
        orientation="items_as_rows",
        description="AgentDojo — utility under attack",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── AppWorld ───────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="appworld",
        csv_file="response_matrix.csv",
        registry_name="bench/appworld",
        response_type="continuous",
        orientation="models_as_rows",
        description="AppWorld — multi-app interaction tasks",
        url="https://appworld.dev/",
        tags=["agentic", "app-interaction"],
        metadata_cols=["id", "method", "method_full", "llm", "llm_full", "url", "date"],
        index_col="model_agent",
        scale_factor=0.01,    ),
    # ── BFCL ──────────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="bfcl",
        csv_file="response_matrix.csv",
        registry_name="bench/bfcl",
        response_type="binary",
        orientation="models_as_rows",
        description="BFCL v3 — function calling (pass/fail)",
        url="https://gorilla.cs.berkeley.edu/leaderboard.html",
        tags=["coding", "function-calling"],
    ),
    # NOTE: error_type_matrix.csv and error_type_coarse_matrix.csv contain string
    # labels (e.g., "pass", "value_error:exec_result_count"), not numeric values.
    # They are not usable as response matrices and are excluded.
    # ── BigCodeBench ──────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="bigcodebench",
        csv_file="response_matrix.csv",
        registry_name="bench/bigcodebench",
        response_type="binary",
        orientation="models_as_rows",
        description="BigCodeBench — complete split",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="bigcodebench",
        csv_file="response_matrix_instruct.csv",
        registry_name="bench/bigcodebench_instruct",
        response_type="binary",
        orientation="models_as_rows",
        description="BigCodeBench — instruct split",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="bigcodebench",
        csv_file="response_matrix_hard_complete.csv",
        registry_name="bench/bigcodebench_hard_complete",
        response_type="binary",
        orientation="models_as_rows",
        description="BigCodeBench — hard tasks, complete",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="bigcodebench",
        csv_file="response_matrix_hard_instruct.csv",
        registry_name="bench/bigcodebench_hard_instruct",
        response_type="binary",
        orientation="models_as_rows",
        description="BigCodeBench — hard tasks, instruct",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── ClineBench ────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="clinebench",
        csv_file="results_matrix.csv",
        registry_name="bench/clinebench",
        response_type="continuous",
        orientation="items_as_rows",
        description="ClineBench — AI coding agent evaluation",
        url="https://github.com/cline/cline",
        tags=["coding", "agentic"],
        metadata_cols=[
            "short_name", "difficulty", "category", "num_tests",
            "terminus_notes", "cline_notes", "languages",
        ],
        subject_cols=["oracle_score", "terminus_score", "cline_score"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── CORE-Bench ────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="corebench",
        csv_file="response_matrix.csv",
        registry_name="bench/corebench",
        response_type="binary",
        orientation="items_as_rows",
        description="CORE-Bench — computational reproducibility (binary)",
        url="https://github.com/siegelz/core-bench",
        tags=["agentic", "reproducibility"],
        content_file="task_metadata.csv",
        content_key="task_id",
        content_col="capsule_title",
    ),
    BenchmarkSpec(
        benchmark="corebench",
        csv_file="response_matrix_scores.csv",
        registry_name="bench/corebench_scores",
        response_type="continuous",
        orientation="items_as_rows",
        description="CORE-Bench — computational reproducibility (scores)",
        url="https://github.com/siegelz/core-bench",
        tags=["agentic", "reproducibility"],
        content_file="task_metadata.csv",
        content_key="task_id",
        content_col="capsule_title",
    ),
    # ── CRUXEval ──────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="cruxeval",
        csv_file="response_matrix.csv",
        registry_name="bench/cruxeval",
        response_type="continuous",
        orientation="items_as_rows",
        description="CRUXEval — code reasoning (continuous)",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="cruxeval",
        csv_file="response_matrix_binary.csv",
        registry_name="bench/cruxeval_binary",
        response_type="binary",
        orientation="items_as_rows",
        description="CRUXEval — code reasoning (binary)",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="cruxeval",
        csv_file="response_matrix_input.csv",
        registry_name="bench/cruxeval_input",
        response_type="continuous",
        orientation="items_as_rows",
        description="CRUXEval — input prediction (continuous)",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="cruxeval",
        csv_file="response_matrix_input_binary.csv",
        registry_name="bench/cruxeval_input_binary",
        response_type="binary",
        orientation="items_as_rows",
        description="CRUXEval — input prediction (binary)",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="cruxeval",
        csv_file="response_matrix_output.csv",
        registry_name="bench/cruxeval_output",
        response_type="continuous",
        orientation="items_as_rows",
        description="CRUXEval — output prediction (continuous)",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="cruxeval",
        csv_file="response_matrix_output_binary.csv",
        registry_name="bench/cruxeval_output_binary",
        response_type="binary",
        orientation="items_as_rows",
        description="CRUXEval — output prediction (binary)",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── Cybench ───────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="cybench",
        csv_file="response_matrix.csv",
        registry_name="bench/cybench",
        response_type="binary",
        orientation="items_as_rows",
        description="Cybench — CTF cybersecurity tasks (unguided)",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
        content_file="task_metadata.csv",
        content_key="task_name",
        content_col="task_name",
    ),
    BenchmarkSpec(
        benchmark="cybench",
        csv_file="response_matrix_subtask_guided.csv",
        registry_name="bench/cybench_guided",
        response_type="binary",
        orientation="items_as_rows",
        description="Cybench — CTF tasks (subtask-guided)",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
        content_file="task_metadata.csv",
        content_key="task_name",
        content_col="task_name",
    ),
    BenchmarkSpec(
        benchmark="cybench",
        csv_file="response_matrix_subtask_scores.csv",
        registry_name="bench/cybench_scores",
        response_type="continuous",
        orientation="items_as_rows",
        description="Cybench — CTF tasks (subtask completion scores)",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
        parse_fractions=True,
        content_file="task_metadata.csv",
        content_key="task_name",
        content_col="task_name",
    ),
    # ── DPAI ──────────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="dpai",
        csv_file="response_matrix_total_score.csv",
        registry_name="bench/dpai",
        response_type="continuous",
        orientation="items_as_rows",
        description="DPAI Arena — total score",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
        scale_factor=0.01,
        content_file="task_metadata.csv",
        content_key="instance_id",
        content_col="problem_statement_preview",
    ),
    BenchmarkSpec(
        benchmark="dpai",
        csv_file="response_matrix_blind_score.csv",
        registry_name="bench/dpai_blind",
        response_type="continuous",
        orientation="items_as_rows",
        description="DPAI Arena — blind evaluation score",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
        scale_factor=0.01,
        content_file="task_metadata.csv",
        content_key="instance_id",
        content_col="problem_statement_preview",
    ),
    BenchmarkSpec(
        benchmark="dpai",
        csv_file="response_matrix_informed_score.csv",
        registry_name="bench/dpai_informed",
        response_type="continuous",
        orientation="items_as_rows",
        description="DPAI Arena — informed evaluation score",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
        scale_factor=0.01,
        content_file="task_metadata.csv",
        content_key="instance_id",
        content_col="problem_statement_preview",
    ),
    BenchmarkSpec(
        benchmark="dpai",
        csv_file="response_matrix_binary_pass50.csv",
        registry_name="bench/dpai_binary",
        response_type="binary",
        orientation="items_as_rows",
        description="DPAI Arena — binary pass/fail (50% threshold)",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
        content_file="task_metadata.csv",
        content_key="instance_id",
        content_col="problem_statement_preview",
    ),
    # ── EDIT-Bench ────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="editbench",
        csv_file="response_matrix.csv",
        registry_name="bench/editbench",
        response_type="continuous",
        orientation="items_as_rows",
        description="EDIT-Bench — code editing (continuous scores)",
        url="https://github.com/waynchi/editbench",
        tags=["coding", "editing"],
        content_file="task_metadata.csv",
        content_key="task_id",
        content_col="instruction_preview",
    ),
    BenchmarkSpec(
        benchmark="editbench",
        csv_file="response_matrix_binary.csv",
        registry_name="bench/editbench_binary",
        response_type="binary",
        orientation="items_as_rows",
        description="EDIT-Bench — code editing (binary)",
        url="https://github.com/waynchi/editbench",
        tags=["coding", "editing"],
        content_file="task_metadata.csv",
        content_key="task_id",
        content_col="instruction_preview",
    ),
    # ── EvalPlus ──────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="evalplus",
        csv_file="response_matrix.csv",
        registry_name="bench/evalplus",
        response_type="binary",
        orientation="models_as_rows",
        description="EvalPlus — HumanEval+ and MBPP+ combined",
        url="https://evalplus.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="evalplus",
        csv_file="response_matrix_humaneval_base.csv",
        registry_name="bench/evalplus_humaneval_base",
        response_type="binary",
        orientation="models_as_rows",
        description="EvalPlus — HumanEval base",
        url="https://evalplus.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="evalplus",
        csv_file="response_matrix_humaneval_plus.csv",
        registry_name="bench/evalplus_humaneval_plus",
        response_type="binary",
        orientation="models_as_rows",
        description="EvalPlus — HumanEval+",
        url="https://evalplus.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="evalplus",
        csv_file="response_matrix_mbpp_base.csv",
        registry_name="bench/evalplus_mbpp_base",
        response_type="binary",
        orientation="models_as_rows",
        description="EvalPlus — MBPP base",
        url="https://evalplus.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="evalplus",
        csv_file="response_matrix_mbpp_plus.csv",
        registry_name="bench/evalplus_mbpp_plus",
        response_type="binary",
        orientation="models_as_rows",
        description="EvalPlus — MBPP+",
        url="https://evalplus.github.io/",
        tags=["coding"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── LiveCodeBench ─────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="livecodebench",
        csv_file="response_matrix.csv",
        registry_name="bench/livecodebench",
        response_type="continuous",
        orientation="models_as_rows",
        description="LiveCodeBench — competitive programming",
        url="https://livecodebench.github.io/",
        tags=["coding", "competitive-programming"],
        content_file="problem_metadata.csv",
        content_key="question_id",
        content_col="question_title",
    ),
    # ── MLE-bench ─────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="mlebench",
        csv_file="response_matrix.csv",
        registry_name="bench/mlebench",
        response_type="continuous",
        orientation="items_as_rows",
        description="MLE-bench — ML engineering (continuous scores)",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
        content_file="item_content.csv",
        content_key="competition_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="mlebench",
        csv_file="response_matrix_binary.csv",
        registry_name="bench/mlebench_binary",
        response_type="binary",
        orientation="items_as_rows",
        description="MLE-bench — ML engineering (binary)",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
        content_file="item_content.csv",
        content_key="competition_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="mlebench",
        csv_file="response_matrix_above_median.csv",
        registry_name="bench/mlebench_above_median",
        response_type="binary",
        orientation="items_as_rows",
        description="MLE-bench — ML engineering (above-median)",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
        content_file="item_content.csv",
        content_key="competition_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="mlebench",
        csv_file="response_matrix_scores.csv",
        registry_name="bench/mlebench_scores",
        response_type="continuous",
        orientation="items_as_rows",
        description="MLE-bench — ML engineering (raw scores)",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
        content_file="item_content.csv",
        content_key="competition_id",
        content_col="content",
    ),
    # ── MMLU-Pro ──────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="mmlupro",
        csv_file="response_matrix.csv",
        registry_name="bench/mmlupro",
        response_type="continuous",
        orientation="items_as_rows",
        description="MMLU-Pro — per-question accuracy (12K+ models)",
        url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        tags=["knowledge", "reasoning"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="mmlupro",
        csv_file="response_matrix_category.csv",
        registry_name="bench/mmlupro_category",
        response_type="continuous",
        orientation="items_as_rows",
        description="MMLU-Pro — per-category accuracy",
        url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        tags=["knowledge", "reasoning"],    ),
    # ── PaperBench ────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="paperbench",
        csv_file="response_matrix.csv",
        registry_name="bench/paperbench",
        response_type="continuous",
        orientation="items_as_rows",
        description="PaperBench — paper reproduction evaluation",
        url="https://arxiv.org/abs/2504.01848",
        tags=["agentic", "research"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="paperbench",
        csv_file="response_matrix_runs.csv",
        registry_name="bench/paperbench_runs",
        response_type="continuous",
        orientation="items_as_rows",
        description="PaperBench — per-run results",
        url="https://arxiv.org/abs/2504.01848",
        tags=["agentic", "research"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # NOTE: scienceagentbench_data/processed/response_matrix.csv is all NA
    # (per-task results not published). Only aggregate_results.csv has data.
    # Excluded from migration.
    # ── SWE-bench ─────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="swebench",
        csv_file="response_matrix.csv",
        registry_name="bench/swebench",
        response_type="binary",
        orientation="models_as_rows",
        description="SWE-bench Verified — GitHub issue resolution",
        url="https://www.swebench.com/",
        tags=["coding", "agentic", "github"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── SWE-PolyBench ─────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="swepolybench",
        csv_file="response_matrix_full.csv",
        registry_name="bench/swepolybench_full",
        response_type="binary",
        orientation="items_as_rows",
        description="SWE-PolyBench — multilingual SWE (full, 1 model)",
        url="https://github.com/amazon-science/SWE-PolyBench",
        tags=["coding", "multilingual"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="swepolybench",
        csv_file="response_matrix_verified.csv",
        registry_name="bench/swepolybench_verified",
        response_type="binary",
        orientation="items_as_rows",
        description="SWE-PolyBench — verified subset",
        url="https://github.com/amazon-science/SWE-PolyBench",
        tags=["coding", "multilingual"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── TAU-bench ─────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="taubench",
        csv_file="response_matrix_combined.csv",
        registry_name="bench/taubench",
        response_type="continuous",
        orientation="items_as_rows",
        description="TAU-bench — all domains combined",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
        metadata_cols=["domain"],
        index_col="task_id",
        content_file="item_content_combined.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="taubench",
        csv_file="response_matrix_v1_airline.csv",
        registry_name="bench/taubench_v1_airline",
        response_type="continuous",
        orientation="items_as_rows",
        description="TAU-bench — airline domain v1",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
        metadata_cols=["domain"],
        index_col="task_id",
        content_file="item_content_v1_airline.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="taubench",
        csv_file="response_matrix_v2_airline.csv",
        registry_name="bench/taubench_v2_airline",
        response_type="continuous",
        orientation="items_as_rows",
        description="TAU-bench — airline domain v2",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
        metadata_cols=["domain"],
        index_col="task_id",
        content_file="item_content_v2_airline.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="taubench",
        csv_file="response_matrix_hal_airline.csv",
        registry_name="bench/taubench_hal_airline",
        response_type="continuous",
        orientation="items_as_rows",
        description="TAU-bench — HAL airline evaluation",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
        metadata_cols=["domain"],
        index_col="task_id",
        content_file="item_content_hal_airline.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="taubench",
        csv_file="response_matrix_retail.csv",
        registry_name="bench/taubench_retail",
        response_type="continuous",
        orientation="items_as_rows",
        description="TAU-bench — retail domain",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
        metadata_cols=["domain"],
        index_col="task_id",
        content_file="item_content_retail.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="taubench",
        csv_file="response_matrix_telecom.csv",
        registry_name="bench/taubench_telecom",
        response_type="continuous",
        orientation="items_as_rows",
        description="TAU-bench — telecom domain",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
        metadata_cols=["domain"],
        index_col="task_id",
        content_file="item_content_telecom.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── Terminal-Bench ────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="terminal_bench",
        csv_file="binary_majority_matrix.csv",
        registry_name="bench/terminalbench",
        response_type="binary",
        orientation="models_as_rows",
        description="Terminal-Bench — CLI task solving (majority vote)",
        url="https://github.com/terminal-bench/terminal-bench",
        tags=["agentic", "cli"],
        content_file="tasks_complete_metadata.csv",
        content_key="task_name",
        content_col="instruction",
    ),
    BenchmarkSpec(
        benchmark="terminal_bench",
        csv_file="resolution_rate_matrix.csv",
        registry_name="bench/terminalbench_resolution",
        response_type="continuous",
        orientation="models_as_rows",
        description="Terminal-Bench — CLI task resolution rate",
        url="https://github.com/terminal-bench/terminal-bench",
        tags=["agentic", "cli"],
        content_file="tasks_complete_metadata.csv",
        content_key="task_name",
        content_col="instruction",
    ),
    # ── VisualWebArena ────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="visualwebarena",
        csv_file="response_matrix.csv",
        registry_name="bench/visualwebarena",
        response_type="continuous",
        orientation="items_as_rows",
        description="VisualWebArena — multimodal web navigation",
        url="https://visualwebarena.github.io/",
        tags=["agentic", "web", "multimodal"],
        content_file="task_metadata.csv",
        content_key="task_id",
        content_col="intent",
    ),
    # ── WebArena ──────────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="webarena",
        csv_file="webarena_response_matrix.csv",
        registry_name="bench/webarena",
        response_type="continuous",
        orientation="items_as_rows",
        description="WebArena — autonomous web agent tasks",
        url="https://webarena.dev/",
        tags=["agentic", "web"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── AgentBench (NEW) ─────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="agentbench",
        csv_file="response_matrix.csv",
        registry_name="bench/agentbench",
        response_type="continuous",
        orientation="models_as_rows",
        description="AgentBench — multi-environment agent evaluation",
        url="https://github.com/THUDM/AgentBench",
        tags=["agentic", "multi-domain"],
        scale_factor=0.01,
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── AlpacaEval (NEW) ─────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="alpacaeval",
        csv_file="response_matrix.csv",
        registry_name="bench/alpacaeval",
        response_type="binary",
        orientation="models_as_rows",
        description="AlpacaEval — instruction following (win/loss vs GPT-4)",
        url="https://tatsu-lab.github.io/alpaca_eval/",
        tags=["instruction-following", "nlp"],
        content_file="item_metadata.csv",
        content_key="item_idx",
        content_col="instruction",
    ),
    # NOTE: alpacaeval_preference.csv has 1-2 range scores (not 0-1), skipped.
    # ── AndroidWorld (NEW) ───────────────────────────────────────────
    BenchmarkSpec(
        benchmark="androidworld",
        csv_file="response_matrix.csv",
        registry_name="bench/androidworld",
        response_type="binary",
        orientation="models_as_rows",
        description="AndroidWorld — mobile device automation tasks",
        url="https://github.com/google-research/android_world",
        tags=["agentic", "mobile"],
        index_col="Agent",
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── BrowserGym (NEW) ─────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="browsergym",
        csv_file="response_matrix.csv",
        registry_name="bench/browsergym",
        response_type="continuous",
        orientation="models_as_rows",
        description="BrowserGym — web agent benchmark aggregates",
        url="https://github.com/ServiceNow/BrowserGym",
        tags=["agentic", "web"],
        index_col="Agent",
        scale_factor=0.01,
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── GAIA (NEW) ───────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="gaia",
        csv_file="response_matrix_hal.csv",
        registry_name="bench/gaia",
        response_type="binary",
        orientation="models_as_rows",
        description="GAIA — general AI assistant tasks",
        url="https://huggingface.co/gaia-benchmark",
        tags=["agentic", "general-purpose"],
        index_col="agent",

        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── LiveBench (NEW) ──────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="livebench",
        csv_file="response_matrix.csv",
        registry_name="bench/livebench",
        response_type="continuous",
        orientation="models_as_rows",
        description="LiveBench — contamination-free LLM benchmark (scores)",
        url="https://livebench.ai/",
        tags=["nlp", "reasoning", "coding", "math"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="livebench",
        csv_file="response_matrix_binary.csv",
        registry_name="bench/livebench_binary",
        response_type="binary",
        orientation="models_as_rows",
        description="LiveBench — contamination-free LLM benchmark (binary)",
        url="https://livebench.ai/",
        tags=["nlp", "reasoning", "coding", "math"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── MathArena (NEW) ──────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="matharena",
        csv_file="response_matrix_all_final_answer.csv",
        registry_name="bench/matharena",
        response_type="continuous",
        orientation="models_as_rows",
        description="MathArena — competition math (all competitions combined)",
        url="https://matharena.ai/",
        tags=["math", "reasoning"],
        index_col="model_name",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="matharena",
        csv_file="response_matrix_aime_2025_binary.csv",
        registry_name="bench/matharena_aime2025",
        response_type="binary",
        orientation="models_as_rows",
        description="MathArena — AIME 2025 (binary pass/fail)",
        url="https://matharena.ai/",
        tags=["math", "reasoning"],
        index_col="model_name",    ),
    # ── OSWorld (NEW) ────────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="osworld",
        csv_file="response_matrix.csv",
        registry_name="bench/osworld",
        response_type="continuous",
        orientation="models_as_rows",
        description="OSWorld — desktop automation tasks (scores)",
        url="https://os-world.github.io/",
        tags=["agentic", "desktop"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="osworld",
        csv_file="response_matrix_binary.csv",
        registry_name="bench/osworld_binary",
        response_type="binary",
        orientation="models_as_rows",
        description="OSWorld — desktop automation tasks (binary)",
        url="https://os-world.github.io/",
        tags=["agentic", "desktop"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── SWE-bench Full (NEW) ─────────────────────────────────────────
    BenchmarkSpec(
        benchmark="swebench_full",
        csv_file="response_matrix.csv",
        registry_name="bench/swebench_full",
        response_type="binary",
        orientation="models_as_rows",
        description="SWE-bench Full — software engineering (2,294 instances)",
        url="https://www.swebench.com/",
        tags=["coding", "agentic", "software-engineering"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── SWE-bench Multilingual (NEW) ─────────────────────────────────
    BenchmarkSpec(
        benchmark="swebench_multilingual",
        csv_file="response_matrix.csv",
        registry_name="bench/swebench_multilingual",
        response_type="binary",
        orientation="models_as_rows",
        description="SWE-bench Multilingual — multi-language SWE (301 instances)",
        url="https://github.com/multi-swe-bench/multi-swe-bench-env",
        tags=["coding", "agentic", "software-engineering", "multilingual"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="swebench_multilingual",
        csv_file="response_matrix_multi_swebench.csv",
        registry_name="bench/swebench_multi_all",
        response_type="binary",
        orientation="models_as_rows",
        description="SWE-bench Multi — all languages combined (2,132 instances)",
        url="https://github.com/multi-swe-bench/multi-swe-bench-env",
        tags=["coding", "agentic", "software-engineering", "multilingual"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── TheAgentCompany (NEW) ────────────────────────────────────────
    BenchmarkSpec(
        benchmark="theagentcompany",
        csv_file="response_matrix.csv",
        registry_name="bench/theagentcompany",
        response_type="continuous",
        orientation="models_as_rows",
        description="TheAgentCompany — enterprise task automation (scores)",
        url="https://github.com/TheAgentCompany/experiments",
        tags=["agentic", "enterprise"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="theagentcompany",
        csv_file="response_matrix_binary.csv",
        registry_name="bench/theagentcompany_binary",
        response_type="binary",
        orientation="models_as_rows",
        description="TheAgentCompany — enterprise task automation (binary)",
        url="https://github.com/TheAgentCompany/experiments",
        tags=["agentic", "enterprise"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── ToolBench (NEW) ──────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="toolbench",
        csv_file="response_matrix.csv",
        registry_name="bench/toolbench",
        response_type="binary",
        orientation="models_as_rows",
        description="StableToolBench — tool-use evaluation",
        url="https://github.com/THUNLP-MT/StableToolBench",
        tags=["agentic", "tool-use"],
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── WildBench (NEW) ──────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="wildbench",
        csv_file="response_matrix.csv",
        registry_name="bench/wildbench",
        response_type="continuous",
        orientation="models_as_rows",
        description="WildBench — open-ended LLM evaluation (1-10 scores)",
        url="https://huggingface.co/spaces/allenai/WildBench",
        tags=["nlp", "instruction-following"],
        scale_factor=0.1,
        content_file="task_metadata.csv",
        content_key="session_id",
        content_col="intent",
    ),
    # ── WorkArena (NEW) ──────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="workarena",
        csv_file="response_matrix.csv",
        registry_name="bench/workarena",
        response_type="continuous",
        orientation="models_as_rows",
        description="WorkArena — ServiceNow enterprise web tasks",
        url="https://github.com/ServiceNow/WorkArena",
        tags=["agentic", "enterprise", "web"],
        index_col="Agent",
    
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── ARC-AGI (NEW) ───────────────────────────────────────────────
    BenchmarkSpec(
        benchmark="arcagi",
        csv_file="response_matrix.csv",
        registry_name="bench/arcagi",
        response_type="binary",
        orientation="models_as_rows",
        description="ARC-AGI v1 — abstract reasoning (400 public eval tasks)",
        url="https://arcprize.org/",
        tags=["reasoning", "abstract"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="arcagi",
        csv_file="response_matrix_v2.csv",
        registry_name="bench/arcagi_v2",
        response_type="binary",
        orientation="models_as_rows",
        description="ARC-AGI v2 — abstract reasoning (120 tasks)",
        url="https://arcprize.org/",
        tags=["reasoning", "abstract"],
        index_col="model",
        content_file="item_content_v2.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── Humanity's Last Exam (NEW) ───────────────────────────────────
    BenchmarkSpec(
        benchmark="hle",
        csv_file="response_matrix.csv",
        registry_name="bench/hle",
        response_type="binary",
        orientation="models_as_rows",
        description="Humanity's Last Exam — expert-level questions (1,792 items)",
        url="https://lastexam.ai/",
        tags=["reasoning", "expert", "multi-domain"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── SWE-bench Java (NEW) ─────────────────────────────────────────
    BenchmarkSpec(
        benchmark="swebench_java",
        csv_file="response_matrix.csv",
        registry_name="bench/swebench_java",
        response_type="binary",
        orientation="models_as_rows",
        description="SWE-bench Java — Java issue resolution (170 instances)",
        url="https://github.com/multi-swe-bench",
        tags=["coding", "agentic", "software-engineering", "java"],
        index_col="model",
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── Global South: SIB-200 (205 languages, topic classification) ───
    BenchmarkSpec(
        benchmark="sib200",
        csv_file="response_matrix.csv",
        registry_name="bench/sib200",
        response_type="binary",
        orientation="items_as_rows",
        description="SIB-200 — Topic classification in 205 languages (GPT-4, GPT-3.5)",
        url="https://github.com/dadelani/sib-200",
        tags=["multilingual", "global-south", "topic-classification", "205-languages"],
        content_file="task_metadata.csv",
        content_key="item_id",
        content_col="text",
    ),
    # ── Global South: AfriMed-QA (Pan-African medical QA) ────────────
    BenchmarkSpec(
        benchmark="afrimedqa",
        csv_file="response_matrix.csv",
        registry_name="bench/afrimedqa",
        response_type="binary",
        orientation="items_as_rows",
        description="AfriMed-QA — Pan-African medical QA across 30 models, 20 specialties",
        url="https://github.com/intron-innovation/AfriMed-QA",
        tags=["global-south", "africa", "medical", "qa"],
        index_col="sample_id",
        content_file="task_metadata.csv",
        content_key="sample_id",
        content_col="question",
    ),
    # ── Global South: Bridging-the-Gap (African languages Winogrande) ─
    BenchmarkSpec(
        benchmark="bridging_gap",
        csv_file="response_matrix.csv",
        registry_name="bench/bridging_gap",
        response_type="binary",
        orientation="items_as_rows",
        description="Bridging-the-Gap — Winogrande in 12 languages (English + 11 African)",
        url="https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages",
        tags=["global-south", "africa", "multilingual", "commonsense-reasoning"],
    ),
    BenchmarkSpec(
        benchmark="bridging_gap",
        csv_file="response_matrix_averaged.csv",
        registry_name="bench/bridging_gap_continuous",
        response_type="continuous",
        orientation="items_as_rows",
        description="Bridging-the-Gap — Winogrande continuous scores (mean of 3 runs)",
        url="https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages",
        tags=["global-south", "africa", "multilingual", "commonsense-reasoning"],
    ),
    # ── Global South: La Leaderboard (Spanish/Catalan/Basque/Galician) ─
    BenchmarkSpec(
        benchmark="la_leaderboard",
        csv_file="response_matrix.csv",
        registry_name="bench/la_leaderboard",
        response_type="continuous",
        orientation="models_as_rows",
        description="La Leaderboard — 69 models × 108 tasks in Spanish, Catalan, Basque, Galician",
        url="https://huggingface.co/datasets/la-leaderboard/results",
        tags=["global-south", "iberian", "multilingual", "leaderboard"],
    ),
    # ── Global South: Portuguese LLM Leaderboard ─────────────────────
    BenchmarkSpec(
        benchmark="pt_leaderboard",
        csv_file="response_matrix.csv",
        registry_name="bench/pt_leaderboard",
        response_type="continuous",
        orientation="models_as_rows",
        description="Portuguese LLM Leaderboard — 1,148 models × 10 Portuguese NLP tasks",
        url="https://huggingface.co/datasets/eduagarcia-temp/llm_pt_leaderboard_raw_results",
        tags=["global-south", "portuguese", "leaderboard"],
    ),
    # ── Global South: Korean LLM Leaderboard ─────────────────────────
    BenchmarkSpec(
        benchmark="ko_leaderboard",
        csv_file="response_matrix.csv",
        registry_name="bench/ko_leaderboard",
        response_type="continuous",
        orientation="models_as_rows",
        description="Open Ko-LLM Leaderboard — 1,159 models × 9 Korean benchmark tasks",
        url="https://huggingface.co/datasets/open-ko-llm-leaderboard/results",
        tags=["east-asia", "korean", "leaderboard"],
    ),
    # ── Global South: Thai LLM Leaderboard ───────────────────────────
    BenchmarkSpec(
        benchmark="thai_leaderboard",
        csv_file="response_matrix.csv",
        registry_name="bench/thai_leaderboard",
        response_type="continuous",
        orientation="models_as_rows",
        description="ThaiLLM Leaderboard — 72 models × 19 Thai benchmark tasks",
        url="https://huggingface.co/datasets/ThaiLLM-Leaderboard/results",
        tags=["southeast-asia", "thai", "leaderboard"],
    ),
    # ── Global South: KMMLU (Korean, human baseline) ─────────────────
    BenchmarkSpec(
        benchmark="kmmlu",
        csv_file="response_matrix.csv",
        registry_name="bench/kmmlu",
        response_type="continuous",
        orientation="items_as_rows",
        description="KMMLU — Korean MMLU, 35,030 items with per-item human accuracy baseline",
        url="https://huggingface.co/datasets/HAERAE-HUB/KMMLU",
        tags=["east-asia", "korean", "knowledge", "human-baseline"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    # ── Global South: HELM African MMLU+Winogrande ───────────────────
    BenchmarkSpec(
        benchmark="helm_multilingual",
        csv_file="response_matrix_afr.csv",
        registry_name="bench/helm_african",
        response_type="binary",
        orientation="items_as_rows",
        description="HELM African MMLU+Winogrande — 23 models × 33,880 items in 11 African languages",
        url="https://crfm.stanford.edu/helm/mmlu-winogrande-afr/latest/",
        tags=["global-south", "africa", "multilingual", "knowledge", "commonsense-reasoning"],
        index_col="question_id",
    ),
    # ── Global South: HELM ThaiExam ──────────────────────────────────
    BenchmarkSpec(
        benchmark="helm_multilingual",
        csv_file="response_matrix_thaiexam.csv",
        registry_name="bench/helm_thaiexam",
        response_type="binary",
        orientation="items_as_rows",
        description="HELM ThaiExam — 42 models × 565 Thai exam items (A-Level, IC, ONET, TGAT, TPAT1)",
        url="https://crfm.stanford.edu/helm/thaiexam/latest/",
        tags=["southeast-asia", "thai", "knowledge", "exams"],
        index_col="question_id",
    ),
    # ── Global South: HELM CLEVA (Chinese) ───────────────────────────
    BenchmarkSpec(
        benchmark="helm_multilingual",
        csv_file="response_matrix_cleva.csv",
        registry_name="bench/helm_cleva",
        response_type="binary",
        orientation="items_as_rows",
        description="HELM CLEVA — 4 models × 5,828 items across 21 Chinese NLP tasks",
        url="https://crfm.stanford.edu/helm/cleva/latest/",
        tags=["east-asia", "chinese", "multilingual", "knowledge"],
        index_col="question_id",
    ),
    # ── Global South: OALL Arabic EXAMS ──────────────────────────────
    BenchmarkSpec(
        benchmark="culturaleval",
        csv_file="response_matrix_oall_arabic_exams.csv",
        registry_name="bench/oall_arabic_exams",
        response_type="binary",
        orientation="models_as_rows",
        description="OALL Arabic EXAMS — 144 models × 537 Arabic exam items",
        url="https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard",
        tags=["global-south", "mena", "arabic", "exams"],
        index_col="model",
    ),
    # ── Global South: OALL Arabic MMLU ───────────────────────────────
    BenchmarkSpec(
        benchmark="culturaleval",
        csv_file="response_matrix_oall_arabic_mmlu.csv",
        registry_name="bench/oall_arabic_mmlu",
        response_type="binary",
        orientation="models_as_rows",
        description="OALL Arabic MMLU — 142 models × 14,042 Arabic-translated MMLU items",
        url="https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard",
        tags=["global-south", "mena", "arabic", "knowledge"],
        index_col="model",
    ),
    # ── Global South: MasakhaNER v2 (sentence-level) ─────────────────
    BenchmarkSpec(
        benchmark="afrieval",
        csv_file="response_matrix_masakhaner_v2_sentence.csv",
        registry_name="bench/masakhaner_v2",
        response_type="binary",
        orientation="items_as_rows",
        description="MasakhaNER v2 — 7 models × 27,559 sentences in 19 African languages (NER correctness)",
        url="https://github.com/masakhane-io/masakhane-ner",
        tags=["global-south", "africa", "multilingual", "ner"],
    ),
]


# ---------------------------------------------------------------------------
# Special parsers
# ---------------------------------------------------------------------------


def _parse_range_value(val):
    """Parse a value that may be a range like '0.75-0.88' → midpoint 0.815."""
    if isinstance(val, str):
        val = val.strip()
        # Match range pattern: number-number (but not negative numbers)
        m = re.match(r"^(\d+\.?\d*)\s*-\s*(\d+\.?\d*)$", val)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2.0
    return pd.to_numeric(val, errors="coerce")


def _parse_fraction_value(val):
    """Parse a fraction like '2/3' → 0.667, 'X' → 0.0, numeric → float."""
    if isinstance(val, str):
        val = val.strip()
        if val.upper() == "X":
            return 0.0
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", val)
        if m:
            numer, denom = int(m.group(1)), int(m.group(2))
            return numer / denom if denom > 0 else float("nan")
    return pd.to_numeric(val, errors="coerce")


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def csv_to_payload(spec: BenchmarkSpec) -> dict | None:
    """Convert a single CSV file to a .pt payload dict.

    Returns None if the CSV file doesn't exist.
    """
    csv_path = SRC_DIR / f"{spec.benchmark}_data" / "processed" / spec.csv_file
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping")
        return None

    df = pd.read_csv(csv_path)

    # ── Identify index column ──────────────────────────────────────────
    if isinstance(spec.index_col, str):
        idx_col_pos = df.columns.get_loc(spec.index_col)
    else:
        idx_col_pos = spec.index_col
    ids = df.iloc[:, idx_col_pos].astype(str).tolist()

    # ── ClineBench special handling ────────────────────────────────────
    if spec.subject_cols is not None:
        # For clinebench: extract specific columns as subject data
        data_cols = []
        for col in spec.subject_cols:
            if col in df.columns:
                data_cols.append(col)
        df_data = df[data_cols].copy()
        # Parse range values (e.g., "0.75-0.88")
        for col in df_data.columns:
            df_data[col] = df_data[col].apply(_parse_range_value)
        df_data = df_data.astype(float)
        col_names = data_cols
    else:
        # ── Strip metadata columns + index column ──────────────────────
        idx_col_name = df.columns[idx_col_pos]
        skip = set(spec.metadata_cols) | {idx_col_name}
        data_cols = [c for c in df.columns if c not in skip]

        if spec.parse_fractions:
            # Parse fraction values (e.g., "2/3" → 0.667, "X" → 0.0)
            df_data = df[data_cols].map(_parse_fraction_value).astype(float)
        else:
            df_data = df[data_cols].apply(pd.to_numeric, errors="coerce")
        col_names = data_cols

    # ── Build tensor ───────────────────────────────────────────────────
    data = torch.tensor(df_data.values, dtype=torch.float32)

    # ── Scale if needed ────────────────────────────────────────────────
    if spec.scale_factor != 1.0:
        # Only scale non-NaN values
        data = data * spec.scale_factor

    # ── Transpose if items-as-rows ─────────────────────────────────────
    if spec.orientation == "items_as_rows":
        # rows are items, columns are subjects (models)
        data = data.T
        subject_ids = list(col_names)
        item_ids = ids
    else:
        # rows are subjects (models), columns are items
        subject_ids = ids
        item_ids = list(col_names)

    # ── Load item content from metadata if available ──────────────────
    item_contents = [""] * len(item_ids)
    if spec.content_file is not None:
        meta_path = SRC_DIR / f"{spec.benchmark}_data" / "processed" / spec.content_file
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            meta_df[spec.content_key] = meta_df[spec.content_key].astype(str)
            content_map = dict(zip(meta_df[spec.content_key], meta_df[spec.content_col].fillna("")))
            item_contents = [str(content_map.get(iid, "")) for iid in item_ids]
            n_matched = sum(1 for c in item_contents if c)
            print(f"  Item content: {n_matched}/{len(item_ids)} items matched from {spec.content_file}")
        else:
            print(f"  WARNING: content file {meta_path} not found, using empty strings")

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "subject_metadata": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global SRC_DIR
    parser = argparse.ArgumentParser(description="Migrate bench data to .pt format")
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Convert only, don't upload to HuggingFace Hub",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Process only this benchmark (e.g., 'cybench')",
    )
    parser.add_argument(
        "--src-dir", type=Path, default=None,
        help="Override source data directory (default: data/ in repo root)",
    )
    args = parser.parse_args()

    if args.src_dir is not None:
        SRC_DIR = args.src_dir

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    specs = SPECS
    if args.benchmark:
        specs = [s for s in specs if s.benchmark == args.benchmark]
        if not specs:
            print(f"No specs found for benchmark '{args.benchmark}'")
            return

    payloads: dict[str, dict] = {}

    print("=" * 70)
    print(f"Processing {len(specs)} dataset specs...")
    print("=" * 70)

    for spec in specs:
        print(f"\n--- {spec.registry_name} ({spec.csv_file}) ---")
        payload = csv_to_payload(spec)
        if payload is None:
            continue

        n_sub, n_items = payload["data"].shape
        nan_count = torch.isnan(payload["data"]).sum().item()
        total = n_sub * n_items
        nan_pct = nan_count / total if total > 0 else 0
        print(f"  {n_sub} subjects x {n_items} items, "
              f"{nan_pct:.1%} missing ({nan_count}/{total})")

        # Validate values are in expected range
        valid = payload["data"][~torch.isnan(payload["data"])]
        if len(valid) > 0:
            vmin, vmax = valid.min().item(), valid.max().item()
            print(f"  Value range: [{vmin:.4f}, {vmax:.4f}]")
            if vmax > 1.01 or vmin < -0.01:
                print(f"  WARNING: Values outside [0, 1] range!")

        # Save .pt file
        filename = f"{spec.registry_name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)
        payloads[spec.registry_name] = payload

    # Upload if requested
    if not args.no_upload:
        try:
            from huggingface_hub import HfApi, upload_file

            print("\n" + "=" * 70)
            print("Uploading to HuggingFace Hub...")
            print("=" * 70)
            api = HfApi()

            # Create the repo if it doesn't exist yet
            try:
                api.repo_info(DST_REPO, repo_type="dataset")
                print(f"Destination repo {DST_REPO} already exists.")
            except Exception:
                print(f"Creating dataset repo {DST_REPO}...")
                api.create_repo(DST_REPO, repo_type="dataset", private=False)

            for name in sorted(payloads):
                filename = f"{name}.pt"
                local_path = TMP_DIR / filename
                print(f"  Uploading {filename}...")
                upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=filename,
                    repo_id=DST_REPO,
                    repo_type="dataset",
                )
            print(f"\nUploaded {len(payloads)} files to {DST_REPO}")
        except ImportError:
            print("\nhuggingface_hub not installed. Skipping upload.")
        except Exception as e:
            print(f"\nUpload failed: {e}")

    # Print registry dimensions for bench.py
    print("\n" + "=" * 70)
    print("Dataset dimensions for bench.py registry:")
    print("=" * 70)
    for name in sorted(payloads):
        p = payloads[name]
        n_sub, n_items = p["data"].shape
        spec = next(s for s in specs if s.registry_name == name)
        print(f'    DatasetInfo(name="{name}", family="bench", '
              f'response_type="{spec.response_type}", '
              f"n_subjects={n_sub}, n_items={n_items}, "
              f'description="{spec.description}"),')

    print(f"\nTotal: {len(payloads)} datasets")
    print(f"Files saved to: {TMP_DIR}")


# --- Newly curated benchmarks (2026-03-21) ---

SPECS_NEW = [
    BenchmarkSpec(
        benchmark="rewardbench",
        csv_file="response_matrix.csv",
        registry_name="bench/rewardbench",
        response_type="binary",
        orientation="models_as_rows",
        description="RewardBench — reward model evaluation (chosen vs rejected)",
        url="https://github.com/allenai/reward-bench",
        tags=["preference", "reward-model"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="lawbench",
        csv_file="response_matrix.csv",
        registry_name="bench/lawbench",
        response_type="binary",
        orientation="models_as_rows",
        description="LawBench — Chinese legal reasoning (51 models, 20 tasks, zero-shot)",
        url="https://github.com/open-compass/LawBench",
        tags=["legal", "chinese", "domain-specific"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="ultrafeedback",
        csv_file="response_matrix.csv",
        registry_name="bench/ultrafeedback",
        response_type="continuous",
        orientation="models_as_rows",
        description="UltraFeedback — multi-model instruction following (overall scores, 64K prompts)",
        url="https://github.com/OpenBMB/UltraFeedback",
        tags=["preference", "instruction-following"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
    BenchmarkSpec(
        benchmark="financebench",
        csv_file="response_matrix.csv",
        registry_name="bench/financebench",
        response_type="binary",
        orientation="models_as_rows",
        description="FinanceBench — financial QA over SEC filings (16 configs, 150 items)",
        url="https://github.com/patronus-ai/financebench",
        tags=["finance", "domain-specific", "qa"],
        content_file="item_content.csv",
        content_key="item_id",
        content_col="content",
    ),
]

SPECS.extend(SPECS_NEW)

if __name__ == "__main__":
    main()
