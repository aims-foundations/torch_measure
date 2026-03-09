#!/usr/bin/env python3
"""
Build a response matrix for ScienceAgentBench (SAB).

This script constructs a tasks x models response matrix CSV from all
publicly available per-task per-model evaluation data for ScienceAgentBench.

Data sources:
  1. HuggingFace `osunlp/ScienceAgentBench` -- task metadata (102 tasks)
  2. ScienceAgentBench paper (NeurIPS 2024 / ICLR 2025) -- aggregate results per model
  3. HAL Leaderboard (hal.cs.princeton.edu/scienceagentbench) -- additional models

Per-task per-model results are NOT publicly released in the ScienceAgentBench
repository or on HuggingFace. The evaluation log JSONL files (containing
per-instance success_rate, valid_program, codebert_score) are produced only
by running the full Docker-based evaluation pipeline locally. The HAL
leaderboard provides aggregate scores and encrypted traces (requiring
hal-decrypt).

This script therefore:
  - Downloads the full 102-task metadata from HuggingFace
  - Embeds the aggregate results from the paper (Table 3) and HAL leaderboard
  - Constructs a response matrix at the AGGREGATE level (per-model, not per-task)
  - Produces task_metadata.csv with full task-level details
  - Produces response_matrix.csv with the best available data

Output directory: ../processed/
"""

import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "processed"
RAW_DIR = SCRIPT_DIR.parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Task metadata from HuggingFace
# ---------------------------------------------------------------------------
def load_task_metadata():
    """Load the 102 tasks from HuggingFace or cached JSON."""
    cache_path = RAW_DIR / "task_metadata_full.json"

    if cache_path.exists():
        with open(cache_path, "r") as f:
            tasks = json.load(f)
        if len(tasks) == 102:
            print(f"[INFO] Loaded {len(tasks)} tasks from cache: {cache_path}")
            return tasks

    # Download from HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
        tasks = []
        for ex in ds:
            tasks.append({
                "instance_id": ex["instance_id"],
                "domain": ex["domain"],
                "subtask_categories": ex["subtask_categories"],
                "github_name": ex["github_name"],
                "gold_program_name": ex["gold_program_name"],
                "eval_script_name": ex["eval_script_name"],
                "output_fname": ex["output_fname"],
                "task_inst": ex["task_inst"][:300],
            })
        with open(cache_path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"[INFO] Downloaded {len(tasks)} tasks from HuggingFace")
        return tasks
    except ImportError:
        print("[ERROR] `datasets` library not installed. pip install datasets")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Aggregate results from the paper (Table 3) and HAL leaderboard
# ---------------------------------------------------------------------------
# Paper results: best-of-3 runs per model x framework configuration
# Metrics: SR = Success Rate (%), VER = Valid Execution Rate (%),
#           CBS = CodeBERTScore (0-100 scale), Cost = avg USD per task
#
# Column naming: {model}_{framework}_{knowledge}
# knowledge: "no_knowledge" or "with_knowledge"

PAPER_RESULTS = [
    # (model, framework, knowledge, SR, VER, CBS, Cost)
    # Direct Prompting - Without Knowledge
    ("Llama-3.1-70B", "Direct", "no_knowledge", 5.9, 29.4, 81.5, 0.001),
    ("Llama-3.1-405B", "Direct", "no_knowledge", 3.9, 35.3, 79.4, 0.010),
    ("Mistral-Large-2", "Direct", "no_knowledge", 13.7, 47.1, 83.2, 0.009),
    ("GPT-4o", "Direct", "no_knowledge", 11.8, 52.9, 82.6, 0.011),
    ("Claude-3.5-Sonnet", "Direct", "no_knowledge", 17.7, 51.0, 83.6, 0.017),
    ("OpenAI-o1-preview", "Direct", "no_knowledge", 34.3, 70.6, 87.1, 0.221),
    # Direct Prompting - With Knowledge
    ("Llama-3.1-70B", "Direct", "with_knowledge", 4.9, 27.5, 82.1, 0.001),
    ("Llama-3.1-405B", "Direct", "with_knowledge", 2.9, 25.5, 81.3, 0.011),
    ("Mistral-Large-2", "Direct", "with_knowledge", 16.7, 39.2, 84.7, 0.009),
    ("GPT-4o", "Direct", "with_knowledge", 10.8, 41.2, 83.8, 0.012),
    ("Claude-3.5-Sonnet", "Direct", "with_knowledge", 21.6, 41.2, 85.4, 0.017),
    ("OpenAI-o1-preview", "Direct", "with_knowledge", 31.4, 63.7, 87.4, 0.236),
    # OpenHands CodeAct - Without Knowledge
    ("Llama-3.1-70B", "OpenHands", "no_knowledge", 6.9, 30.4, 63.5, 0.145),
    ("Llama-3.1-405B", "OpenHands", "no_knowledge", 5.9, 52.0, 65.8, 0.383),
    ("Mistral-Large-2", "OpenHands", "no_knowledge", 9.8, 53.9, 72.5, 0.513),
    ("GPT-4o", "OpenHands", "no_knowledge", 19.6, 78.4, 83.1, 0.803),
    ("Claude-3.5-Sonnet", "OpenHands", "no_knowledge", 21.6, 87.3, 83.6, 0.958),
    # OpenHands CodeAct - With Knowledge
    ("Llama-3.1-70B", "OpenHands", "with_knowledge", 2.9, 25.5, 65.7, 0.252),
    ("Llama-3.1-405B", "OpenHands", "with_knowledge", 8.8, 58.8, 71.4, 0.740),
    ("Mistral-Large-2", "OpenHands", "with_knowledge", 13.7, 50.0, 78.8, 0.759),
    ("GPT-4o", "OpenHands", "with_knowledge", 27.5, 73.5, 86.3, 1.094),
    ("Claude-3.5-Sonnet", "OpenHands", "with_knowledge", 24.5, 88.2, 85.1, 0.900),
    # Self-Debug - Without Knowledge
    ("Llama-3.1-70B", "SelfDebug", "no_knowledge", 13.7, 80.4, 82.7, 0.007),
    ("Llama-3.1-405B", "SelfDebug", "no_knowledge", 14.7, 78.4, 82.9, 0.047),
    ("Mistral-Large-2", "SelfDebug", "no_knowledge", 23.5, 83.3, 85.1, 0.034),
    ("GPT-4o", "SelfDebug", "no_knowledge", 22.6, 83.3, 84.4, 0.047),
    ("Claude-3.5-Sonnet", "SelfDebug", "no_knowledge", 32.4, 92.2, 86.4, 0.057),
    ("OpenAI-o1-preview", "SelfDebug", "no_knowledge", 42.2, 92.2, 88.4, 0.636),
    # Self-Debug - With Knowledge
    ("Llama-3.1-70B", "SelfDebug", "with_knowledge", 16.7, 73.5, 83.4, 0.008),
    ("Llama-3.1-405B", "SelfDebug", "with_knowledge", 13.7, 79.4, 83.6, 0.055),
    ("Mistral-Large-2", "SelfDebug", "with_knowledge", 27.5, 78.4, 86.8, 0.036),
    ("GPT-4o", "SelfDebug", "with_knowledge", 23.5, 71.6, 85.6, 0.046),
    ("Claude-3.5-Sonnet", "SelfDebug", "with_knowledge", 34.3, 86.3, 87.1, 0.061),
    ("OpenAI-o1-preview", "SelfDebug", "with_knowledge", 41.2, 91.2, 88.9, 0.713),
]

# HAL Leaderboard results (single-run accuracy, not best-of-3)
# These use different evaluation semantics than the paper
HAL_RESULTS = [
    # (agent_name, model, accuracy_pct, cost_usd)
    ("SAB-SelfDebug", "o3-Medium-Apr2025", 33.33, 11.69),
    ("SAB-SelfDebug", "Claude-Sonnet-4.5-High-Sep2025", 30.39, 7.47),
    ("SAB-SelfDebug", "Claude-3.7-Sonnet-High-Feb2025", 30.39, 11.74),
    ("SAB-SelfDebug", "GPT-5-Medium-Aug2025", 30.39, 18.26),
    ("SAB-SelfDebug", "Claude-Sonnet-4.5-Sep2025", 29.41, 7.39),
    ("SAB-SelfDebug", "o4-mini-Low-Apr2025", 27.45, 3.95),
    ("SAB-SelfDebug", "o4-mini-High-Apr2025", 27.45, 11.18),
    ("SAB-SelfDebug", "Claude-Opus-4.1-Aug2025", 27.45, 33.37),
    ("SAB-SelfDebug", "Claude-Opus-4.1-High-Aug2025", 26.47, 33.75),
    ("SAB-SelfDebug", "GPT-4.1-Apr2025", 24.51, 7.42),
    ("SAB-SelfDebug", "Claude-Haiku-4.5-High-Oct2025", 23.53, 3.41),
    ("SAB-SelfDebug", "DeepSeek-R1-Jan2025", 23.53, 18.24),
    ("SAB-SelfDebug", "Claude-3.7-Sonnet-Feb2025", 22.55, 7.12),
    ("HAL-Generalist", "o4-mini-High-Apr2025", 21.57, 76.30),
    ("HAL-Generalist", "o4-mini-Low-Apr2025", 19.61, 77.32),
    ("SAB-SelfDebug", "Claude-Haiku-4.5-Oct2025", 18.63, 2.66),
    ("HAL-Generalist", "Claude-3.7-Sonnet-High-Feb2025", 17.65, 48.28),
    ("SAB-SelfDebug", "DeepSeek-V3-Mar2025", 15.69, 2.09),
    ("SAB-SelfDebug", "Gemini-2.0-Flash-Feb2025", 12.75, 0.19),
    ("HAL-Generalist", "Claude-3.7-Sonnet-Feb2025", 10.78, 41.22),
    ("HAL-Generalist", "o3-Medium-Apr2025", 9.80, 31.08),
    ("HAL-Generalist", "GPT-4.1-Apr2025", 6.86, 68.95),
    ("HAL-Generalist", "DeepSeek-V3-Mar2025", 0.98, 55.73),
]


# ---------------------------------------------------------------------------
# 3. Build task metadata CSV
# ---------------------------------------------------------------------------
def build_task_metadata_csv(tasks):
    """Write task_metadata.csv with per-task details."""
    out_path = OUTPUT_DIR / "task_metadata.csv"
    fieldnames = [
        "instance_id", "domain", "subtask_categories",
        "github_name", "gold_program_name", "eval_script_name",
        "output_fname", "task_inst_preview"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(tasks, key=lambda x: x["instance_id"]):
            writer.writerow({
                "instance_id": t["instance_id"],
                "domain": t["domain"],
                "subtask_categories": t["subtask_categories"],
                "github_name": t["github_name"],
                "gold_program_name": t["gold_program_name"],
                "eval_script_name": t["eval_script_name"],
                "output_fname": t["output_fname"],
                "task_inst_preview": t.get("task_inst", "")[:200],
            })
    print(f"[INFO] Wrote task metadata: {out_path} ({len(tasks)} tasks)")
    return out_path


# ---------------------------------------------------------------------------
# 4. Build response matrix CSV
# ---------------------------------------------------------------------------
def build_response_matrix(tasks):
    """
    Build the response matrix CSV.

    Since per-task per-model evaluation logs are NOT publicly released,
    we construct the matrix at two levels:

    A) PAPER_RESULTS: Aggregate-level response matrix
       - Rows = 102 tasks, Columns = model configurations
       - Cell values = aggregate SR (%) from paper, applied uniformly
       - This represents the expected per-task success probability

    B) Additionally we save a separate aggregate_results.csv with all
       per-configuration metrics.

    The response_matrix.csv uses:
       - Rows: instance_id (1..102)
       - Columns: model configuration labels
       - Values: "NA" (per-task results not publicly available)
       - A note column explains data availability
    """

    # --- Build model configuration names ---
    # Paper models: use best configuration (Self-Debug, no knowledge) as
    # primary, plus all configurations
    model_configs = OrderedDict()
    for model, framework, knowledge, sr, ver, cbs, cost in PAPER_RESULTS:
        col_name = f"{model}__{framework}__{knowledge}"
        model_configs[col_name] = {
            "source": "paper",
            "model": model,
            "framework": framework,
            "knowledge": knowledge,
            "success_rate_pct": sr,
            "valid_execution_rate_pct": ver,
            "codebert_score": cbs,
            "cost_usd": cost,
        }

    # HAL models
    for agent, model, acc, cost in HAL_RESULTS:
        col_name = f"{model}__{agent}__HAL"
        model_configs[col_name] = {
            "source": "HAL_leaderboard",
            "model": model,
            "framework": agent,
            "knowledge": "unknown",
            "success_rate_pct": acc,
            "valid_execution_rate_pct": None,
            "codebert_score": None,
            "cost_usd": cost,
        }

    all_columns = list(model_configs.keys())

    # --- Write response matrix ---
    out_path = OUTPUT_DIR / "response_matrix.csv"
    fieldnames = ["instance_id", "domain", "subtask_categories"] + all_columns

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(tasks, key=lambda x: x["instance_id"]):
            row = {
                "instance_id": t["instance_id"],
                "domain": t["domain"],
                "subtask_categories": t["subtask_categories"],
            }
            # Per-task results are not publicly available
            for col in all_columns:
                row[col] = "NA"
            writer.writerow(row)

    print(f"[INFO] Wrote response matrix: {out_path}")
    print(f"       Dimensions: {len(tasks)} tasks x {len(all_columns)} model configs")
    print(f"       Fill rate: 0% (per-task results not publicly released)")
    print(f"       Cells marked 'NA' -- see aggregate_results.csv for scores")

    # --- Write aggregate results ---
    agg_path = OUTPUT_DIR / "aggregate_results.csv"
    agg_fields = [
        "model_config", "source", "model", "framework", "knowledge",
        "success_rate_pct", "valid_execution_rate_pct",
        "codebert_score", "cost_usd"
    ]
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for col_name, info in model_configs.items():
            writer.writerow({
                "model_config": col_name,
                "source": info["source"],
                "model": info["model"],
                "framework": info["framework"],
                "knowledge": info["knowledge"],
                "success_rate_pct": info["success_rate_pct"],
                "valid_execution_rate_pct": info.get("valid_execution_rate_pct", ""),
                "codebert_score": info.get("codebert_score", ""),
                "cost_usd": info["cost_usd"],
            })
    print(f"[INFO] Wrote aggregate results: {agg_path}")
    print(f"       {len(model_configs)} model configurations total")

    return out_path, agg_path, model_configs


# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------
def print_summary(tasks, model_configs):
    """Print summary statistics."""
    from collections import Counter

    print("\n" + "=" * 70)
    print("SCIENCEAGENTBENCH RESPONSE MATRIX -- SUMMARY REPORT")
    print("=" * 70)

    # Task statistics
    print(f"\n--- Task Statistics ---")
    print(f"Total tasks: {len(tasks)}")
    domains = Counter(t["domain"] for t in tasks)
    print(f"Domains ({len(domains)}):")
    for d, cnt in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {d}: {cnt} tasks ({100*cnt/len(tasks):.1f}%)")

    cats = Counter(t["subtask_categories"] for t in tasks)
    print(f"\nSubtask categories ({len(cats)} unique):")
    for c, cnt in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {c}: {cnt}")
    if len(cats) > 10:
        print(f"  ... and {len(cats) - 10} more")

    repos = Counter(t["github_name"] for t in tasks)
    print(f"\nSource repositories: {len(repos)} unique")
    for r, cnt in sorted(repos.items(), key=lambda x: -x[1])[:10]:
        print(f"  {r}: {cnt}")

    # Model statistics
    print(f"\n--- Model/Agent Statistics ---")
    paper_configs = [k for k, v in model_configs.items() if v["source"] == "paper"]
    hal_configs = [k for k, v in model_configs.items() if v["source"] == "HAL_leaderboard"]
    print(f"Total model configurations: {len(model_configs)}")
    print(f"  From paper (Table 3): {len(paper_configs)}")
    print(f"  From HAL leaderboard: {len(hal_configs)}")

    # Unique models
    paper_models = set(v["model"] for v in model_configs.values() if v["source"] == "paper")
    hal_models = set(v["model"] for v in model_configs.values() if v["source"] == "HAL_leaderboard")
    print(f"\nUnique base models:")
    print(f"  Paper: {sorted(paper_models)}")
    print(f"  HAL: {sorted(hal_models)}")

    # Frameworks
    frameworks = set(v["framework"] for v in model_configs.values())
    print(f"\nAgent frameworks: {sorted(frameworks)}")

    # Score ranges
    srs = [v["success_rate_pct"] for v in model_configs.values()]
    print(f"\n--- Score Ranges ---")
    print(f"Success Rate: {min(srs):.1f}% - {max(srs):.1f}%")
    print(f"  Mean: {sum(srs)/len(srs):.1f}%")

    vers = [v["valid_execution_rate_pct"] for v in model_configs.values()
            if v["valid_execution_rate_pct"] is not None]
    if vers:
        print(f"Valid Execution Rate: {min(vers):.1f}% - {max(vers):.1f}%")
        print(f"  Mean: {sum(vers)/len(vers):.1f}%")

    cbss = [v["codebert_score"] for v in model_configs.values()
            if v["codebert_score"] is not None]
    if cbss:
        print(f"CodeBERTScore: {min(cbss):.1f} - {max(cbss):.1f}")

    costs = [v["cost_usd"] for v in model_configs.values()]
    print(f"Cost per task: ${min(costs):.3f} - ${max(costs):.2f}")

    # Data availability
    print(f"\n--- Data Availability ---")
    print(f"Response matrix dimensions: {len(tasks)} tasks x {len(model_configs)} configs")
    print(f"Fill rate: 0% (per-task per-model results NOT publicly released)")
    print(f"Score type: Success Rate (%) -- binary pass/fail per task")
    print(f"  Additional metrics: VER (%), CBS (0-100), Cost (USD)")
    print()
    print("IMPORTANT: ScienceAgentBench does NOT publicly release per-task")
    print("per-model evaluation logs (JSONL files). The evaluation requires:")
    print("  1. Running the agent to produce code for each of 102 tasks")
    print("  2. Running Docker-based evaluation of generated code")
    print("  3. Computing metrics from evaluation logs")
    print()
    print("To obtain per-task results, you must run the full evaluation")
    print("pipeline from: https://github.com/OSU-NLP-Group/ScienceAgentBench")
    print("or decrypt HAL leaderboard traces via: hal-decrypt")
    print()
    print("The aggregate_results.csv provides all publicly available scores")
    print("at the configuration level (57 configurations from paper + HAL).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Building ScienceAgentBench Response Matrix")
    print("=" * 70)

    # Step 1: Load task metadata
    tasks = load_task_metadata()

    # Step 2: Build task metadata CSV
    meta_path = build_task_metadata_csv(tasks)

    # Step 3: Build response matrix
    matrix_path, agg_path, model_configs = build_response_matrix(tasks)

    # Step 4: Print summary
    print_summary(tasks, model_configs)

    print(f"\nOutput files:")
    print(f"  {meta_path}")
    print(f"  {matrix_path}")
    print(f"  {agg_path}")


if __name__ == "__main__":
    main()
