#!/usr/bin/env python3
"""
Build a per-task response matrix for WebArena benchmark.

Aggregates per-task pass/fail (1/0) results from multiple model/agent
configurations into a single matrix:
  rows = tasks (0..811), columns = model/agent configurations.

Data sources:
  1. Original WebArena paper v2 execution traces (merged_log.txt files)
     - GPT-3.5-turbo-16k + {CoT, Direct} x {with/without UA hint}
     - GPT-4-0613 + CoT
     - text-bison-001 + CoT
  2. Original WebArena paper v1 execution traces
     - GPT-3.5-turbo + {Direct, Reasoning}
  3. GPT-4o (WebArena team leaderboard traces)
  4. STeP agent + GPT-4-turbo (from agent-evals repo)
  5. GUI-API Hybrid Agent + GPT-4o (from HuggingFace)
  6. ColorBrowserAgent (from GitHub)
  7. WebOperator + GPT-4o (from GitHub)

Output:
  - webarena_response_matrix.csv
  - webarena_response_matrix_summary.csv (aggregate stats per model)
"""

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUT_DIR = _BENCHMARK_DIR / "processed"

NUM_TASKS = 812  # Tasks 0..811


def parse_merged_log(filepath, model_name):
    """
    Parse a merged_log.txt file to extract per-task PASS/FAIL results.
    Returns dict: {task_id: 1 (pass) or 0 (fail)}
    """
    results = {}
    with open(filepath, "r") as f:
        for line in f:
            if "[Result]" in line:
                # Extract PASS/FAIL
                is_pass = "(PASS)" in line
                # Extract task ID from config file path
                # Patterns: config_files/123.json or /tmp.../123.json
                match = re.search(r"/(\d+)\.json", line)
                if match:
                    task_id = int(match.group(1))
                    results[task_id] = 1 if is_pass else 0
    return results


def load_v2_traces():
    """Load all v2 merged_log.txt files from the original WebArena traces."""
    v2_dir = RAW_DIR / "102023_release_v2"
    models = {}

    mapping = {
        "gpt35_16k_cot_na": "gpt-3.5-turbo-16k_CoT_UAhint",
        "gpt35_16k_cot": "gpt-3.5-turbo-16k_CoT",
        "gpt35_16k_direct_na": "gpt-3.5-turbo-16k_Direct_UAhint",
        "gpt35_16k_direct": "gpt-3.5-turbo-16k_Direct",
        "gpt4_8k_cot": "gpt-4-0613_CoT",
        "text_bison_001_cot": "text-bison-001_CoT",
    }

    for fname, model_name in mapping.items():
        fpath = v2_dir / f"{fname}_merged_log.txt"
        if fpath.exists():
            models[model_name] = parse_merged_log(fpath, model_name)
            print(f"  v2 {model_name}: {sum(models[model_name].values())} pass / "
                  f"{len(models[model_name])} total")

    return models


def load_v1_traces():
    """Load v1 merged_log.txt files."""
    v1_dir = RAW_DIR / "072023_release_v1"
    models = {}

    mapping = {
        "gpt3.5direct": "gpt-3.5-turbo_Direct_v1",
        "gpt3.5dreasoning": "gpt-3.5-turbo_CoT_v1",
    }

    for fname, model_name in mapping.items():
        fpath = v1_dir / f"{fname}_merged_log.txt"
        if fpath.exists():
            models[model_name] = parse_merged_log(fpath, model_name)
            print(f"  v1 {model_name}: {sum(models[model_name].values())} pass / "
                  f"{len(models[model_name])} total")

    return models


def load_gpt4o_traces():
    """Load GPT-4o merged_log from leaderboard traces."""
    fpath = RAW_DIR / "gpt4o_merged_log.txt"
    if not fpath.exists():
        print("  GPT-4o merged_log not found, skipping")
        return {}

    results = parse_merged_log(fpath, "gpt-4o_CoT")
    print(f"  gpt-4o_CoT: {sum(results.values())} pass / {len(results)} total")
    return {"gpt-4o_CoT": results}


def load_step_agent():
    """Load STeP agent + GPT-4-turbo per-task results from agent-evals."""
    base_dir = RAW_DIR / "agent-evals" / "webagents-step" / "data" / "webarena" / "eval"
    step_dir = base_dir / "step_full" / "gpt-4-turbo-2024-04-09" / "run1"

    if not step_dir.exists():
        print("  STeP agent data not found, skipping")
        return {}

    results = {}
    # Walk through timestamped directories and collect JSON files
    for subdir in step_dir.iterdir():
        if subdir.is_dir():
            for jfile in subdir.glob("*.json"):
                if jfile.stem.isdigit():
                    task_id = int(jfile.stem)
                    try:
                        with open(jfile) as f:
                            data = json.load(f)
                        # Get reward from last trajectory entry
                        if "trajectory" in data and data["trajectory"]:
                            last = data["trajectory"][-1]
                            reward = last.get("reward", 0.0)
                            results[task_id] = 1 if reward == 1.0 else 0
                    except (json.JSONDecodeError, KeyError):
                        pass

    model_name = "STeP_gpt-4-turbo"
    print(f"  {model_name}: {sum(results.values())} pass / {len(results)} total")
    return {model_name: results}


def load_gui_api_hybrid():
    """Load GUI-API Hybrid Agent (Beyond Browsing) per-task results."""
    fpath = RAW_DIR / "gui_api_hybrid_output.jsonl"
    if not fpath.exists():
        print("  GUI-API hybrid data not found, skipping")
        return {}

    results = {}
    with open(fpath) as f:
        for line in f:
            data = json.loads(line)
            task_id = data.get("task_id")
            correct = data.get("correct", False)
            if task_id is not None:
                results[int(task_id)] = 1 if correct else 0

    model_name = "GUI-API-Hybrid_gpt-4o"
    print(f"  {model_name}: {sum(results.values())} pass / {len(results)} total")
    return {model_name: results}


def load_colorbrowseragent():
    """Load ColorBrowserAgent per-task results."""
    fpath = RAW_DIR / "colorbrowseragent" / "per_task_results.json"
    if not fpath.exists():
        print("  ColorBrowserAgent data not found, skipping")
        return {}

    with open(fpath) as f:
        raw = json.load(f)

    results = {}
    for task_id_str, reward in raw.items():
        if reward is not None:
            results[int(task_id_str)] = 1 if reward == 1.0 else 0

    model_name = "ColorBrowserAgent"
    print(f"  {model_name}: {sum(results.values())} pass / {len(results)} total")
    return {model_name: results}


def load_weboperator():
    """Load WebOperator + GPT-4o per-task results."""
    fpath = RAW_DIR / "weboperator_summary.json"
    if not fpath.exists():
        print("  WebOperator data not found, skipping")
        return {}

    with open(fpath) as f:
        raw = json.load(f)

    results = {}
    for task_id_str, info in raw.items():
        task_id = int(task_id_str)
        score = info.get("score", 0.0)
        results[task_id] = 1 if score == 1.0 else 0

    model_name = "WebOperator_gpt-4o"
    print(f"  {model_name}: {sum(results.values())} pass / {len(results)} total")
    return {model_name: results}


def load_gbox_claude():
    """Load GBOX/Claude Code per-task results from log files."""
    log_dir = RAW_DIR / "gbox_webarena" / "gbox" / "trajectories" / "log_files"
    if not log_dir.exists():
        print("  GBOX/Claude Code data not found, skipping")
        return {}

    results = {}
    for logfile in log_dir.glob("task_*_*.log"):
        # Extract task ID from filename: task_123_20251025_182851.log
        parts = logfile.stem.split("_")
        if len(parts) >= 2:
            try:
                task_id = int(parts[1])
            except ValueError:
                continue

            with open(logfile) as f:
                content = f.read()

            if "(PASS)" in content:
                results[task_id] = 1
            elif "(FAIL)" in content:
                results[task_id] = 0
            # If neither, task might have errored - skip

    model_name = "ClaudeCode_GBOX"
    print(f"  {model_name}: {sum(results.values())} pass / {len(results)} total")
    return {model_name: results}


def build_matrix():
    """Build the complete response matrix."""
    print("Loading per-task results from all sources...")
    print()

    all_models = {}

    print("[1] Loading v2 execution traces...")
    all_models.update(load_v2_traces())
    print()

    print("[2] Loading v1 execution traces...")
    all_models.update(load_v1_traces())
    print()

    print("[3] Loading GPT-4o traces...")
    all_models.update(load_gpt4o_traces())
    print()

    print("[4] Loading STeP agent results...")
    all_models.update(load_step_agent())
    print()

    print("[5] Loading GUI-API Hybrid Agent results...")
    all_models.update(load_gui_api_hybrid())
    print()

    print("[6] Loading ColorBrowserAgent results...")
    all_models.update(load_colorbrowseragent())
    print()

    print("[7] Loading WebOperator results...")
    all_models.update(load_weboperator())
    print()

    print("[8] Loading GBOX/Claude Code results...")
    all_models.update(load_gbox_claude())
    print()

    # Build the matrix DataFrame
    print(f"Building response matrix with {len(all_models)} model configs...")
    task_ids = list(range(NUM_TASKS))

    matrix_data = {"task_id": task_ids}
    for model_name, results in sorted(all_models.items()):
        col = []
        for tid in task_ids:
            col.append(results.get(tid, np.nan))
        matrix_data[model_name] = col

    df = pd.DataFrame(matrix_data)
    df = df.set_index("task_id")

    # Save response matrix
    out_path = OUT_DIR / "webarena_response_matrix.csv"
    df.to_csv(out_path)
    print(f"Saved response matrix to {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Tasks: {df.shape[0]}, Models: {df.shape[1]}")
    print()

    # Generate summary stats
    summary_rows = []
    for col in df.columns:
        valid = df[col].dropna()
        n_tasks = len(valid)
        n_pass = int(valid.sum())
        n_fail = n_tasks - n_pass
        success_rate = n_pass / n_tasks * 100 if n_tasks > 0 else 0
        coverage = n_tasks / NUM_TASKS * 100

        summary_rows.append({
            "model": col,
            "n_tasks_evaluated": n_tasks,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "success_rate_pct": round(success_rate, 2),
            "coverage_pct": round(coverage, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("success_rate_pct", ascending=False)
    summary_path = OUT_DIR / "webarena_response_matrix_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print()

    # Print summary table
    print("=" * 80)
    print("WebArena Response Matrix Summary")
    print("=" * 80)
    print(f"{'Model':<40} {'Tasks':>6} {'Pass':>6} {'Fail':>6} {'Rate%':>8} {'Cover%':>8}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<40} {row['n_tasks_evaluated']:>6} "
              f"{row['n_pass']:>6} {row['n_fail']:>6} "
              f"{row['success_rate_pct']:>8.2f} {row['coverage_pct']:>8.2f}")
    print("-" * 80)

    # Compute pairwise coverage (tasks where both models were evaluated)
    print()
    print("Pairwise coverage (# tasks with data for both models):")
    models = list(df.columns)
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            both_valid = df[[models[i], models[j]]].dropna().shape[0]
            if both_valid < NUM_TASKS:
                print(f"  {models[i]} x {models[j]}: {both_valid}/{NUM_TASKS}")

    # NaN statistics
    total_cells = df.shape[0] * df.shape[1]
    total_nan = df.isna().sum().sum()
    print(f"\nTotal matrix cells: {total_cells}")
    print(f"Total NaN (missing): {int(total_nan)} ({total_nan/total_cells*100:.1f}%)")
    print(f"Total valid: {total_cells - int(total_nan)} ({(total_cells-total_nan)/total_cells*100:.1f}%)")

    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_matrix()
