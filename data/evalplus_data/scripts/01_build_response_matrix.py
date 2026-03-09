#!/usr/bin/env python3
"""
Build EvalPlus (HumanEval+ / MBPP+) response matrices from code samples.

This script:
1. Discovers per-model code sample directories (from evalplus GitHub releases)
2. Converts directory-format samples to JSONL for evalplus
3. Runs evalplus evaluate to get per-task pass/fail
4. Aggregates results into response matrices (models x tasks)
5. Saves CSV outputs

Data sources:
- Code samples: https://github.com/evalplus/evalplus/releases (v0.1.0, v0.2.0)
- Leaderboard: https://evalplus.github.io/leaderboard.html (results.json)
- HumanEval+: 164 tasks, MBPP+: 399 tasks
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Paths
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
HUMANEVAL_SAMPLES = RAW_DIR / "humaneval_samples" / "extracted"
MBPP_SAMPLES = RAW_DIR / "mbpp_samples" / "extracted"
EVAL_RESULTS_DIR = RAW_DIR / "eval_results"
LEADERBOARD_FILE = RAW_DIR / "leaderboard_results.json"


def find_model_dirs(base_dir, task_prefix, min_tasks=100):
    """Find all model directories containing task subdirectories.

    Args:
        base_dir: Root directory to search
        task_prefix: Expected prefix for task directories (e.g., 'HumanEval')
        min_tasks: Minimum number of task directories to qualify

    Returns:
        dict mapping model_name -> directory_path
    """
    result = {}
    for root, dirs, files in os.walk(base_dir):
        task_dirs = [d for d in dirs if d.startswith(task_prefix)]
        if len(task_dirs) >= min_tasks:
            model_name = os.path.basename(root)
            # Clean model name: remove _temp_0.0 or _temp_0 suffix
            clean_name = re.sub(r"_temp_0(\.0)?$", "", model_name)
            result[clean_name] = root
    return result


def convert_dir_to_jsonl(model_dir, task_prefix, output_path):
    """Convert a directory-format sample set to JSONL.

    Each task directory contains a 0.py file with the full solution.

    Args:
        model_dir: Path to model's sample directory
        task_prefix: 'HumanEval' or 'Mbpp'
        output_path: Where to write the JSONL file
    """
    entries = []
    for task_dir_name in sorted(os.listdir(model_dir)):
        if not task_dir_name.startswith(task_prefix):
            continue
        # Convert directory name to task_id: HumanEval_0 -> HumanEval/0
        task_id = task_dir_name.replace("_", "/", 1)
        py_file = os.path.join(model_dir, task_dir_name, "0.py")
        if os.path.exists(py_file):
            with open(py_file, "r") as f:
                code = f.read()
            entries.append({"task_id": task_id, "solution": code})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return len(entries)


def run_evalplus(dataset, jsonl_path, parallel=8):
    """Run evalplus evaluate and return the result JSON path.

    Args:
        dataset: 'humaneval' or 'mbpp'
        jsonl_path: Path to the JSONL samples file
        parallel: Number of parallel workers

    Returns:
        Path to the eval results JSON file, or None on failure
    """
    result_path = str(jsonl_path).replace(".jsonl", "_eval_results.json")

    # Skip if already evaluated
    if os.path.exists(result_path):
        print(f"  [SKIP] Already evaluated: {result_path}")
        return result_path

    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        dataset,
        "--samples", str(jsonl_path),
        "--parallel", str(parallel),
        "--i_just_wanna_run",
        "--test_details",
    ]

    print(f"  [RUN] {' '.join(cmd[-6:])}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per model
        )
        if proc.returncode != 0:
            print(f"  [ERROR] evalplus failed: {proc.stderr[-500:]}")
            return None
        # Parse pass@1 from output
        for line in proc.stdout.split("\n"):
            if "pass@1" in line:
                print(f"  {line.strip()}")
    except subprocess.TimeoutExpired:
        print("  [ERROR] Evaluation timed out (600s)")
        return None

    if os.path.exists(result_path):
        return result_path
    return None


def extract_per_task_results(result_json_path, benchmark_type="plus"):
    """Extract per-task pass/fail from evalplus result JSON.

    Args:
        result_json_path: Path to eval_results.json
        benchmark_type: 'base' for original tests, 'plus' for EvalPlus tests

    Returns:
        dict mapping task_id -> 1 (pass) or 0 (fail)
    """
    with open(result_json_path, "r") as f:
        data = json.load(f)

    status_key = f"{benchmark_type}_status"
    task_results = {}

    for task_id, results_list in data.get("eval", {}).items():
        # For greedy decoding (temp=0), there's typically 1 sample per task
        if results_list:
            status = results_list[0].get(status_key, "fail")
            task_results[task_id] = 1 if status == "pass" else 0

    return task_results


def build_response_matrix(all_results, task_ids):
    """Build a models x tasks response matrix DataFrame.

    Args:
        all_results: dict mapping model_name -> {task_id: 0/1}
        task_ids: sorted list of all task IDs

    Returns:
        pandas DataFrame with models as rows, tasks as columns
    """
    rows = {}
    for model_name, task_results in sorted(all_results.items()):
        rows[model_name] = {tid: task_results.get(tid, None) for tid in task_ids}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[sorted(df.columns, key=lambda x: int(x.split("/")[1]))]
    df.index.name = "model"
    return df


def build_model_summary(all_results, leaderboard_data=None):
    """Build model summary with pass rates and metadata.

    Args:
        all_results: dict of {benchmark: {model: {task: 0/1}}}
        leaderboard_data: Optional leaderboard JSON for additional metadata

    Returns:
        pandas DataFrame with model summary
    """
    rows = []
    for model_name in sorted(
        set().union(
            *[set(results.keys()) for results in all_results.values()]
        )
    ):
        row = {"model": model_name}
        for benchmark, results in all_results.items():
            if model_name in results:
                task_results = results[model_name]
                n_pass = sum(v for v in task_results.values() if v is not None)
                n_total = sum(1 for v in task_results.values() if v is not None)
                row[f"{benchmark}_pass"] = n_pass
                row[f"{benchmark}_total"] = n_total
                row[f"{benchmark}_pass_rate"] = (
                    n_pass / n_total * 100 if n_total > 0 else None
                )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("model")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build EvalPlus response matrices"
    )
    parser.add_argument(
        "--parallel", type=int, default=8,
        help="Parallel workers for evalplus evaluation"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation, only build matrices from existing results"
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["humaneval_base", "humaneval_plus", "mbpp_base", "mbpp_plus"],
        help="Which benchmarks to include"
    )
    args = parser.parse_args()

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ========================================================
    # Step 1: Discover model directories
    # ========================================================
    print("=" * 70)
    print("Step 1: Discovering model sample directories")
    print("=" * 70)

    he_models = find_model_dirs(HUMANEVAL_SAMPLES, "HumanEval", min_tasks=150)
    mbpp_models = find_model_dirs(MBPP_SAMPLES, "Mbpp", min_tasks=350)

    print(f"\nFound {len(he_models)} HumanEval+ models:")
    for name in sorted(he_models.keys()):
        print(f"  - {name}")

    print(f"\nFound {len(mbpp_models)} MBPP+ models:")
    for name in sorted(mbpp_models.keys()):
        print(f"  - {name}")

    # ========================================================
    # Step 2: Convert to JSONL and evaluate
    # ========================================================
    print("\n" + "=" * 70)
    print("Step 2: Converting samples and running evaluations")
    print("=" * 70)

    # Process HumanEval+
    he_results = {}
    if not args.skip_eval:
        print("\n--- HumanEval+ Evaluation ---")
        for model_name, model_dir in sorted(he_models.items()):
            print(f"\n[{model_name}]")
            jsonl_path = EVAL_RESULTS_DIR / f"humaneval_{model_name}.jsonl"

            n = convert_dir_to_jsonl(model_dir, "HumanEval", str(jsonl_path))
            print(f"  Converted {n} tasks to JSONL")

            result_path = run_evalplus("humaneval", jsonl_path, args.parallel)
            if result_path:
                he_results[model_name] = result_path

        # Process MBPP+
        mbpp_result_paths = {}
        print("\n--- MBPP+ Evaluation ---")
        for model_name, model_dir in sorted(mbpp_models.items()):
            print(f"\n[{model_name}]")
            jsonl_path = EVAL_RESULTS_DIR / f"mbpp_{model_name}.jsonl"

            n = convert_dir_to_jsonl(model_dir, "Mbpp", str(jsonl_path))
            print(f"  Converted {n} tasks to JSONL")

            result_path = run_evalplus("mbpp", jsonl_path, args.parallel)
            if result_path:
                mbpp_result_paths[model_name] = result_path
    else:
        # Load from existing results
        print("\n[SKIP] Skipping evaluation, loading existing results...")
        for f in EVAL_RESULTS_DIR.glob("humaneval_*_eval_results.json"):
            model_name = f.stem.replace("humaneval_", "").replace(
                "_eval_results", ""
            )
            he_results[model_name] = str(f)
        mbpp_result_paths = {}
        for f in EVAL_RESULTS_DIR.glob("mbpp_*_eval_results.json"):
            model_name = f.stem.replace("mbpp_", "").replace(
                "_eval_results", ""
            )
            mbpp_result_paths[model_name] = str(f)

    # ========================================================
    # Step 3: Extract per-task results and build matrices
    # ========================================================
    print("\n" + "=" * 70)
    print("Step 3: Building response matrices")
    print("=" * 70)

    all_benchmark_results = {}

    # HumanEval base and plus
    if he_results:
        he_base_all = {}
        he_plus_all = {}
        for model_name, result_path in sorted(he_results.items()):
            he_base_all[model_name] = extract_per_task_results(
                result_path, "base"
            )
            he_plus_all[model_name] = extract_per_task_results(
                result_path, "plus"
            )
        all_benchmark_results["humaneval_base"] = he_base_all
        all_benchmark_results["humaneval_plus"] = he_plus_all

        # Get all task IDs
        all_he_tasks = sorted(
            set().union(*[set(r.keys()) for r in he_base_all.values()]),
            key=lambda x: int(x.split("/")[1]),
        )

        # Build HumanEval matrices
        for btype in ["base", "plus"]:
            key = f"humaneval_{btype}"
            results_dict = all_benchmark_results[key]
            df = build_response_matrix(results_dict, all_he_tasks)
            out_path = PROCESSED_DIR / f"response_matrix_humaneval_{btype}.csv"
            df.to_csv(out_path)
            n_models, n_tasks = df.shape
            n_ones = df.sum().sum()
            n_total = df.notna().sum().sum()
            print(
                f"\n  {key}: {n_models} models x {n_tasks} tasks, "
                f"overall pass rate: {n_ones/n_total*100:.1f}%"
            )
            print(f"  Saved to: {out_path}")

    # MBPP base and plus
    if mbpp_result_paths:
        mbpp_base_all = {}
        mbpp_plus_all = {}
        for model_name, result_path in sorted(mbpp_result_paths.items()):
            mbpp_base_all[model_name] = extract_per_task_results(
                result_path, "base"
            )
            mbpp_plus_all[model_name] = extract_per_task_results(
                result_path, "plus"
            )
        all_benchmark_results["mbpp_base"] = mbpp_base_all
        all_benchmark_results["mbpp_plus"] = mbpp_plus_all

        all_mbpp_tasks = sorted(
            set().union(*[set(r.keys()) for r in mbpp_base_all.values()]),
            key=lambda x: int(x.split("/")[1]),
        )

        for btype in ["base", "plus"]:
            key = f"mbpp_{btype}"
            results_dict = all_benchmark_results[key]
            df = build_response_matrix(results_dict, all_mbpp_tasks)
            out_path = PROCESSED_DIR / f"response_matrix_mbpp_{btype}.csv"
            df.to_csv(out_path)
            n_models, n_tasks = df.shape
            n_ones = df.sum().sum()
            n_total = df.notna().sum().sum()
            print(
                f"\n  {key}: {n_models} models x {n_tasks} tasks, "
                f"overall pass rate: {n_ones/n_total*100:.1f}%"
            )
            print(f"  Saved to: {out_path}")

    # ========================================================
    # Step 4: Build combined response matrix (humaneval+ primary)
    # ========================================================
    print("\n" + "=" * 70)
    print("Step 4: Building combined response matrix")
    print("=" * 70)

    # Primary matrix: HumanEval+ (the main EvalPlus benchmark)
    if "humaneval_plus" in all_benchmark_results:
        he_plus_results = all_benchmark_results["humaneval_plus"]
        all_he_tasks = sorted(
            set().union(*[set(r.keys()) for r in he_plus_results.values()]),
            key=lambda x: int(x.split("/")[1]),
        )
        df_combined = build_response_matrix(he_plus_results, all_he_tasks)
        combined_path = PROCESSED_DIR / "response_matrix.csv"
        df_combined.to_csv(combined_path)
        print(f"\n  Combined (HumanEval+) matrix: {df_combined.shape}")
        print(f"  Saved to: {combined_path}")

    # ========================================================
    # Step 5: Build model summary
    # ========================================================
    print("\n" + "=" * 70)
    print("Step 5: Building model summary")
    print("=" * 70)

    df_summary = build_model_summary(all_benchmark_results)

    # Add leaderboard data if available
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            lb_data = json.load(f)
        lb_rows = []
        for model_name, info in lb_data.items():
            lb_rows.append({
                "model": model_name,
                "leaderboard_humaneval": info["pass@1"].get("humaneval"),
                "leaderboard_humaneval_plus": info["pass@1"].get("humaneval+"),
                "leaderboard_mbpp": info["pass@1"].get("mbpp"),
                "leaderboard_mbpp_plus": info["pass@1"].get("mbpp+"),
                "size_b": info.get("size"),
                "prompted": info.get("prompted"),
            })
        df_lb = pd.DataFrame(lb_rows).set_index("model")
        # Merge: evaluated models + full leaderboard
        df_summary = df_summary.join(df_lb, how="outer")

    summary_path = PROCESSED_DIR / "model_summary.csv"
    df_summary.to_csv(summary_path)
    print(f"\n  Model summary: {df_summary.shape[0]} models")
    print(f"  Saved to: {summary_path}")

    # Print summary stats
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    for benchmark in ["humaneval_plus", "humaneval_base", "mbpp_plus", "mbpp_base"]:
        col = f"{benchmark}_pass_rate"
        if col in df_summary.columns:
            valid = df_summary[col].dropna()
            if len(valid) > 0:
                print(f"\n  {benchmark}:")
                print(f"    Models evaluated: {len(valid)}")
                print(f"    Pass rate range: {valid.min():.1f}% - {valid.max():.1f}%")
                print(f"    Mean pass rate: {valid.mean():.1f}%")
                print(f"    Median pass rate: {valid.median():.1f}%")

    if "leaderboard_humaneval_plus" in df_summary.columns:
        valid_lb = df_summary["leaderboard_humaneval_plus"].dropna()
        print(f"\n  Leaderboard (HumanEval+): {len(valid_lb)} models")
        print(f"    Range: {valid_lb.min():.1f}% - {valid_lb.max():.1f}%")
        print(f"    Mean: {valid_lb.mean():.1f}%, Median: {valid_lb.median():.1f}%")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
