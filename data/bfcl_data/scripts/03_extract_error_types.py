"""
Extract per-task error types from BFCL score files.

For each (model, task) pair, records either "pass" or the specific error type
from the score file. This gives rubric-like diagnostic information beyond
binary pass/fail.

Outputs:
1. error_type_matrix.csv — models × tasks, cells are error type strings or "pass"
2. error_type_summary.csv — per-model error type distribution counts
3. error_type_taxonomy.csv — full error type taxonomy with counts and categories

Error types are organized into a coarse hierarchy:
  - ast_decoder: model output couldn't be parsed as a function call
  - value_error: wrong parameter value (string, list, dict, etc.)
  - type_error: wrong parameter type
  - *_checker: correct parse but wrong function/count/match
  - multi_turn: multi-step interaction failures
  - irrelevance/relevance: model should/shouldn't have called a function
  - executable: execution-based checking failures
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SCORE_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "score")
RESULT_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "result")
OUTPUT_DIR = str(_BENCHMARK_DIR / "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coarse error categories for grouping
COARSE_CATEGORY = {
    "ast_decoder": "parse_error",
    "executable_decoder": "parse_error",
    "value_error": "wrong_value",
    "type_error": "wrong_type",
    "simple_function_checker": "wrong_function",
    "multiple_function_checker": "wrong_function",
    "parallel_function_checker_no_order": "wrong_function",
    "simple_exec_checker": "wrong_function",
    "executable_checker": "execution_error",
    "executable_checker_rest": "execution_error",
    "multi_turn": "multi_turn_error",
    "irrelevance_error": "relevance_error",
    "relevance_error": "relevance_error",
}


def get_coarse_category(error_type):
    """Map fine-grained error type to coarse category."""
    prefix = error_type.split(":")[0]
    return COARSE_CATEGORY.get(prefix, "other")


def load_score_file_with_errors(path):
    """Load score file, returning (header, dict of task_id -> error_type)."""
    with open(path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    header = lines[0]
    failed = {}
    for entry in lines[1:]:
        task_id = entry.get("prompt", {}).get("id")
        if task_id is None:
            task_id = str(entry.get("id", ""))
        task_id = str(task_id)

        # Get error type
        et = entry.get("error_type", None)
        if et is None:
            err = entry.get("error", {})
            if isinstance(err, dict):
                et = err.get("error_type", "unknown")
            else:
                et = "unknown"

        failed[task_id] = et
    return header, failed


def get_all_task_ids_from_result(result_dir, model_name, category):
    """Get all task IDs for a category from the result file."""
    result_file = os.path.join(
        result_dir, model_name, f"BFCL_v3_{category}_result.json")
    if not os.path.exists(result_file):
        return None
    ids = []
    with open(result_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ids.append(str(entry.get("id", "")))
            except json.JSONDecodeError:
                continue
    return ids


def main():
    # Discover models and categories
    model_dirs = sorted([
        d for d in os.listdir(SCORE_DIR)
        if os.path.isdir(os.path.join(SCORE_DIR, d))
    ])
    print(f"Found {len(model_dirs)} models")

    sample_model = model_dirs[0]
    score_files = sorted([
        f for f in os.listdir(os.path.join(SCORE_DIR, sample_model))
        if f.endswith("_score.json")
    ])
    categories = [
        f.replace("BFCL_v3_", "").replace("_score.json", "")
        for f in score_files
    ]
    print(f"Found {len(categories)} categories")

    # Build task ID lists (same logic as 01_build_response_matrix.py)
    ref_model = "gpt-4o-2024-11-20-FC"
    if ref_model not in model_dirs:
        ref_model = model_dirs[0]

    category_task_ids = {}
    total_tasks = 0
    for cat in categories:
        ids = get_all_task_ids_from_result(RESULT_DIR, ref_model, cat)
        if ids is None:
            for alt_model in model_dirs:
                ids = get_all_task_ids_from_result(RESULT_DIR, alt_model, cat)
                if ids is not None:
                    break
        if ids is not None:
            global_ids = [f"{cat}::{tid}" for tid in ids]
            category_task_ids[cat] = (ids, global_ids)
            total_tasks += len(ids)

    all_global_ids = []
    for cat in categories:
        if cat in category_task_ids:
            _, gids = category_task_ids[cat]
            all_global_ids.extend(gids)

    print(f"Total tasks: {total_tasks}")
    print(f"\nBuilding error type matrix...")

    # Build the error type matrix (string matrix)
    global_id_to_col = {gid: i for i, gid in enumerate(all_global_ids)}
    # Initialize with empty strings (will be "pass" or error type)
    error_matrix = np.full(
        (len(model_dirs), len(all_global_ids)), "", dtype=object)

    all_error_types = Counter()
    per_model_errors = {}

    for model_idx, model_name in enumerate(model_dirs):
        model_score_dir = os.path.join(SCORE_DIR, model_name)
        model_errors = Counter()

        for cat in categories:
            if cat not in category_task_ids:
                continue
            local_ids, global_ids = category_task_ids[cat]

            score_path = os.path.join(
                model_score_dir, f"BFCL_v3_{cat}_score.json")
            if not os.path.exists(score_path):
                continue

            _, failed = load_score_file_with_errors(score_path)

            model_result_ids = get_all_task_ids_from_result(
                RESULT_DIR, model_name, cat)

            if model_result_ids is not None:
                for tid in model_result_ids:
                    gid = f"{cat}::{tid}"
                    if gid in global_id_to_col:
                        col = global_id_to_col[gid]
                        if tid in failed:
                            et = failed[tid]
                            error_matrix[model_idx, col] = et
                            model_errors[et] += 1
                            all_error_types[et] += 1
                        else:
                            error_matrix[model_idx, col] = "pass"
                            model_errors["pass"] += 1
            else:
                for tid, gid in zip(local_ids, global_ids):
                    col = global_id_to_col[gid]
                    if tid in failed:
                        et = failed[tid]
                        error_matrix[model_idx, col] = et
                        model_errors[et] += 1
                        all_error_types[et] += 1
                    else:
                        error_matrix[model_idx, col] = "pass"
                        model_errors["pass"] += 1

        per_model_errors[model_name] = model_errors

        if (model_idx + 1) % 20 == 0:
            print(f"  Processed {model_idx + 1}/{len(model_dirs)} models")

    print(f"  Processed {len(model_dirs)}/{len(model_dirs)} models")

    # 1. Save error type matrix
    df = pd.DataFrame(error_matrix, index=model_dirs, columns=all_global_ids)
    df.index.name = "model"
    csv_path = os.path.join(OUTPUT_DIR, "error_type_matrix.csv")
    df.to_csv(csv_path)
    print(f"\nSaved {csv_path}")
    print(f"  Shape: {df.shape}")

    # Count cells
    n_pass = np.sum(error_matrix == "pass")
    n_fail = np.sum((error_matrix != "pass") & (error_matrix != ""))
    n_empty = np.sum(error_matrix == "")
    print(f"  Pass: {n_pass}, Fail: {n_fail}, Empty: {n_empty}")

    # 2. Save error type taxonomy
    print("\n=== Error Type Taxonomy ===")
    taxonomy = []
    for et, count in all_error_types.most_common():
        coarse = get_coarse_category(et)
        taxonomy.append({
            "error_type": et,
            "coarse_category": coarse,
            "count": count,
        })
        print(f"  {count:6d}  [{coarse:20s}]  {et}")

    taxonomy_df = pd.DataFrame(taxonomy)
    taxonomy_df.to_csv(
        os.path.join(OUTPUT_DIR, "error_type_taxonomy.csv"), index=False)
    print(f"\nSaved error_type_taxonomy.csv ({len(taxonomy)} error types)")

    # 3. Save per-model error type summary
    all_et_names = sorted(all_error_types.keys())
    summary_rows = []
    for model in model_dirs:
        row = {"model": model}
        errors = per_model_errors.get(model, {})
        row["n_pass"] = errors.get("pass", 0)
        row["n_fail"] = sum(v for k, v in errors.items() if k != "pass")
        row["n_total"] = row["n_pass"] + row["n_fail"]
        row["accuracy"] = (row["n_pass"] / row["n_total"]
                           if row["n_total"] > 0 else 0)
        for et in all_et_names:
            row[f"n_{et}"] = errors.get(et, 0)
        # Coarse categories
        coarse_counts = Counter()
        for et, count in errors.items():
            if et != "pass":
                coarse_counts[get_coarse_category(et)] += count
        for cc in sorted(set(COARSE_CATEGORY.values())):
            row[f"coarse_{cc}"] = coarse_counts.get(cc, 0)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("accuracy", ascending=False)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "error_type_summary.csv"), index=False)
    print(f"Saved error_type_summary.csv")

    # 4. Also save a coarse error type matrix
    coarse_matrix = np.full(
        (len(model_dirs), len(all_global_ids)), "", dtype=object)
    for i in range(error_matrix.shape[0]):
        for j in range(error_matrix.shape[1]):
            val = error_matrix[i, j]
            if val == "pass":
                coarse_matrix[i, j] = "pass"
            elif val == "":
                coarse_matrix[i, j] = ""
            else:
                coarse_matrix[i, j] = get_coarse_category(val)

    coarse_df = pd.DataFrame(
        coarse_matrix, index=model_dirs, columns=all_global_ids)
    coarse_df.index.name = "model"
    coarse_path = os.path.join(OUTPUT_DIR, "error_type_coarse_matrix.csv")
    coarse_df.to_csv(coarse_path)
    print(f"Saved {coarse_path}")

    # Print coarse distribution
    print("\n=== Coarse Error Category Distribution ===")
    coarse_counts = Counter()
    for val in coarse_matrix.flat:
        if val and val != "pass":
            coarse_counts[val] += 1
    for cc, count in coarse_counts.most_common():
        pct = 100 * count / n_fail
        print(f"  {cc:20s}  {count:6d}  ({pct:.1f}% of failures)")


if __name__ == "__main__":
    main()
