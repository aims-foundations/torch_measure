"""
Build a per-task binary response matrix and extract per-task error types from
BFCL score files.

The BFCL-Result repo stores score files as NDJSON:
  - Line 0: {"accuracy": ..., "correct_count": ..., "total_count": ...}
  - Lines 1+: failed task entries with {"id": ..., "valid": false, ...}

Strategy: for each model x category, load the score file, get the total count
from the header, collect failed IDs, and mark all other IDs as pass (1).

Uses the 2024-12-29 snapshot (most complete: 92 models, 22 categories).

Outputs:
  - processed/response_matrix.csv      (models x tasks, binary 0/1)
  - processed/category_summary.csv     (per-category accuracy stats)
  - processed/model_summary.csv        (per-model accuracy stats)
  - processed/error_type_matrix.csv    (models x tasks, error type strings or "pass")
  - processed/error_type_summary.csv   (per-model error type distribution counts)
  - processed/error_type_taxonomy.csv  (full error type taxonomy with counts and categories)
  - processed/error_type_coarse_matrix.csv (models x tasks, coarse error categories)
"""

INFO = {
    'description': """Build a per-task binary response matrix and extract per-task error types from BFCL score files""",
    'testing_condition': '',
    'paper_url': 'https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html',
    'data_source_url': 'https://github.com/HuanzhiMao/BFCL-Result',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'Apache-2.0',
    'citation': '@misc{patil2025bfcl,\n  title={BFCL},\n  year={2025},\n  howpublished={\\url{https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html}},\n}',
    'tags': ['coding'],
}


import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
SCORE_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "score")
RESULT_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "result")
OUTPUT_DIR = str(_BENCHMARK_DIR / "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download():
    """Download raw data from external sources."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clone_dir = RAW_DIR / "BFCL-Result"
    if not clone_dir.exists():
        print("Cloning BFCL-Result repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/HuanzhiMao/BFCL-Result.git", str(clone_dir)],
            check=True,
        )
    else:
        print("BFCL-Result repo already exists, pulling latest...")
        subprocess.run(
            ["git", "-C", str(clone_dir), "pull", "--ff-only"],
            check=False,
        )

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


def load_score_file(path):
    """Load an NDJSON score file. Returns (header_dict, set_of_failed_task_ids).

    Score file format:
      Line 0: {"accuracy": ..., "correct_count": ..., "total_count": ...}
      Lines 1+: failed entries where the actual task ID is in entry["prompt"]["id"]
                (NOT entry["id"] which is a numeric line index)
    """
    with open(path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    header = lines[0]
    failed_ids = set()
    for entry in lines[1:]:
        # The task ID is in prompt.id (string like "exec_simple_92")
        # entry["id"] is just a numeric index
        task_id = entry.get("prompt", {}).get("id")
        if task_id is None:
            # Fallback: some entries might have id directly as string
            task_id = str(entry.get("id", ""))
        failed_ids.add(str(task_id))
    return header, failed_ids


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
    """
    Get all task IDs for a category from the result file.
    Result files have one JSON per line with an "id" field.
    """
    # Map category name to result file name
    result_file = os.path.join(result_dir, model_name,
                               f"BFCL_v3_{category}_result.json")
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


def get_coarse_category(error_type):
    """Map fine-grained error type to coarse category."""
    prefix = error_type.split(":")[0]
    return COARSE_CATEGORY.get(prefix, "other")


def build_response_matrix(model_dirs, categories, category_task_ids,
                          all_global_ids, global_id_to_col):
    """Build the binary response matrix (original 01 workflow)."""
    print(f"\nBuilding {len(model_dirs)} x {len(all_global_ids)} response matrix...")

    matrix = np.full((len(model_dirs), len(all_global_ids)), np.nan)

    for model_idx, model_name in enumerate(model_dirs):
        model_score_dir = os.path.join(SCORE_DIR, model_name)

        for cat in categories:
            if cat not in category_task_ids:
                continue
            local_ids, global_ids = category_task_ids[cat]

            score_path = os.path.join(model_score_dir,
                                      f"BFCL_v3_{cat}_score.json")
            if not os.path.exists(score_path):
                # Model doesn't have this category
                continue

            header, failed_ids = load_score_file(score_path)

            # Use model's own result file for task IDs if available
            model_result_ids = get_all_task_ids_from_result(
                RESULT_DIR, model_name, cat)

            if model_result_ids is not None:
                for tid in model_result_ids:
                    gid = f"{cat}::{tid}"
                    if gid in global_id_to_col:
                        col = global_id_to_col[gid]
                        matrix[model_idx, col] = 0 if tid in failed_ids else 1
            else:
                # Fall back to reference model's task IDs
                for tid, gid in zip(local_ids, global_ids):
                    col = global_id_to_col[gid]
                    matrix[model_idx, col] = 0 if tid in failed_ids else 1

        if (model_idx + 1) % 10 == 0:
            print(f"  Processed {model_idx + 1}/{len(model_dirs)} models")

    print(f"  Processed {len(model_dirs)}/{len(model_dirs)} models")

    # Create DataFrame
    df = pd.DataFrame(matrix, index=model_dirs, columns=all_global_ids)
    df.index.name = "model"

    # Save full matrix
    csv_path = os.path.join(OUTPUT_DIR, "response_matrix.csv")
    df.to_csv(csv_path)
    print(f"\nSaved {csv_path}")
    print(f"  Shape: {df.shape}")

    # Report coverage
    n_filled = np.count_nonzero(~np.isnan(matrix))
    n_total = matrix.size
    print(f"  Filled cells: {n_filled}/{n_total} ({100*n_filled/n_total:.1f}%)")
    print(f"  Pass rate: {np.nanmean(matrix):.3f}")

    # Per-category accuracy summary
    print("\n=== Per-Category Summary ===")
    cat_stats = []
    for cat in categories:
        if cat not in category_task_ids:
            continue
        _, gids = category_task_ids[cat]
        cols = [global_id_to_col[gid] for gid in gids]
        cat_matrix = matrix[:, cols]
        n_items = len(gids)
        mean_acc = np.nanmean(cat_matrix)
        coverage = np.count_nonzero(~np.isnan(cat_matrix)) / cat_matrix.size
        cat_stats.append({
            "category": cat,
            "n_items": n_items,
            "mean_accuracy": mean_acc,
            "coverage": coverage,
        })
        print(f"  {cat:35s}  items={n_items:5d}  "
              f"mean_acc={mean_acc:.3f}  coverage={coverage:.3f}")

    cat_stats_df = pd.DataFrame(cat_stats)
    cat_stats_df.to_csv(os.path.join(OUTPUT_DIR, "category_summary.csv"),
                        index=False)

    # Per-model summary
    print("\n=== Per-Model Summary (top/bottom 10) ===")
    model_accs = np.nanmean(matrix, axis=1)
    model_summary = pd.DataFrame({
        "model": model_dirs,
        "mean_accuracy": model_accs,
        "n_items_covered": np.count_nonzero(~np.isnan(matrix), axis=1),
    }).sort_values("mean_accuracy", ascending=False)
    model_summary.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"),
                         index=False)

    print("  Top 10:")
    for _, row in model_summary.head(10).iterrows():
        print(f"    {row['model']:50s}  acc={row['mean_accuracy']:.3f}  "
              f"items={row['n_items_covered']}")
    print("  Bottom 10:")
    for _, row in model_summary.tail(10).iterrows():
        print(f"    {row['model']:50s}  acc={row['mean_accuracy']:.3f}  "
              f"items={row['n_items_covered']}")

    # Task difficulty summary
    task_difficulty = np.nanmean(matrix, axis=0)
    print(f"\n=== Task Difficulty Distribution ===")
    print(f"  Always pass (acc=1.0): {np.sum(task_difficulty == 1.0)}")
    print(f"  Always fail (acc=0.0): {np.sum(task_difficulty == 0.0)}")
    print(f"  Mixed (0<acc<1): {np.sum((task_difficulty > 0) & (task_difficulty < 1))}")
    print(f"  Mean difficulty: {np.nanmean(task_difficulty):.3f}")
    print(f"  Median difficulty: {np.nanmedian(task_difficulty):.3f}")


def extract_error_types(model_dirs, categories, category_task_ids,
                        all_global_ids, global_id_to_col):
    """Extract per-task error types (original 03 workflow)."""
    print(f"\nBuilding error type matrix...")

    # Build the error type matrix (string matrix)
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


def main():
    download()
    # Discover all models (directories in score dir, excluding CSVs)
    model_dirs = sorted([
        d for d in os.listdir(SCORE_DIR)
        if os.path.isdir(os.path.join(SCORE_DIR, d))
    ])
    print(f"Found {len(model_dirs)} models with score files")

    # Discover all categories from the first model
    sample_model = model_dirs[0]
    score_files = sorted([
        f for f in os.listdir(os.path.join(SCORE_DIR, sample_model))
        if f.endswith("_score.json")
    ])
    categories = [
        f.replace("BFCL_v3_", "").replace("_score.json", "")
        for f in score_files
    ]
    print(f"Found {len(categories)} categories: {categories}")

    # Build task ID lists per category using result files from a reference model
    # We need to know ALL task IDs, not just the failed ones
    print("\nDiscovering task IDs from result files...")

    # Use gpt-4o as reference (likely most complete)
    ref_model = "gpt-4o-2024-11-20-FC"
    if ref_model not in model_dirs:
        ref_model = model_dirs[0]

    category_task_ids = {}
    total_tasks = 0
    for cat in categories:
        ids = get_all_task_ids_from_result(RESULT_DIR, ref_model, cat)
        if ids is None:
            print(f"  WARNING: No result file for {ref_model}/{cat}")
            # Try another model
            for alt_model in model_dirs:
                ids = get_all_task_ids_from_result(RESULT_DIR, alt_model, cat)
                if ids is not None:
                    print(f"    -> Using {alt_model} instead")
                    break
        if ids is not None:
            # Prefix category to make IDs globally unique
            global_ids = [f"{cat}::{tid}" for tid in ids]
            category_task_ids[cat] = (ids, global_ids)
            total_tasks += len(ids)
            print(f"  {cat}: {len(ids)} tasks")
        else:
            print(f"  {cat}: SKIPPED (no result file found)")

    print(f"\nTotal unique task items: {total_tasks}")

    # Build the global ID list and index
    all_global_ids = []
    for cat in categories:
        if cat in category_task_ids:
            _, gids = category_task_ids[cat]
            all_global_ids.extend(gids)

    global_id_to_col = {gid: i for i, gid in enumerate(all_global_ids)}

    # Step 1: Build binary response matrix
    build_response_matrix(model_dirs, categories, category_task_ids,
                          all_global_ids, global_id_to_col)

    # Step 2: Extract error types
    extract_error_types(model_dirs, categories, category_task_ids,
                        all_global_ids, global_id_to_col)


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
