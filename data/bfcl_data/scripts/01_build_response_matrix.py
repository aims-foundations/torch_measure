"""
Build a per-task binary response matrix from BFCL score files.

The BFCL-Result repo stores score files as NDJSON:
  - Line 0: {"accuracy": ..., "correct_count": ..., "total_count": ...}
  - Lines 1+: failed task entries with {"id": ..., "valid": false, ...}

Strategy: for each model × category, load the score file, get the total count
from the header, collect failed IDs, and mark all other IDs as pass (1).

Uses the 2024-12-29 snapshot (most complete: 92 models, 22 categories).

Output: processed/response_matrix.csv (models × tasks, binary 0/1)
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SCORE_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "score")
RESULT_DIR = str(_BENCHMARK_DIR / "raw" / "BFCL-Result" / "2024-12-29" / "result")
OUTPUT_DIR = str(_BENCHMARK_DIR / "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def main():
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

    # Build the response matrix
    # Columns = all task IDs (globally unique), Rows = models
    all_global_ids = []
    for cat in categories:
        if cat in category_task_ids:
            _, gids = category_task_ids[cat]
            all_global_ids.extend(gids)

    print(f"\nBuilding {len(model_dirs)} × {len(all_global_ids)} response matrix...")

    matrix = np.full((len(model_dirs), len(all_global_ids)), np.nan)
    global_id_to_col = {gid: i for i, gid in enumerate(all_global_ids)}

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


if __name__ == "__main__":
    main()
