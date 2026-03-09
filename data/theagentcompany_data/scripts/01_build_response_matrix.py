#!/usr/bin/env python3
"""
Build TheAgentCompany response matrix from the experiments repo.

Data source:
  https://github.com/TheAgentCompany/experiments
  Branch: main, path: evaluation/1.0.0/<date_model>/results/eval_<task_id>.json

Each eval JSON has the structure:
  {
    "checkpoints": [{"total": N, "result": M}, ...],
    "final_score": {"total": T, "result": R}
  }

Outputs:
  - response_matrix.csv: Continuous (0-1) score matrix (models x tasks)
  - response_matrix_binary.csv: Binary pass/fail matrix (1 if score==1.0, else 0)
  - model_summary.csv: Per-model summary statistics
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
EXPERIMENTS_DIR = RAW_DIR / "experiments"
EVAL_DIR = EXPERIMENTS_DIR / "evaluation" / "1.0.0"
PROCESSED_DIR = BASE_DIR / "processed"

REPO_URL = "https://github.com/TheAgentCompany/experiments.git"


def ensure_raw_data():
    """Clone the experiments repo if not already present."""
    if EXPERIMENTS_DIR.exists() and EVAL_DIR.exists():
        n_models = len(list(EVAL_DIR.iterdir()))
        if n_models > 0:
            print(f"Raw data already present: {EVAL_DIR}")
            print(f"  Found {n_models} model directories")
            return

    print(f"Cloning {REPO_URL} ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(EXPERIMENTS_DIR)],
        check=True,
    )
    print("Clone complete.")


def normalize_task_id(filename):
    """Extract a canonical task ID from an eval result filename.

    Handles two naming conventions:
      - eval_<task>-image.json  (most models)
      - eval_<task>.json        (some models, e.g. MUSE)

    Returns the canonical task ID, e.g. 'admin-arrange-meeting-rooms'.
    """
    name = filename
    if name.endswith(".json"):
        name = name[:-5]
    if name.startswith("eval_"):
        name = name[5:]
    if name.endswith("-image"):
        name = name[:-6]
    return name


def load_all_results():
    """Load per-task scores for every model.

    Returns:
        model_results: dict of {model_name: {task_id: score_fraction}}
        all_task_ids: sorted list of all unique task IDs
    """
    model_results = {}
    all_task_ids = set()

    for model_dir in sorted(EVAL_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        results_dir = model_dir / "results"
        if not results_dir.exists():
            print(f"WARNING: No results/ dir in {model_dir.name}, skipping",
                  file=sys.stderr)
            continue

        model_name = model_dir.name
        model_scores = {}

        for fpath in sorted(results_dir.glob("eval_*.json")):
            task_id = normalize_task_id(fpath.name)

            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING: Skipping {fpath}: {e}", file=sys.stderr)
                continue

            if "final_score" not in data:
                print(f"WARNING: No 'final_score' in {fpath}, skipping",
                      file=sys.stderr)
                continue

            total = data["final_score"]["total"]
            result = data["final_score"]["result"]

            if total > 0:
                score = result / total
            else:
                score = 0.0

            model_scores[task_id] = score
            all_task_ids.add(task_id)

        model_results[model_name] = model_scores
        print(f"  Loaded {model_name}: {len(model_scores)} tasks")

    return model_results, sorted(all_task_ids)


def build_response_matrix(model_results, task_ids):
    """Build response matrix: models (rows) x tasks (columns).

    Uses NaN for tasks not evaluated by a model.
    """
    rows = {}
    for model_name in sorted(model_results.keys()):
        scores = model_results[model_name]
        rows[model_name] = [
            scores.get(tid, np.nan) for tid in task_ids
        ]

    df = pd.DataFrame.from_dict(rows, orient="index", columns=task_ids)
    df.index.name = "model"
    return df


def build_binary_matrix(response_matrix):
    """Convert continuous scores to binary: 1 if score == 1.0, else 0.
    Preserves NaN for unevaluated tasks."""
    binary = response_matrix.copy()
    binary[binary.notna()] = (binary[binary.notna()] == 1.0).astype(float)
    return binary


def build_model_summary(response_matrix, binary_matrix):
    """Build per-model summary statistics."""
    rows = []
    for model in response_matrix.index:
        scores = response_matrix.loc[model]
        binary = binary_matrix.loc[model]

        n_evaluated = scores.notna().sum()
        n_total = len(scores)
        avg_score = scores.mean()  # ignores NaN
        perfect_count = (binary == 1.0).sum()
        zero_count = (scores == 0.0).sum()
        partial_count = n_evaluated - perfect_count - zero_count

        rows.append({
            "model": model,
            "tasks_evaluated": int(n_evaluated),
            "tasks_total": n_total,
            "coverage": round(n_evaluated / n_total * 100, 1),
            "avg_score": round(avg_score, 4),
            "avg_score_pct": round(avg_score * 100, 2),
            "perfect_completions": int(perfect_count),
            "perfect_rate_pct": round(perfect_count / n_evaluated * 100, 2)
                if n_evaluated > 0 else 0.0,
            "zero_score_tasks": int(zero_count),
            "partial_score_tasks": int(partial_count),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        "avg_score", ascending=False
    ).reset_index(drop=True)
    return summary


def print_matrix_statistics(response_matrix, binary_matrix, model_summary):
    """Print comprehensive summary statistics."""
    n_models, n_tasks = response_matrix.shape
    total_cells = n_models * n_tasks
    n_evaluated = response_matrix.notna().sum().sum()
    n_missing = total_cells - n_evaluated
    fill_rate = n_evaluated / total_cells

    print("\n" + "=" * 70)
    print("  RESPONSE MATRIX STATISTICS")
    print("=" * 70)
    print(f"  Models (rows):         {n_models}")
    print(f"  Tasks (columns):       {n_tasks}")
    print(f"  Matrix dimensions:     {n_models} x {n_tasks}")
    print(f"  Total cells:           {total_cells:,}")
    print(f"  Evaluated cells:       {int(n_evaluated):,} ({fill_rate*100:.1f}%)")
    print(f"  Missing cells (NaN):   {int(n_missing):,} ({n_missing/total_cells*100:.1f}%)")

    # Continuous score statistics
    all_scores = response_matrix.values[~np.isnan(response_matrix.values)]
    print(f"\n  Continuous score statistics (evaluated cells only):")
    print(f"    Mean score:          {all_scores.mean():.4f} ({all_scores.mean()*100:.1f}%)")
    print(f"    Median score:        {np.median(all_scores):.4f}")
    print(f"    Std:                 {all_scores.std():.4f}")
    print(f"    Score = 0.0:         {(all_scores == 0.0).sum():,}"
          f" ({(all_scores == 0.0).mean()*100:.1f}%)")
    print(f"    Score = 1.0:         {(all_scores == 1.0).sum():,}"
          f" ({(all_scores == 1.0).mean()*100:.1f}%)")
    print(f"    0 < Score < 1:       {((all_scores > 0) & (all_scores < 1)).sum():,}"
          f" ({((all_scores > 0) & (all_scores < 1)).mean()*100:.1f}%)")

    # Per-model statistics
    per_model_avg = response_matrix.mean(axis=1)
    print(f"\n  Per-model average score:")
    best_model = per_model_avg.idxmax()
    worst_model = per_model_avg.idxmin()
    print(f"    Best:   {per_model_avg.max()*100:.1f}% ({best_model})")
    print(f"    Worst:  {per_model_avg.min()*100:.1f}% ({worst_model})")
    print(f"    Median: {per_model_avg.median()*100:.1f}%")
    print(f"    Std:    {per_model_avg.std()*100:.1f}%")

    # Per-task statistics (among models that evaluated them)
    per_task_avg = response_matrix.mean(axis=0)
    per_task_coverage = response_matrix.notna().sum(axis=0)
    print(f"\n  Per-task solve rate (across models):")
    print(f"    Min:    {per_task_avg.min()*100:.1f}%")
    print(f"    Max:    {per_task_avg.max()*100:.1f}%")
    print(f"    Median: {per_task_avg.median()*100:.1f}%")
    print(f"    Std:    {per_task_avg.std()*100:.1f}%")

    # Task difficulty distribution (based on binary pass/fail)
    per_task_pass = binary_matrix.mean(axis=0)
    unsolved = (per_task_pass == 0).sum()
    easy = (per_task_pass > 0.8).sum()
    hard = ((per_task_pass > 0) & (per_task_pass <= 0.2)).sum()
    solved_by_all = (per_task_pass == 1.0).sum()
    print(f"\n  Task difficulty distribution (binary: perfect completion):")
    print(f"    Solved by NO model:    {unsolved}")
    print(f"    Hard (<=20%):          {hard}")
    print(f"    Easy (>80%):           {easy}")
    print(f"    Solved by ALL models:  {solved_by_all}")

    # Task category breakdown
    print(f"\n  Task category breakdown:")
    categories = {}
    for task_id in response_matrix.columns:
        cat = task_id.split("-")[0]
        categories.setdefault(cat, []).append(task_id)

    for cat in sorted(categories.keys()):
        task_list = categories[cat]
        cat_scores = response_matrix[task_list].values
        cat_scores_flat = cat_scores[~np.isnan(cat_scores)]
        if len(cat_scores_flat) > 0:
            avg = cat_scores_flat.mean()
            perfect = (cat_scores_flat == 1.0).mean()
            print(f"    {cat:12s}: {len(task_list):3d} tasks, "
                  f"avg_score={avg*100:.1f}%, perfect_rate={perfect*100:.1f}%")

    # Model coverage
    print(f"\n  Model coverage:")
    full_coverage = (response_matrix.notna().sum(axis=1) == n_tasks).sum()
    partial_coverage = n_models - full_coverage
    print(f"    Full coverage ({n_tasks} tasks):  {full_coverage} models")
    print(f"    Partial coverage:               {partial_coverage} models")
    for model in response_matrix.index:
        n_eval = response_matrix.loc[model].notna().sum()
        if n_eval < n_tasks:
            print(f"      {model}: {n_eval}/{n_tasks} tasks")

    # Top models
    print(f"\n  Top 10 models (by average score):")
    print(f"  {'Rank':<5} {'Model':<55} {'Avg%':>6} {'Perfect':>8}")
    print(f"  {'-'*5} {'-'*55} {'-'*6} {'-'*8}")
    for i, (_, row) in enumerate(model_summary.head(10).iterrows()):
        print(f"  {i+1:<5} {row['model']:<55} "
              f"{row['avg_score_pct']:>5.1f}% {row['perfect_completions']:>7d}")


def main():
    print("TheAgentCompany Response Matrix Builder")
    print("=" * 70)

    # Step 1: Ensure raw data
    ensure_raw_data()

    # Step 2: Load all results
    print(f"\nLoading results from: {EVAL_DIR}")
    model_results, task_ids = load_all_results()
    print(f"\nFound {len(model_results)} models")
    print(f"Found {len(task_ids)} unique task IDs")

    # Step 3: Build response matrices
    print("\nBuilding response matrices...")
    response_matrix = build_response_matrix(model_results, task_ids)
    binary_matrix = build_binary_matrix(response_matrix)
    print(f"Response matrix shape: {response_matrix.shape}")

    # Step 4: Build model summary
    print("Building model summary...")
    model_summary = build_model_summary(response_matrix, binary_matrix)

    # Step 5: Save outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matrix_path = PROCESSED_DIR / "response_matrix.csv"
    response_matrix.to_csv(matrix_path)
    print(f"\nSaved continuous response matrix to: {matrix_path}")

    binary_path = PROCESSED_DIR / "response_matrix_binary.csv"
    binary_matrix.to_csv(binary_path)
    print(f"Saved binary response matrix to: {binary_path}")

    summary_path = PROCESSED_DIR / "model_summary.csv"
    model_summary.to_csv(summary_path, index=False)
    print(f"Saved model summary to: {summary_path}")

    # Step 6: Print comprehensive statistics
    print_matrix_statistics(response_matrix, binary_matrix, model_summary)

    # Final file listing
    print(f"\n{'='*70}")
    print(f"  OUTPUT FILES")
    print(f"{'='*70}")
    for f in sorted(PROCESSED_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s}  {size_kb:.1f} KB")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
