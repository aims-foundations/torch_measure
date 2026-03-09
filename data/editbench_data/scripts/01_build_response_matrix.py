#!/usr/bin/env python3
"""
01_build_response_matrix.py

Builds a response matrix for EDIT-Bench (Real VS Code Code Editing Benchmark
from Copilot Arena).

Data sources:
  - GitHub: waynchi/editbench -> results/whole_file/*.json (44 model files)
  - HuggingFace: copilot-arena/editbench (task metadata, 540 tasks)
  - Website: waynechi.com/edit-bench (leaderboard reference)

The response matrix has:
  - Rows: 540 tasks (indexed by problem_id)
  - Columns: 44 models
  - Values: float scores in [0, 1] representing test pass rate per task

Output:
  - processed/response_matrix.csv         (540 x 44, tasks x models)
  - processed/response_matrix_binary.csv  (540 x 44, binary pass/fail)
  - processed/task_metadata.csv           (540 rows with task attributes)

Score type: Test pass rate (float in [0, 1]).
  - 1.0 = all tests pass (pass@1)
  - 0.0 = no tests pass
  - Intermediate values = fraction of sub-tests passing

References:
  - Paper: arxiv.org/abs/2511.04486
  - GitHub: github.com/waynchi/editbench
  - Dataset: huggingface.co/datasets/copilot-arena/editbench
  - Leaderboard: waynechi.com/edit-bench
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_RESULTS_DIR = PROJECT_DIR / "raw_results"
PROCESSED_DIR = PROJECT_DIR / "processed"


def load_model_results(results_dir: Path) -> dict:
    """Load all model JSON result files from the results directory.

    Each JSON file is named {model_name}.json and contains:
      - "question_N": float (test pass rate for problem N)
      - "average_test_rate": float (aggregate)
      - "pass_rate": float (aggregate)

    Returns:
        dict mapping model_name -> full JSON data (including aggregates)
    """
    results = {}
    for json_file in sorted(results_dir.glob("*.json")):
        model_name = json_file.stem
        with open(json_file, "r") as f:
            results[model_name] = json.load(f)
    return results


def extract_question_ids(all_data: dict) -> list:
    """Extract and sort all unique question IDs across all models.

    Returns:
        Sorted list of question_id strings (e.g., ["question_1", "question_2", ...])
    """
    q_ids = set()
    for model_data in all_data.values():
        for key in model_data:
            if key.startswith("question_"):
                q_ids.add(key)
    return sorted(q_ids, key=lambda x: int(x.split("_")[1]))


def build_response_matrix(all_data: dict, question_ids: list) -> pd.DataFrame:
    """Build the tasks x models response matrix.

    Args:
        all_data: dict of model_name -> {question_id: score, ...}
        question_ids: sorted list of question_id strings

    Returns:
        DataFrame with index=task_id (question_N), columns=model_names
    """
    models = sorted(all_data.keys())
    matrix = {}
    for model in models:
        scores = []
        for q_id in question_ids:
            score = all_data[model].get(q_id, np.nan)
            scores.append(score)
        matrix[model] = scores

    df = pd.DataFrame(matrix, index=question_ids)
    df.index.name = "task_id"
    return df


def build_task_metadata(question_ids: list) -> pd.DataFrame:
    """Build task metadata from the HuggingFace dataset.

    Uses the 'complete' split (540 tasks) and 'test' split (108 core tasks).

    Returns:
        DataFrame with task metadata including problem_id, languages,
        difficulty, and split membership.
    """
    try:
        from datasets import load_dataset

        ds_complete = load_dataset("copilot-arena/editbench", split="complete")
        ds_test = load_dataset("copilot-arena/editbench", split="test")
        core_pids = set(item["problem_id"] for item in ds_test)
    except Exception as e:
        print(f"  WARNING: Could not load HuggingFace dataset: {e}")
        print("  Building minimal metadata from question IDs only.")
        df = pd.DataFrame({
            "task_id": question_ids,
            "problem_id": [int(q.split("_")[1]) for q in question_ids],
        })
        return df.set_index("task_id")

    metadata_rows = []
    for row in ds_complete:
        pid = row["problem_id"]
        q_id = f"question_{pid}"
        metadata_rows.append({
            "task_id": q_id,
            "problem_id": pid,
            "pair_id": row.get("pair_id", ""),
            "programming_language": row.get("programming_language", ""),
            "natural_language": row.get("natural_language", ""),
            "cursor_position": row.get("cursor_position", ""),
            "python_version": row.get("python_version", ""),
            "instruction_length": len(row.get("instruction", "")),
            "original_code_length": len(row.get("original_code", "")),
            "highlighted_code_length": len(row.get("highlighted_code", "")),
            "test_code_length": len(row.get("test_code", "")),
            "split": row.get("split", ""),
            "is_core": pid in core_pids,
            "instruction_preview": row.get("instruction", "")[:200],
        })

    df = pd.DataFrame(metadata_rows)
    df = df.set_index("task_id")
    df = df.sort_values("problem_id")
    # Reindex to match question_ids order
    df = df.reindex([q for q in question_ids if q in df.index])
    return df


def classify_difficulty(response_matrix: pd.DataFrame, threshold: int = 20):
    """Classify tasks as 'easy' or 'hard' using official EDIT-Bench methodology.

    A task is 'easy' if >= threshold models achieve a perfect score (1.0).
    Otherwise it is 'hard'.

    Returns:
        (difficulty_series, perfect_count_series)
    """
    perfect_counts = (response_matrix == 1.0).sum(axis=1)
    difficulty = pd.Series("hard", index=response_matrix.index, name="difficulty")
    difficulty[perfect_counts >= threshold] = "easy"
    return difficulty, perfect_counts


def print_summary(matrix, binary_matrix, metadata, all_data):
    """Print comprehensive summary statistics."""
    n_tasks, n_models = matrix.shape
    total_cells = n_tasks * n_models
    filled_cells = int(matrix.notna().sum().sum())
    fill_rate = filled_cells / total_cells * 100

    print("\n" + "=" * 72)
    print("EDIT-BENCH RESPONSE MATRIX — SUMMARY REPORT")
    print("=" * 72)

    print(f"\n--- Matrix Dimensions ---")
    print(f"  Tasks (rows):       {n_tasks}")
    print(f"  Models (columns):   {n_models}")
    print(f"  Total cells:        {total_cells}")
    print(f"  Filled cells:       {filled_cells}")
    print(f"  Fill rate:          {fill_rate:.2f}%")

    print(f"\n--- Score Type ---")
    print(f"  Primary:   Test pass rate (continuous float in [0.0, 1.0])")
    print(f"    1.0 = all sub-tests passed (full pass)")
    print(f"    0.0 = all sub-tests failed (full fail)")
    print(f"    Intermediate = fraction of sub-tests passed")
    print(f"  Secondary: Binary pass/fail (1 if score == 1.0, else 0)")

    all_scores = matrix.values[~np.isnan(matrix.values)]
    unique_scores = sorted(set(all_scores))
    binary_pct = 100 * np.sum((all_scores == 0.0) | (all_scores == 1.0)) / len(all_scores)
    print(f"\n--- Score Distribution ---")
    print(f"  Mean:               {np.mean(all_scores):.4f}")
    print(f"  Std:                {np.std(all_scores):.4f}")
    print(f"  Median:             {np.median(all_scores):.4f}")
    print(f"  Unique values:      {len(unique_scores)}")
    print(f"  Binary (0 or 1):    {binary_pct:.1f}% of all entries")
    print(f"  Fraction = 0.0:     {(all_scores == 0.0).mean():.4f}")
    print(f"  Fraction = 1.0:     {(all_scores == 1.0).mean():.4f}")

    # Per-model stats
    print(f"\n--- Per-Model Pass Rates (binary pass@1, sorted desc) ---")
    print(f"  {'Model':<40s} {'Tasks':>6s} {'MeanScore':>10s} {'Pass@1':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*10} {'-'*7}")
    model_pass = binary_matrix.mean(axis=0).sort_values(ascending=False)
    model_mean = matrix.mean(axis=0)
    model_count = matrix.notna().sum(axis=0)
    for model in model_pass.index:
        print(f"  {model:<40s} {int(model_count[model]):>6d} "
              f"{model_mean[model]:>10.4f} {model_pass[model]:>7.4f}")

    # Task difficulty
    task_means = matrix.mean(axis=1)
    print(f"\n--- Task Difficulty ---")
    print(f"  Easiest task:       {task_means.idxmax()} "
          f"(mean={task_means.max():.4f})")
    print(f"  Hardest task:       {task_means.idxmin()} "
          f"(mean={task_means.min():.4f})")
    n_all_pass = (task_means == 1.0).sum()
    n_all_fail = (task_means == 0.0).sum()
    print(f"  Tasks all models pass: {n_all_pass}")
    print(f"  Tasks all models fail: {n_all_fail}")

    # Difficulty classification
    if "difficulty" in metadata.columns:
        easy_n = (metadata["difficulty"] == "easy").sum()
        hard_n = (metadata["difficulty"] == "hard").sum()
        print(f"  Easy (>=20 models perfect): {easy_n}")
        print(f"  Hard (<20 models perfect):  {hard_n}")

    # Core vs non-core
    if "is_core" in metadata.columns:
        core_ids = metadata[metadata["is_core"]].index
        non_core_ids = metadata[~metadata["is_core"]].index
        core_vals = matrix.loc[matrix.index.isin(core_ids)].values
        core_vals = core_vals[~np.isnan(core_vals)]
        non_core_vals = matrix.loc[matrix.index.isin(non_core_ids)].values
        non_core_vals = non_core_vals[~np.isnan(non_core_vals)]
        print(f"\n--- Core vs Complete Split ---")
        print(f"  Core (108 tasks):     mean = {core_vals.mean():.4f}")
        print(f"  Non-core (432 tasks): mean = {non_core_vals.mean():.4f}")

    # Language distributions
    if "programming_language" in metadata.columns:
        print(f"\n--- Programming Language Distribution ---")
        for lang, count in metadata["programming_language"].value_counts().items():
            lang_ids = metadata[metadata["programming_language"] == lang].index
            lang_scores = matrix.loc[matrix.index.isin(lang_ids)].values
            lang_scores = lang_scores[~np.isnan(lang_scores)]
            print(f"  {lang:<25s} {count:>4d} tasks, "
                  f"mean_score={np.mean(lang_scores):.4f}")

    if "natural_language" in metadata.columns:
        print(f"\n--- Natural Language Distribution ---")
        for lang, count in metadata["natural_language"].value_counts().items():
            lang_ids = metadata[metadata["natural_language"] == lang].index
            lang_scores = matrix.loc[matrix.index.isin(lang_ids)].values
            lang_scores = lang_scores[~np.isnan(lang_scores)]
            print(f"  {lang:<25s} {count:>4d} tasks, "
                  f"mean_score={np.mean(lang_scores):.4f}")

    # Model names list
    print(f"\n--- All Model Names ({n_models}) ---")
    for i, model in enumerate(sorted(matrix.columns), 1):
        atr = all_data[model].get("average_test_rate", "N/A")
        pr = all_data[model].get("pass_rate", "N/A")
        atr_s = f"{atr:.4f}" if isinstance(atr, float) else atr
        pr_s = f"{pr:.4f}" if isinstance(pr, float) else pr
        print(f"  {i:>2d}. {model:<40s} pass_rate={pr_s}  avg_test_rate={atr_s}")

    print("\n" + "=" * 72)


def main():
    print("=" * 72)
    print("EDIT-Bench Response Matrix Builder")
    print("=" * 72)

    # ── Step 1: Load raw model results ─────────────────────────────────────
    print(f"\n[1/5] Loading model result JSON files from {RAW_RESULTS_DIR}...")
    all_data = load_model_results(RAW_RESULTS_DIR)
    print(f"  Loaded {len(all_data)} models")

    if not all_data:
        print("ERROR: No JSON files found. Run the download step first.")
        return 1

    # ── Step 2: Extract question IDs ───────────────────────────────────────
    print("\n[2/5] Extracting question IDs...")
    question_ids = extract_question_ids(all_data)
    print(f"  Found {len(question_ids)} unique tasks")
    nums = [int(q.split("_")[1]) for q in question_ids]
    print(f"  ID range: {min(nums)} to {max(nums)}")
    expected = set(range(min(nums), max(nums) + 1))
    missing = sorted(expected - set(nums))
    print(f"  Missing IDs (gaps in numbering): {missing}")

    # ── Step 3: Build response matrix ──────────────────────────────────────
    print("\n[3/5] Building response matrix...")
    response_matrix = build_response_matrix(all_data, question_ids)
    total_cells = response_matrix.shape[0] * response_matrix.shape[1]
    filled_cells = int(response_matrix.notna().sum().sum())
    fill_rate = filled_cells / total_cells
    print(f"  Matrix shape: {response_matrix.shape} (tasks x models)")
    print(f"  Fill rate: {filled_cells}/{total_cells} ({100*fill_rate:.1f}%)")

    # Binarize: strict pass@1 (score == 1.0 -> 1, else 0, preserve NaN)
    binary_matrix = (response_matrix == 1.0).astype(float)
    binary_matrix[response_matrix.isna()] = np.nan

    # ── Step 4: Build task metadata ────────────────────────────────────────
    print("\n[4/5] Building task metadata from HuggingFace...")
    task_metadata = build_task_metadata(question_ids)
    print(f"  Task metadata shape: {task_metadata.shape}")

    # Add difficulty classification
    difficulty, perfect_counts = classify_difficulty(response_matrix)
    task_metadata["difficulty"] = difficulty
    task_metadata["num_models_perfect"] = perfect_counts

    # Add per-task aggregate stats
    task_metadata["mean_score"] = response_matrix.mean(axis=1)
    task_metadata["median_score"] = response_matrix.median(axis=1)
    task_metadata["std_score"] = response_matrix.std(axis=1)
    task_metadata["binary_pass_rate"] = binary_matrix.mean(axis=1)

    # ── Step 5: Save outputs ───────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs to {PROCESSED_DIR}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Response matrix (continuous scores)
    out1 = PROCESSED_DIR / "response_matrix.csv"
    response_matrix.to_csv(out1)
    print(f"  {out1} ({out1.stat().st_size / 1024:.1f} KB)")

    # Response matrix (binary pass/fail)
    out2 = PROCESSED_DIR / "response_matrix_binary.csv"
    binary_matrix.to_csv(out2)
    print(f"  {out2} ({out2.stat().st_size / 1024:.1f} KB)")

    # Task metadata
    out3 = PROCESSED_DIR / "task_metadata.csv"
    task_metadata.to_csv(out3)
    print(f"  {out3} ({out3.stat().st_size / 1024:.1f} KB)")

    # ── Summary Report ─────────────────────────────────────────────────────
    print_summary(response_matrix, binary_matrix, task_metadata, all_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
