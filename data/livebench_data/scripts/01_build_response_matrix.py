"""
Build LiveBench response matrices from HuggingFace per-model per-question judgment data.

Data sources:
  - livebench/model_judgment (HuggingFace, "leaderboard" split):
    Per-model per-question scores for 195 models on 494 questions across 3 categories
    (coding, instruction_following, language) and 7 tasks.
    Scores are continuous [0, 1], with some tasks being binary (0/1) and others
    having partial credit (e.g., connections: {0, 0.25, 0.33, 0.5, 0.67, 0.75, 1}).

  - livebench/{category} (HuggingFace, "test" split) for each of 6 categories:
    Question metadata (question_id, task, category, ground_truth, release_date, etc.)
    Used to build task_metadata.csv.

Note on coverage:
  The model_judgment HF dataset covers 3 of 6 LiveBench categories:
    - coding (LCB_generation, coding_completion): 128 questions
    - language (connections, plot_unscrambling, typos): 290 questions
    - instruction_following (paraphrase, story_generation): 76 questions

  The remaining 3 categories (math, reasoning, data_analysis) with 11 tasks:
    - math (AMPS_Hard, math_comp, olympiad): 368 questions
    - reasoning (spatial, web_of_lies_v2, zebra_puzzle): 200 questions
    - data_analysis (cta, tablejoin, tablereformat): 150 questions
  These require running the LiveBench ground-truth scoring pipeline locally
  (gen_ground_truth_judgment.py) on model answers from livebench/model_answer.
  They are NOT included in the model_judgment HF dataset.

Outputs:
  - response_matrix.csv: Continuous scores (models x questions) for all available data
  - response_matrix_binary.csv: Binarized scores (>=0.5 -> 1, else 0)
  - question_metadata.csv: Per-question metadata from HuggingFace
  - model_summary.csv: Per-model aggregate statistics

References:
  - GitHub: https://github.com/LiveBench/LiveBench
  - Paper: https://arxiv.org/abs/2406.19314
  - Website: https://livebench.ai
  - HF org: https://huggingface.co/livebench
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# LiveBench categories
ALL_CATEGORIES = [
    "coding", "data_analysis", "instruction_following",
    "language", "math", "reasoning",
]
JUDGMENT_CATEGORIES = ["coding", "instruction_following", "language"]
MISSING_CATEGORIES = ["data_analysis", "math", "reasoning"]


def download_model_judgment():
    """Download the model_judgment dataset from HuggingFace.

    Returns a list of dicts with keys:
        question_id, task, model, score, turn, tstamp, category
    """
    cache_path = os.path.join(RAW_DIR, "model_judgment.parquet")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Loading cached model_judgment from {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  Loaded {len(df)} rows from cache")
        return df

    print("  Downloading livebench/model_judgment from HuggingFace...")

    try:
        from datasets import load_dataset
        ds = load_dataset("livebench/model_judgment", split="leaderboard")
        df = ds.to_pandas()
        df.to_parquet(cache_path, index=False)
        print(f"  Downloaded and cached {len(df)} rows to {cache_path}")
        return df

    except ImportError:
        print("  'datasets' library not available, trying direct parquet download...")
        url = (
            "https://huggingface.co/datasets/livebench/model_judgment/resolve/main/"
            "data/leaderboard-00000-of-00001.parquet"
        )
        print(f"  Downloading from {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        with open(cache_path, "wb") as f:
            f.write(data)
        df = pd.read_parquet(cache_path)
        print(f"  Downloaded {len(df)} rows")
        return df


def download_question_metadata():
    """Download question metadata from per-category HuggingFace datasets.

    Returns a DataFrame with columns:
        question_id, category, task, livebench_release_date, livebench_removal_date, ...
    """
    cache_path = os.path.join(RAW_DIR, "question_metadata.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        print(f"  Loading cached question metadata from {cache_path}")
        return pd.read_csv(cache_path)

    print("  Downloading question metadata from HuggingFace category datasets...")

    all_rows = []

    try:
        from datasets import load_dataset

        for cat_name in ALL_CATEGORIES:
            print(f"    Loading livebench/{cat_name}...")
            ds = load_dataset(f"livebench/{cat_name}", split="test")
            for item in ds:
                row = {
                    "question_id": item["question_id"],
                    "category": item.get("category", cat_name),
                    "task": item.get("task", ""),
                }
                # Optional fields
                for field in [
                    "livebench_release_date", "livebench_removal_date",
                    "subtask", "hardness", "level",
                ]:
                    if field in item:
                        val = item[field]
                        if hasattr(val, "strftime"):
                            val = val.strftime("%Y-%m-%d")
                        row[field] = val
                all_rows.append(row)

        meta_df = pd.DataFrame(all_rows)
        meta_df.to_csv(cache_path, index=False)
        print(f"  Saved {len(meta_df)} question metadata rows to {cache_path}")
        return meta_df

    except ImportError:
        print("  WARNING: 'datasets' library not available.")
        print("  Question metadata will be derived from model_judgment data only.")
        return None

    except Exception as e:
        print(f"  WARNING: Failed to download question metadata: {e}")
        return None


def build_response_matrix(judgment_df):
    """Build response matrices from judgment data.

    Args:
        judgment_df: DataFrame with columns [question_id, task, model, score, category]

    Returns:
        matrix_df: DataFrame with models as index, question_ids as columns, scores as values
    """
    print(f"\n{'='*70}")
    print(f"  Building Response Matrix")
    print(f"{'='*70}")

    # Pivot to models x questions
    # Handle potential duplicates by taking the mean
    pivot_df = judgment_df.pivot_table(
        index="model",
        columns="question_id",
        values="score",
        aggfunc="mean",
    )

    n_models, n_questions = pivot_df.shape
    total_cells = n_models * n_questions
    n_valid = pivot_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    # Score statistics
    all_scores = pivot_df.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]

    print(f"  Models:          {n_models}")
    print(f"  Questions:       {n_questions}")
    print(f"  Matrix dims:     {n_models} x {n_questions}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  Valid cells:     {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells:   {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:       {fill_rate*100:.1f}%")

    if len(valid_scores) > 0:
        print(f"\n  Score distribution (0-1 continuous):")
        print(f"    Mean:   {np.mean(valid_scores):.4f}")
        print(f"    Median: {np.median(valid_scores):.4f}")
        print(f"    Std:    {np.std(valid_scores):.4f}")
        print(f"    Min:    {np.min(valid_scores):.4f}")
        print(f"    Max:    {np.max(valid_scores):.4f}")

        # Score distribution buckets
        print(f"\n  Score distribution (buckets):")
        bins = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.01]
        labels = [
            "  0.00 (exact)", "  0.01-0.10", "  0.10-0.25", "  0.25-0.50",
            "  0.50-0.75", "  0.75-0.90", "  0.90-0.99", "  0.99-1.00 (excl exact)",
            "  1.00 (exact)",
        ]
        # Count exact 0 and exact 1 separately
        n_exact_0 = np.sum(valid_scores == 0.0)
        n_exact_1 = np.sum(valid_scores == 1.0)
        n_between = len(valid_scores) - n_exact_0 - n_exact_1
        pct_0 = n_exact_0 / len(valid_scores) * 100
        pct_1 = n_exact_1 / len(valid_scores) * 100
        pct_between = n_between / len(valid_scores) * 100
        print(f"    Score = 0:     {n_exact_0:>8,} ({pct_0:5.1f}%)")
        print(f"    0 < Score < 1: {n_between:>8,} ({pct_between:5.1f}%)")
        print(f"    Score = 1:     {n_exact_1:>8,} ({pct_1:5.1f}%)")

    # Per-model stats
    per_model_mean = pivot_df.mean(axis=1)
    per_model_count = pivot_df.notna().sum(axis=1)
    print(f"\n  Per-model mean score:")
    best_model = per_model_mean.idxmax()
    worst_model = per_model_mean.idxmin()
    print(f"    Best:   {per_model_mean.max():.4f} ({best_model})")
    print(f"    Worst:  {per_model_mean.min():.4f} ({worst_model})")
    print(f"    Median: {per_model_mean.median():.4f}")
    print(f"    Std:    {per_model_mean.std():.4f}")

    print(f"\n  Per-model question coverage:")
    print(f"    Min:    {per_model_count.min()} ({per_model_count.idxmin()})")
    print(f"    Max:    {per_model_count.max()} ({per_model_count.idxmax()})")
    print(f"    Median: {per_model_count.median():.0f}")

    # Per-question stats
    per_question_mean = pivot_df.mean(axis=0)
    per_question_count = pivot_df.notna().sum(axis=0)
    print(f"\n  Per-question mean score:")
    print(f"    Min:    {per_question_mean.min():.4f}")
    print(f"    Max:    {per_question_mean.max():.4f}")
    print(f"    Median: {per_question_mean.median():.4f}")
    print(f"    Std:    {per_question_mean.std():.4f}")

    # Question difficulty distribution
    easy = (per_question_mean >= 0.8).sum()
    medium = ((per_question_mean >= 0.3) & (per_question_mean < 0.8)).sum()
    hard = ((per_question_mean > 0) & (per_question_mean < 0.3)).sum()
    unsolved = (per_question_mean == 0).sum()
    print(f"\n  Question difficulty (by mean score):")
    print(f"    Unsolved (0.0):    {unsolved}")
    print(f"    Hard (<0.3):       {hard}")
    print(f"    Medium (0.3-0.8):  {medium}")
    print(f"    Easy (>=0.8):      {easy}")

    return pivot_df


def build_per_category_stats(judgment_df):
    """Print per-category and per-task statistics."""
    print(f"\n{'='*70}")
    print(f"  Per-Category & Per-Task Breakdown")
    print(f"{'='*70}")

    for category in sorted(judgment_df["category"].unique()):
        cat_df = judgment_df[judgment_df["category"] == category]
        n_models = cat_df["model"].nunique()
        n_questions = cat_df["question_id"].nunique()
        mean_score = cat_df["score"].mean()

        print(f"\n  {category.upper()}: {n_models} models, {n_questions} questions, "
              f"mean={mean_score:.3f}")

        for task in sorted(cat_df["task"].unique()):
            task_df = cat_df[cat_df["task"] == task]
            t_models = task_df["model"].nunique()
            t_questions = task_df["question_id"].nunique()
            t_mean = task_df["score"].mean()
            t_binary = (task_df["score"].isin([0.0, 1.0])).mean()
            print(f"    {task:30s}  models={t_models:4d}  questions={t_questions:4d}  "
                  f"mean={t_mean:.3f}  binary_pct={t_binary*100:.0f}%")


def build_model_summary(judgment_df, matrix_df):
    """Build a model summary CSV with per-category and overall statistics."""
    print(f"\n{'='*70}")
    print(f"  Building Model Summary")
    print(f"{'='*70}")

    rows = []
    for model in sorted(judgment_df["model"].unique()):
        model_data = judgment_df[judgment_df["model"] == model]
        row = {
            "model": model,
            "n_questions_scored": len(model_data),
            "n_questions_total": matrix_df.shape[1],
            "overall_mean_score": model_data["score"].mean(),
        }

        # Per-category scores
        for category in JUDGMENT_CATEGORIES:
            cat_data = model_data[model_data["category"] == category]
            if len(cat_data) > 0:
                row[f"{category}_mean"] = cat_data["score"].mean()
                row[f"{category}_n"] = len(cat_data)
            else:
                row[f"{category}_mean"] = None
                row[f"{category}_n"] = 0

        # Per-task scores
        for task in sorted(judgment_df["task"].unique()):
            task_data = model_data[model_data["task"] == task]
            if len(task_data) > 0:
                row[f"task_{task}_mean"] = task_data["score"].mean()
            else:
                row[f"task_{task}_mean"] = None

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("overall_mean_score", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"  Total models: {len(summary_df)}")

    print(f"\n  Top 15 models (by overall mean score):")
    for _, r in summary_df.head(15).iterrows():
        n_q = int(r["n_questions_scored"])
        score = r["overall_mean_score"]
        print(f"    {r['model']:50s}  score={score:.3f}  n={n_q}")

    print(f"\n  Bottom 5 models (by overall mean score):")
    for _, r in summary_df.tail(5).iterrows():
        n_q = int(r["n_questions_scored"])
        score = r["overall_mean_score"]
        print(f"    {r['model']:50s}  score={score:.3f}  n={n_q}")

    print(f"\n  Saved: {output_path}")
    return summary_df


def save_question_metadata(question_meta_df, judgment_df):
    """Save enriched question metadata with score statistics."""
    if question_meta_df is None:
        # Build minimal metadata from judgment data
        print("\n  Building question metadata from judgment data (no HF metadata available)...")
        meta_rows = []
        for qid in judgment_df["question_id"].unique():
            q_data = judgment_df[judgment_df["question_id"] == qid].iloc[0]
            meta_rows.append({
                "question_id": qid,
                "category": q_data["category"],
                "task": q_data["task"],
            })
        question_meta_df = pd.DataFrame(meta_rows)

    # Compute per-question stats from judgment data
    q_stats = judgment_df.groupby("question_id").agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        n_models_scored=("score", "count"),
    ).reset_index()

    # Get task/category mapping from judgment data
    q_task_map = judgment_df[["question_id", "task", "category"]].drop_duplicates()

    # Merge
    merged = q_stats.merge(q_task_map, on="question_id", how="left")

    # Also merge with full metadata if available
    if "livebench_release_date" in question_meta_df.columns:
        extra_cols = [c for c in question_meta_df.columns
                      if c not in ["category", "task"]]
        if "question_id" in extra_cols:
            merged = merged.merge(
                question_meta_df[extra_cols],
                on="question_id",
                how="left",
            )

    output_path = os.path.join(PROCESSED_DIR, "question_metadata.csv")
    merged.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"  Question Metadata")
    print(f"{'='*70}")
    print(f"  Total questions with scores: {len(merged)}")

    # Category breakdown
    if "category" in merged.columns:
        print(f"\n  Category breakdown:")
        for cat in sorted(merged["category"].unique()):
            cat_q = merged[merged["category"] == cat]
            print(f"    {cat:25s}  n={len(cat_q):4d}  "
                  f"mean_score={cat_q['mean_score'].mean():.3f}")

    # Task breakdown
    if "task" in merged.columns:
        print(f"\n  Task breakdown:")
        for task in sorted(merged["task"].unique()):
            task_q = merged[merged["task"] == task]
            print(f"    {task:30s}  n={len(task_q):4d}  "
                  f"mean_score={task_q['mean_score'].mean():.3f}  "
                  f"models_per_q={task_q['n_models_scored'].mean():.0f}")

    print(f"\n  Saved: {output_path}")
    return merged


def main():
    print("LiveBench Response Matrix Builder")
    print("=" * 70)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # ----------------------------------------------------------------
    # Step 1: Download model judgment data
    # ----------------------------------------------------------------
    print("STEP 1: Downloading model judgment data")
    print("-" * 70)
    judgment_df = download_model_judgment()

    # ----------------------------------------------------------------
    # Step 2: Download question metadata
    # ----------------------------------------------------------------
    print("\nSTEP 2: Downloading question metadata")
    print("-" * 70)
    question_meta_df = download_question_metadata()

    # ----------------------------------------------------------------
    # Step 3: Per-category & per-task statistics
    # ----------------------------------------------------------------
    print("\nSTEP 3: Per-category and per-task statistics")
    print("-" * 70)
    build_per_category_stats(judgment_df)

    # ----------------------------------------------------------------
    # Step 4: Build response matrix
    # ----------------------------------------------------------------
    print("\nSTEP 4: Building response matrix")
    print("-" * 70)
    matrix_df = build_response_matrix(judgment_df)

    # Save continuous score matrix (primary output)
    matrix_df.index.name = "Model"
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"\n  Saved continuous response matrix: {output_path}")

    # Save binary response matrix (threshold at 0.5)
    binary_df = (matrix_df >= 0.5).astype(int)
    # Keep NaN as NaN
    binary_df = binary_df.where(matrix_df.notna())
    binary_path = os.path.join(PROCESSED_DIR, "response_matrix_binary.csv")
    binary_df.to_csv(binary_path)
    print(f"  Saved binary response matrix (threshold=0.5): {binary_path}")

    # Binary matrix stats
    valid_binary = binary_df.values.flatten()
    valid_binary = valid_binary[~np.isnan(valid_binary)]
    print(f"\n  Binary matrix stats:")
    print(f"    Pass (>=0.5): {int(np.sum(valid_binary)):,} "
          f"({np.sum(valid_binary)/len(valid_binary)*100:.1f}%)")
    print(f"    Fail (<0.5):  {int(len(valid_binary) - np.sum(valid_binary)):,} "
          f"({(1 - np.sum(valid_binary)/len(valid_binary))*100:.1f}%)")

    # ----------------------------------------------------------------
    # Step 5: Build model summary
    # ----------------------------------------------------------------
    print("\nSTEP 5: Building model summary")
    print("-" * 70)
    build_model_summary(judgment_df, matrix_df)

    # ----------------------------------------------------------------
    # Step 6: Save question metadata
    # ----------------------------------------------------------------
    print("\nSTEP 6: Saving question metadata")
    print("-" * 70)
    save_question_metadata(question_meta_df, judgment_df)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  PRIMARY response matrix:")
    print(f"    Dimensions: {matrix_df.shape[0]} models x {matrix_df.shape[1]} questions")
    n_valid = matrix_df.notna().sum().sum()
    total = matrix_df.shape[0] * matrix_df.shape[1]
    print(f"    Fill rate:  {n_valid/total*100:.1f}%")
    print(f"    Score type: Continuous [0, 1]")
    print(f"    Categories: {', '.join(JUDGMENT_CATEGORIES)}")

    print(f"\n  COVERAGE NOTE:")
    print(f"    The model_judgment HF dataset covers 3 of 6 LiveBench categories.")
    print(f"    Available: {', '.join(JUDGMENT_CATEGORIES)} ({matrix_df.shape[1]} questions)")
    print(f"    Missing:   {', '.join(MISSING_CATEGORIES)}")
    print(f"    To get scores for missing categories, run the LiveBench")
    print(f"    gen_ground_truth_judgment.py scoring pipeline on model_answer data.")
    print(f"    See: https://github.com/LiveBench/LiveBench")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")

    print(f"\n  Score interpretation:")
    print(f"    Continuous: 0.0 (wrong) to 1.0 (perfect)")
    print(f"    Binary tasks (coding, typos): only 0 or 1")
    print(f"    Partial credit tasks (connections, plot_unscrambling, etc.): fractional values")
    print(f"    Binary matrix: threshold at >= 0.5")


if __name__ == "__main__":
    main()
