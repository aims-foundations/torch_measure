"""
Build BiGGen-Bench response matrices from multi-judge evaluation data.

Data source:
  - prometheus-eval/BiGGen-Bench-Results on HuggingFace
  - 99 models x 695 items x 5 judges, scores 1-5

Processing:
  1. Download dataset from HuggingFace
  2. Build per-judge response matrices (models x items, scores 1-5)
  3. Build combined multi-judge response matrix (mean across judges)

Outputs:
  - processed/response_matrix.csv: Combined (mean across judges), models x items
  - processed/response_matrix_judge_<name>.csv: Per-judge matrices
  - processed/judge_agreement.csv: Inter-judge agreement statistics
"""

import os
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_data():
    """Download BiGGen-Bench-Results from HuggingFace."""
    cache_path = os.path.join(RAW_DIR, "biggen_results.parquet")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached data: {cache_path}")
        return pd.read_parquet(cache_path)

    print("  Downloading prometheus-eval/BiGGen-Bench-Results from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "prometheus-eval/BiGGen-Bench-Results",
        token=os.environ.get("HF_TOKEN"),
    )

    # The dataset may have multiple splits; combine them
    all_dfs = []
    for split_name in ds:
        split_df = ds[split_name].to_pandas()
        split_df["_split"] = split_name
        all_dfs.append(split_df)
        print(f"    Split '{split_name}': {len(split_df):,} rows, columns: {list(split_df.columns)}")

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(cache_path)
    print(f"  Cached: {cache_path} ({len(df):,} rows)")

    return df


def extract_scores(df):
    """Extract model, item_id, judge, score from the dataset."""
    print("\n  Extracting scores from dataset...")
    print(f"  Available columns: {list(df.columns)}")

    records = []

    # Try to identify column structure
    # BiGGen-Bench-Results typically has: model, instance_id/item_id, judge_model, score
    model_col = None
    item_col = None
    judge_col = None
    score_col = None

    for col in df.columns:
        cl = col.lower()
        if cl in ("model", "model_id", "model_name"):
            model_col = col
        elif cl in ("instance_id", "item_id", "question_id", "id", "task_id"):
            item_col = col
        elif cl in ("judge", "judge_model", "evaluator", "judge_name"):
            judge_col = col
        elif cl in ("score", "rating", "eval_score"):
            score_col = col

    # If direct columns found, use them
    if model_col and item_col and score_col:
        print(f"  Using columns: model={model_col}, item={item_col}, score={score_col}")
        if judge_col:
            print(f"    judge={judge_col}")

        for _, row in df.iterrows():
            model = str(row[model_col])
            item_id = str(row[item_col])
            score = row[score_col]
            judge = str(row[judge_col]) if judge_col else "default"

            try:
                score = float(score)
                if not np.isnan(score):
                    records.append({
                        "model": model,
                        "item_id": item_id,
                        "judge": judge,
                        "score": score,
                    })
            except (ValueError, TypeError):
                pass

    else:
        # Try to parse nested structure
        # BiGGen may have results nested in JSON columns
        print("  Trying to parse nested structure...")

        for _, row in df.iterrows():
            row_dict = row.to_dict()

            # Try to find model/item/judge/score in various nested formats
            model = ""
            item_id = ""

            for key in ["model", "model_id", "model_name"]:
                if key in row_dict and row_dict[key]:
                    model = str(row_dict[key])
                    break

            for key in ["instance_id", "item_id", "question_id", "id"]:
                if key in row_dict and row_dict[key]:
                    item_id = str(row_dict[key])
                    break

            if not model or not item_id:
                continue

            # Look for judge-score pairs
            for key, val in row_dict.items():
                if isinstance(val, dict) and "score" in val:
                    judge = key
                    try:
                        score = float(val["score"])
                        records.append({
                            "model": model,
                            "item_id": item_id,
                            "judge": judge,
                            "score": score,
                        })
                    except (ValueError, TypeError):
                        pass
                elif "score" in key.lower() or "rating" in key.lower():
                    judge = key.replace("_score", "").replace("_rating", "")
                    try:
                        score = float(val)
                        if not np.isnan(score):
                            records.append({
                                "model": model,
                                "item_id": item_id,
                                "judge": judge,
                                "score": score,
                            })
                    except (ValueError, TypeError):
                        pass

    print(f"  Extracted {len(records):,} score records")
    scores_df = pd.DataFrame(records)

    if len(scores_df) > 0:
        print(f"  Unique models: {scores_df['model'].nunique()}")
        print(f"  Unique items:  {scores_df['item_id'].nunique()}")
        print(f"  Unique judges: {scores_df['judge'].nunique()}")
        print(f"  Judges: {sorted(scores_df['judge'].unique())}")

    return scores_df


def build_response_matrices(scores_df):
    """Build per-judge and combined response matrices."""
    print("\nBuilding response matrices...")

    judges = sorted(scores_df["judge"].unique())
    judge_matrices = {}

    for judge in judges:
        judge_df = scores_df[scores_df["judge"] == judge]
        matrix = judge_df.pivot_table(
            index="model",
            columns="item_id",
            values="score",
            aggfunc="mean",
        )
        matrix.index.name = "Model"
        judge_matrices[judge] = matrix

        # Save per-judge matrix
        safe_name = judge.replace("/", "_").replace(" ", "_").replace(".", "_")
        output_path = os.path.join(PROCESSED_DIR, f"response_matrix_judge_{safe_name}.csv")
        matrix.to_csv(output_path)
        n_m, n_i = matrix.shape
        print(f"  Judge '{judge}': {n_m} models x {n_i} items -> {output_path}")

    # Build combined matrix (mean across judges)
    print("\n  Building combined multi-judge matrix...")
    combined_scores = scores_df.groupby(["model", "item_id"])["score"].mean().reset_index()
    combined_matrix = combined_scores.pivot_table(
        index="model",
        columns="item_id",
        values="score",
        aggfunc="mean",
    )
    combined_matrix.index.name = "Model"

    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    combined_matrix.to_csv(output_path)
    print(f"  Combined: {combined_matrix.shape[0]} models x {combined_matrix.shape[1]} items -> {output_path}")

    # Build judge agreement statistics
    if len(judges) > 1:
        build_judge_agreement(scores_df, judges)

    return combined_matrix, judge_matrices


def build_judge_agreement(scores_df, judges):
    """Compute inter-judge agreement statistics."""
    print("\n  Computing inter-judge agreement...")

    # For each (model, item) pair, compute agreement across judges
    pivot = scores_df.pivot_table(
        index=["model", "item_id"],
        columns="judge",
        values="score",
        aggfunc="mean",
    )

    # Pairwise correlations between judges
    corr_matrix = pivot.corr()

    rows = []
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            if j1 in corr_matrix.columns and j2 in corr_matrix.columns:
                corr_val = corr_matrix.loc[j1, j2]
                # Mean absolute difference
                mask = pivot[[j1, j2]].dropna()
                mad = (mask[j1] - mask[j2]).abs().mean() if len(mask) > 0 else np.nan
                rows.append({
                    "judge_1": j1,
                    "judge_2": j2,
                    "correlation": corr_val,
                    "mean_abs_diff": mad,
                    "n_overlap": len(mask),
                })

    agreement_df = pd.DataFrame(rows)
    output_path = os.path.join(PROCESSED_DIR, "judge_agreement.csv")
    agreement_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    if len(agreement_df) > 0:
        print(f"  Mean pairwise correlation: {agreement_df['correlation'].mean():.3f}")
        print(f"  Mean absolute difference:  {agreement_df['mean_abs_diff'].mean():.3f}")


def print_statistics(scores_df, combined_matrix, judge_matrices):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  BIGGEN-BENCH STATISTICS")
    print(f"{'='*60}")

    n_models, n_items = combined_matrix.shape
    total_cells = n_models * n_items
    n_valid = combined_matrix.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Combined matrix dimensions:")
    print(f"    Models:        {n_models}")
    print(f"    Items:         {n_items}")
    print(f"    Judges:        {len(judge_matrices)}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"    Fill rate:     {fill_rate*100:.1f}%")

    # Per-judge matrix sizes
    print(f"\n  Per-judge matrix sizes:")
    for judge, matrix in judge_matrices.items():
        n_m, n_i = matrix.shape
        fill = matrix.notna().sum().sum() / (n_m * n_i) * 100
        print(f"    {judge:40s}  {n_m:3d} x {n_i:3d}  fill={fill:.1f}%")

    # Score distribution
    all_scores = combined_matrix.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Combined score distribution (1-5 scale):")
        print(f"    Mean:   {np.mean(valid_scores):.3f}")
        print(f"    Median: {np.median(valid_scores):.3f}")
        print(f"    Std:    {np.std(valid_scores):.3f}")
        print(f"    Min:    {np.min(valid_scores):.3f}")
        print(f"    Max:    {np.max(valid_scores):.3f}")

        # Histogram
        print(f"\n  Score histogram:")
        for score_val in range(1, 6):
            count = np.sum(
                (valid_scores >= score_val - 0.5) & (valid_scores < score_val + 0.5)
            )
            pct = count / len(valid_scores) * 100
            bar = "#" * int(pct)
            print(f"    {score_val}: {count:8,} ({pct:5.1f}%) {bar}")

    # Per-model stats
    per_model_mean = combined_matrix.mean(axis=1).sort_values(ascending=False)
    print(f"\n  Per-model mean score (combined):")
    print(f"    Best:   {per_model_mean.iloc[0]:.3f} ({per_model_mean.index[0]})")
    print(f"    Worst:  {per_model_mean.iloc[-1]:.3f} ({per_model_mean.index[-1]})")
    print(f"    Median: {per_model_mean.median():.3f}")
    print(f"    Std:    {per_model_mean.std():.3f}")

    print(f"\n  Top 15 models:")
    for model, score in per_model_mean.head(15).items():
        print(f"    {model:50s}  {score:.3f}")

    print(f"\n  Bottom 5 models:")
    for model, score in per_model_mean.tail(5).items():
        print(f"    {model:50s}  {score:.3f}")

    # Per-item difficulty
    per_item_mean = combined_matrix.mean(axis=0)
    print(f"\n  Per-item difficulty:")
    print(f"    Easiest: {per_item_mean.max():.3f}")
    print(f"    Hardest: {per_item_mean.min():.3f}")
    print(f"    Median:  {per_item_mean.median():.3f}")
    print(f"    Std:     {per_item_mean.std():.3f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:55s}  {size_kb:.1f} KB")


def main():
    print("BiGGen-Bench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download
    print("STEP 1: Downloading BiGGen-Bench-Results")
    print("-" * 60)
    df = download_data()

    # Step 2: Extract scores
    print("\nSTEP 2: Extracting scores")
    print("-" * 60)
    scores_df = extract_scores(df)

    if len(scores_df) == 0:
        print("  ERROR: No scores extracted. Check dataset format.")
        print(f"  Dataset columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].to_dict()}")
        sys.exit(1)

    # Step 3: Build matrices
    print("\nSTEP 3: Building response matrices")
    print("-" * 60)
    combined_matrix, judge_matrices = build_response_matrices(scores_df)

    # Step 4: Statistics
    print("\nSTEP 4: Detailed statistics")
    print("-" * 60)
    print_statistics(scores_df, combined_matrix, judge_matrices)


if __name__ == "__main__":
    main()
