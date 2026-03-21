"""
Build Nectar response matrix from GPT-4 ranking data.

Data source:
  - berkeley-nest/Nectar on HuggingFace
  - ~40 models x ~183K prompts, GPT-4 rankings (1-7, lower is better)
  - Each prompt has multiple model responses ranked by GPT-4

Processing:
  1. Stream dataset from HuggingFace (large dataset)
  2. Extract per-model per-prompt ranks
  3. Normalize ranks to [0,1] where 1 = best (rank 1) and 0 = worst
  4. Build response_matrix.csv (models x prompts)

Rank normalization:
  - normalized_score = 1 - (rank - 1) / (n_ranked - 1)
  - rank 1 -> 1.0, rank n -> 0.0
  - If only 1 response: score = 1.0

Outputs:
  - raw/extracted_ranks.csv: Cached per-model per-prompt ranks
  - processed/response_matrix.csv: Models (rows) x prompts (columns), normalized scores [0,1]
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


def stream_and_extract():
    """Stream Nectar dataset and extract per-model per-prompt ranks."""
    cache_path = os.path.join(RAW_DIR, "extracted_ranks.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached extraction: {cache_path}")
        return pd.read_csv(cache_path)

    print("  Streaming berkeley-nest/Nectar from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "berkeley-nest/Nectar",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    records = []
    n_processed = 0

    for item in ds:
        # Each item has a prompt and ranked answers from different models
        prompt_id = str(item.get("id", n_processed))
        if "prompt" in item and "id" not in item:
            # Use hash of prompt as ID
            prompt_id = str(hash(item["prompt"]) % (10**12))

        answers = item.get("answers", [])
        if not answers:
            n_processed += 1
            continue

        n_ranked = len(answers)

        for answer in answers:
            model = answer.get("model", "")
            rank = answer.get("rank", None)

            if not model or rank is None:
                continue

            try:
                rank = int(rank)
            except (ValueError, TypeError):
                continue

            # Normalize rank to [0, 1]
            if n_ranked > 1:
                normalized = 1.0 - (rank - 1) / (n_ranked - 1)
            else:
                normalized = 1.0

            records.append({
                "model": model,
                "prompt_id": prompt_id,
                "rank": rank,
                "n_ranked": n_ranked,
                "normalized_score": normalized,
            })

        n_processed += 1
        if n_processed % 10000 == 0:
            print(f"    Processed {n_processed:,} prompts, {len(records):,} ranks extracted...")

    print(f"  Total prompts processed: {n_processed:,}")
    print(f"  Total rank records: {len(records):,}")

    ranks_df = pd.DataFrame(records)
    ranks_df.to_csv(cache_path, index=False)
    print(f"  Cached extraction: {cache_path}")

    return ranks_df


def build_response_matrix(ranks_df):
    """Build response matrix from extracted ranks."""
    print("\nBuilding response matrix...")

    # Pivot to matrix using normalized scores
    matrix_df = ranks_df.pivot_table(
        index="model",
        columns="prompt_id",
        values="normalized_score",
        aggfunc="mean",
    )
    matrix_df.index.name = "Model"

    n_models, n_prompts = matrix_df.shape
    print(f"  Matrix: {n_models} models x {n_prompts:,} prompts")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def print_statistics(ranks_df, matrix_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  NECTAR STATISTICS")
    print(f"{'='*60}")

    n_models, n_prompts = matrix_df.shape
    total_cells = n_models * n_prompts
    n_valid = matrix_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Matrix dimensions:")
    print(f"    Models:        {n_models}")
    print(f"    Prompts:       {n_prompts:,}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"    Fill rate:     {fill_rate*100:.1f}%")

    # Normalized score distribution
    all_scores = matrix_df.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Normalized score distribution [0,1]:")
        print(f"    Mean:   {np.mean(valid_scores):.3f}")
        print(f"    Median: {np.median(valid_scores):.3f}")
        print(f"    Std:    {np.std(valid_scores):.3f}")
        print(f"    Min:    {np.min(valid_scores):.3f}")
        print(f"    Max:    {np.max(valid_scores):.3f}")

        # Histogram in 10 bins
        print(f"\n  Score histogram (10 bins):")
        hist, bin_edges = np.histogram(valid_scores, bins=10, range=(0, 1))
        for i in range(len(hist)):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            pct = hist[i] / len(valid_scores) * 100
            bar = "#" * int(pct)
            print(f"    [{lo:.1f}, {hi:.1f}): {hist[i]:8,} ({pct:5.1f}%) {bar}")

    # Raw rank distribution
    if "rank" in ranks_df.columns:
        print(f"\n  Raw rank distribution:")
        for rank_val in sorted(ranks_df["rank"].unique()):
            count = (ranks_df["rank"] == rank_val).sum()
            pct = count / len(ranks_df) * 100
            print(f"    Rank {rank_val}: {count:8,} ({pct:5.1f}%)")

    # Number of ranked responses per prompt
    if "n_ranked" in ranks_df.columns:
        print(f"\n  Responses per prompt:")
        n_ranked_stats = ranks_df.groupby("prompt_id")["n_ranked"].first()
        print(f"    Mean:   {n_ranked_stats.mean():.1f}")
        print(f"    Median: {n_ranked_stats.median():.0f}")
        print(f"    Min:    {n_ranked_stats.min()}")
        print(f"    Max:    {n_ranked_stats.max()}")

    # Per-model stats
    per_model_mean = matrix_df.mean(axis=1).sort_values(ascending=False)
    per_model_coverage = matrix_df.notna().sum(axis=1)

    print(f"\n  Per-model statistics:")
    print(f"    Mean normalized score range: [{per_model_mean.min():.3f}, {per_model_mean.max():.3f}]")

    print(f"\n  Top 15 models (by mean normalized score):")
    for model in per_model_mean.head(15).index:
        score = per_model_mean[model]
        coverage = per_model_coverage[model]
        print(f"    {model:45s}  score={score:.3f}  coverage={coverage:,}")

    print(f"\n  Bottom 5 models:")
    for model in per_model_mean.tail(5).index:
        score = per_model_mean[model]
        coverage = per_model_coverage[model]
        print(f"    {model:45s}  score={score:.3f}  coverage={coverage:,}")

    # Per-prompt difficulty
    per_prompt_mean = matrix_df.mean(axis=0)
    print(f"\n  Per-prompt difficulty (mean normalized score):")
    print(f"    Easiest: {per_prompt_mean.max():.3f}")
    print(f"    Hardest: {per_prompt_mean.min():.3f}")
    print(f"    Median:  {per_prompt_mean.median():.3f}")
    print(f"    Std:     {per_prompt_mean.std():.3f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("Nectar Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Stream and extract
    print("STEP 1: Streaming Nectar dataset and extracting ranks")
    print("-" * 60)
    ranks_df = stream_and_extract()

    # Step 2: Build matrix
    print("\nSTEP 2: Building response matrix")
    print("-" * 60)
    matrix_df = build_response_matrix(ranks_df)

    # Step 3: Statistics
    print("\nSTEP 3: Detailed statistics")
    print("-" * 60)
    print_statistics(ranks_df, matrix_df)


if __name__ == "__main__":
    main()
