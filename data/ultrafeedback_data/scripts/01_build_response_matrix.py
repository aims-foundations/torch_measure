"""
Build UltraFeedback response matrix from GPT-4 multi-aspect ratings.

Data source:
  - openbmb/UltraFeedback on HuggingFace
  - 17 models x ~64K prompts, GPT-4 multi-aspect ratings (1-5)
  - Each model response is rated on: instruction_following, honesty,
    truthfulness, helpfulness (each 1-5)

Processing:
  1. Stream dataset from HuggingFace (large: ~3.5GB)
  2. Extract per-model per-prompt mean scores across aspects
  3. Build response_matrix.csv (models x prompts)

Outputs:
  - raw/ultrafeedback_raw.parquet: Cached raw data (optional)
  - processed/response_matrix.csv: Models (rows) x prompts (columns), mean aspect scores 1-5
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
    """Stream UltraFeedback dataset and extract per-model per-prompt scores."""
    cache_path = os.path.join(RAW_DIR, "extracted_scores.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached extraction: {cache_path}")
        return pd.read_csv(cache_path)

    print("  Streaming openbmb/UltraFeedback from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "openbmb/UltraFeedback",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    records = []
    n_processed = 0
    n_skipped = 0

    for item in ds:
        prompt_id = item.get("source", "") + "_" + str(n_processed)
        # Use a stable prompt ID if available
        if "id" in item:
            prompt_id = str(item["id"])

        completions = item.get("completions", [])
        if not completions:
            n_skipped += 1
            continue

        for completion in completions:
            model = completion.get("model", "")
            if not model:
                continue

            # Extract annotations (GPT-4 ratings)
            annotations = completion.get("annotations", {})
            if not annotations:
                # Try alternative field names
                annotations = completion.get("ratings", {})

            aspect_scores = []
            for aspect_name, aspect_data in annotations.items():
                if isinstance(aspect_data, dict):
                    rating = aspect_data.get("Rating", aspect_data.get("rating", None))
                elif isinstance(aspect_data, (int, float)):
                    rating = aspect_data
                else:
                    continue

                if rating is not None:
                    try:
                        rating = float(rating)
                        if 1 <= rating <= 5:
                            aspect_scores.append(rating)
                    except (ValueError, TypeError):
                        pass

            if aspect_scores:
                mean_score = np.mean(aspect_scores)
                records.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "mean_score": mean_score,
                    "n_aspects": len(aspect_scores),
                })

        n_processed += 1
        if n_processed % 10000 == 0:
            print(f"    Processed {n_processed:,} prompts, {len(records):,} scores extracted...")

    print(f"  Total prompts processed: {n_processed:,}")
    print(f"  Prompts skipped (no completions): {n_skipped:,}")
    print(f"  Total score records: {len(records):,}")

    scores_df = pd.DataFrame(records)
    scores_df.to_csv(cache_path, index=False)
    print(f"  Cached extraction: {cache_path}")

    return scores_df


def build_response_matrix(scores_df):
    """Build response matrix from extracted scores."""
    print("\nBuilding response matrix...")

    # Pivot to matrix
    matrix_df = scores_df.pivot_table(
        index="model",
        columns="prompt_id",
        values="mean_score",
        aggfunc="mean",
    )
    matrix_df.index.name = "Model"

    n_models, n_prompts = matrix_df.shape
    print(f"  Raw matrix: {n_models} models x {n_prompts} prompts")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def print_statistics(scores_df, matrix_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  ULTRAFEEDBACK STATISTICS")
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

    # Score distribution
    all_scores = matrix_df.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Score distribution (mean aspect score, 1-5):")
        print(f"    Mean:   {np.mean(valid_scores):.3f}")
        print(f"    Median: {np.median(valid_scores):.3f}")
        print(f"    Std:    {np.std(valid_scores):.3f}")
        print(f"    Min:    {np.min(valid_scores):.3f}")
        print(f"    Max:    {np.max(valid_scores):.3f}")

        # Histogram by integer bins
        print(f"\n  Score histogram (rounded to integer):")
        for score_val in range(1, 6):
            count = np.sum(
                (valid_scores >= score_val - 0.5) & (valid_scores < score_val + 0.5)
            )
            pct = count / len(valid_scores) * 100
            bar = "#" * int(pct)
            print(f"    {score_val}: {count:8,} ({pct:5.1f}%) {bar}")

    # Per-model stats
    per_model_mean = matrix_df.mean(axis=1).sort_values(ascending=False)
    per_model_coverage = matrix_df.notna().sum(axis=1)

    print(f"\n  Per-model statistics:")
    print(f"    Mean scores range: [{per_model_mean.min():.3f}, {per_model_mean.max():.3f}]")
    print(f"    Coverage range:    [{per_model_coverage.min():,}, {per_model_coverage.max():,}]")

    print(f"\n  All models ranked by mean score:")
    for model in per_model_mean.index:
        score = per_model_mean[model]
        coverage = per_model_coverage[model]
        print(f"    {model:45s}  mean={score:.3f}  coverage={coverage:,}")

    # Per-prompt difficulty
    per_prompt_mean = matrix_df.mean(axis=0)
    print(f"\n  Per-prompt difficulty:")
    print(f"    Easiest: {per_prompt_mean.max():.3f}")
    print(f"    Hardest: {per_prompt_mean.min():.3f}")
    print(f"    Median:  {per_prompt_mean.median():.3f}")
    print(f"    Std:     {per_prompt_mean.std():.3f}")

    # Aspect count distribution
    if "n_aspects" in scores_df.columns:
        print(f"\n  Aspect count distribution:")
        for n, count in scores_df["n_aspects"].value_counts().sort_index().items():
            print(f"    {n} aspects: {count:,}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("UltraFeedback Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Stream and extract
    print("STEP 1: Streaming UltraFeedback dataset and extracting scores")
    print("-" * 60)
    scores_df = stream_and_extract()

    # Step 2: Build matrix
    print("\nSTEP 2: Building response matrix")
    print("-" * 60)
    matrix_df = build_response_matrix(scores_df)

    # Step 3: Statistics
    print("\nSTEP 3: Detailed statistics")
    print("-" * 60)
    print_statistics(scores_df, matrix_df)


if __name__ == "__main__":
    main()
