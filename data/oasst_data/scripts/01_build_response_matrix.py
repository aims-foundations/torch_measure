"""
Build OpenAssistant OASST1 response matrix from human-ranked conversation data.

Data source:
  - OpenAssistant/oasst1 on HuggingFace
  - ~161K messages in conversation trees with ~461K quality ratings
  - Multiple assistant responses per prompt are ranked by human annotators

Processing:
  1. Load dataset from HuggingFace
  2. Index messages by message_id, identify parent-child relationships
  3. Group assistant responses by parent (prompt) message
  4. Extract human-assigned ranks for alternative responses
  5. Normalize ranks to [0,1] where 1 = best (rank 0) and 0 = worst
  6. Build response_matrix.csv (rank tiers x prompts)

Rank normalization:
  - normalized_score = 1.0 - rank / max_rank_in_group
  - rank 0 (best) -> 1.0, rank N (worst) -> 0.0

Outputs:
  - raw/extracted_ranks.csv: Cached per-message rank data
  - processed/response_matrix.csv: Rank tiers (rows) x prompts (columns), normalized scores [0,1]
  - processed/prompt_metadata.csv: Per-prompt metadata (language, n_alternatives, tree_id)
"""

import os
import sys
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_and_extract():
    """Load OASST1 dataset and extract ranked assistant responses."""
    cache_path = os.path.join(RAW_DIR, "extracted_ranks.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached extraction: {cache_path}")
        return pd.read_csv(cache_path)

    print("  Loading OpenAssistant/oasst1 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "OpenAssistant/oasst1",
        split="train+validation",
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"  Total messages: {len(ds):,}")

    # Index all messages by message_id
    messages = {}
    for item in ds:
        msg_id = item["message_id"]
        messages[msg_id] = {
            "message_id": msg_id,
            "parent_id": item.get("parent_id"),
            "role": item.get("role"),
            "rank": item.get("rank"),
            "lang": item.get("lang", ""),
            "message_tree_id": item.get("message_tree_id"),
        }

    # Group assistant responses by parent prompt
    prompt_groups = defaultdict(list)
    for msg_id, msg in messages.items():
        if msg["role"] == "assistant" and msg["rank"] is not None and msg["parent_id"] is not None:
            prompt_groups[msg["parent_id"]].append(msg)

    # Extract records for groups with >= 2 alternatives
    records = []
    for parent_id, assistants in prompt_groups.items():
        if len(assistants) < 2:
            continue

        max_rank = max(int(a["rank"]) for a in assistants)
        n_alts = len(assistants)

        for a in assistants:
            rank = int(a["rank"])
            if max_rank > 0:
                normalized = 1.0 - rank / max_rank
            else:
                normalized = 1.0

            records.append({
                "parent_id": parent_id,
                "message_id": a["message_id"],
                "rank": rank,
                "n_alternatives": n_alts,
                "max_rank": max_rank,
                "normalized_score": normalized,
                "lang": a["lang"],
                "message_tree_id": a["message_tree_id"],
            })

    print(f"  Prompt groups with >= 2 alternatives: {len(set(r['parent_id'] for r in records)):,}")
    print(f"  Total rank records: {len(records):,}")

    ranks_df = pd.DataFrame(records)
    ranks_df.to_csv(cache_path, index=False)
    print(f"  Cached extraction: {cache_path}")

    return ranks_df


def build_response_matrix(ranks_df):
    """Build response matrix from extracted ranks.

    Rows = rank tiers (rank_0 = best, rank_1 = second, etc.)
    Columns = prompts (parent_id)
    Values = normalized score [0, 1]
    """
    print("\nBuilding response matrix...")

    # Determine max rank tier
    max_rank = int(ranks_df["rank"].max())
    n_subjects = max_rank + 1
    rank_labels = [f"rank_{i}" for i in range(n_subjects)]

    # Get sorted prompt IDs
    prompt_ids = sorted(ranks_df["parent_id"].unique())
    n_prompts = len(prompt_ids)

    print(f"  Rank tiers: {n_subjects}")
    print(f"  Prompts: {n_prompts:,}")

    # Build matrix using pivot
    # Each row in ranks_df maps a rank tier to a prompt
    ranks_df["rank_label"] = "rank_" + ranks_df["rank"].astype(str)

    matrix_df = ranks_df.pivot_table(
        index="rank_label",
        columns="parent_id",
        values="normalized_score",
        aggfunc="mean",
    )

    # Ensure all rank tiers are present as rows
    for label in rank_labels:
        if label not in matrix_df.index:
            matrix_df.loc[label] = np.nan

    # Sort rows by rank tier
    matrix_df = matrix_df.loc[[l for l in rank_labels if l in matrix_df.index]]
    matrix_df.index.name = "RankTier"

    n_rows, n_cols = matrix_df.shape
    print(f"  Matrix: {n_rows} rank tiers x {n_cols:,} prompts")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def build_prompt_metadata(ranks_df):
    """Build per-prompt metadata."""
    print("\nBuilding prompt metadata...")

    meta = ranks_df.groupby("parent_id").agg(
        n_alternatives=("message_id", "count"),
        max_rank=("max_rank", "first"),
        lang=("lang", "first"),
        message_tree_id=("message_tree_id", "first"),
        mean_score=("normalized_score", "mean"),
        std_score=("normalized_score", "std"),
    ).reset_index()

    meta_path = os.path.join(PROCESSED_DIR, "prompt_metadata.csv")
    meta.to_csv(meta_path, index=False)
    print(f"  Saved {len(meta):,} prompts to {meta_path}")

    return meta


def print_statistics(ranks_df, matrix_df, meta_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  OASST1 STATISTICS")
    print(f"{'='*60}")

    n_rows, n_cols = matrix_df.shape
    total_cells = n_rows * n_cols
    n_valid = matrix_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Matrix dimensions:")
    print(f"    Rank tiers:    {n_rows}")
    print(f"    Prompts:       {n_cols:,}")
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

    # Rank distribution
    print(f"\n  Raw rank distribution:")
    rank_counts = ranks_df["rank"].value_counts().sort_index()
    for rank_val, count in rank_counts.items():
        pct = count / len(ranks_df) * 100
        print(f"    Rank {rank_val}: {count:8,} ({pct:5.1f}%)")

    # Alternatives per prompt
    print(f"\n  Alternatives per prompt:")
    alt_stats = ranks_df.groupby("parent_id")["message_id"].count()
    print(f"    Mean:   {alt_stats.mean():.1f}")
    print(f"    Median: {alt_stats.median():.0f}")
    print(f"    Min:    {alt_stats.min()}")
    print(f"    Max:    {alt_stats.max()}")

    alt_counts = Counter(alt_stats.values)
    print(f"\n  Distribution of alternatives per prompt:")
    for n_alts in sorted(alt_counts.keys()):
        count = alt_counts[n_alts]
        pct = count / len(alt_stats) * 100
        print(f"    {n_alts} alternatives: {count:,} prompts ({pct:.1f}%)")

    # Language distribution
    if "lang" in ranks_df.columns:
        lang_counts = ranks_df.groupby("parent_id")["lang"].first().value_counts()
        print(f"\n  Language distribution (top 10):")
        for lang, count in lang_counts.head(10).items():
            pct = count / len(alt_stats) * 100
            print(f"    {lang:10s}: {count:,} prompts ({pct:.1f}%)")

    # Per-rank-tier stats
    per_tier_mean = matrix_df.mean(axis=1)
    per_tier_coverage = matrix_df.notna().sum(axis=1)
    print(f"\n  Per-rank-tier statistics:")
    for tier in matrix_df.index:
        mean_val = per_tier_mean.get(tier, float("nan"))
        coverage = per_tier_coverage.get(tier, 0)
        print(f"    {tier:15s}  mean_score={mean_val:.3f}  coverage={coverage:,}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("OASST1 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Load and extract
    print("STEP 1: Loading OASST1 dataset and extracting ranks")
    print("-" * 60)
    ranks_df = load_and_extract()

    # Step 2: Build matrix
    print("\nSTEP 2: Building response matrix")
    print("-" * 60)
    matrix_df = build_response_matrix(ranks_df)

    # Step 3: Build metadata
    print("\nSTEP 3: Building prompt metadata")
    print("-" * 60)
    meta_df = build_prompt_metadata(ranks_df)

    # Step 4: Statistics
    print("\nSTEP 4: Detailed statistics")
    print("-" * 60)
    print_statistics(ranks_df, matrix_df, meta_df)


if __name__ == "__main__":
    main()
