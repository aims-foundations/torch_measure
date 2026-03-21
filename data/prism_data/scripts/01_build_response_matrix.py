"""
Build PRISM alignment response matrices from per-participant utterance ratings.

Data source:
  - HannahRoseKirk/prism-alignment on HuggingFace
  - utterances config: ~68K rated utterances from ~1,500 participants
  - survey config: participant demographics (75 countries)

PRISM captures diverse human preferences for LLM outputs across demographics.
Each participant rates multiple LLM responses on a 1-100 cardinal sliding
scale, and selects a preferred (chosen) response at each turn.

Processing:
  1. Download utterances and survey configs from HuggingFace
  2. Build response matrix: participants (rows) x utterances (columns)
  3. Two value types:
     - Continuous: score normalized from 1-100 to [0, 1]
     - Binary: if_chosen (1 = chosen, 0 = not chosen)
  4. Merge participant demographics from survey config

Outputs:
  - raw/utterances_raw.parquet: Cached utterance data
  - raw/survey_raw.parquet: Cached survey/demographics data
  - processed/response_matrix_scores.csv: Participants x utterances, continuous [0,1]
  - processed/response_matrix_chosen.csv: Participants x utterances, binary {0,1}
  - processed/participant_metadata.csv: Per-participant demographics
  - processed/utterance_metadata.csv: Per-utterance metadata (model, turn, etc.)
"""

import os
import sys

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SRC_REPO = "HannahRoseKirk/prism-alignment"


def download_utterances():
    """Download the utterances config from HuggingFace."""
    cache_path = os.path.join(RAW_DIR, "utterances_raw.parquet")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached data: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Downloading {SRC_REPO} (utterances config) from HuggingFace ...")
    from datasets import load_dataset

    ds = load_dataset(
        SRC_REPO,
        "utterances",
        split="train",
        token=HF_TOKEN or None,
    )

    df = ds.to_pandas()
    df.to_parquet(cache_path)
    print(f"  Cached: {cache_path} ({len(df):,} rows)")
    print(f"  Columns: {list(df.columns)}")

    return df


def download_survey():
    """Download the survey config (participant demographics) from HuggingFace."""
    cache_path = os.path.join(RAW_DIR, "survey_raw.parquet")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached data: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Downloading {SRC_REPO} (survey config) from HuggingFace ...")
    from datasets import load_dataset

    ds = load_dataset(
        SRC_REPO,
        "survey",
        split="train",
        token=HF_TOKEN or None,
    )

    df = ds.to_pandas()
    df.to_parquet(cache_path)
    print(f"  Cached: {cache_path} ({len(df):,} rows)")
    print(f"  Columns: {list(df.columns)}")

    return df


def build_response_matrices(utt_df):
    """Build response matrices from utterance-level data.

    Returns two DataFrames (scores, chosen) with participants as rows
    and utterances as columns.
    """
    print("\nBuilding response matrices ...")

    # Extract needed columns
    records = utt_df[["user_id", "utterance_id", "score", "if_chosen"]].copy()

    # Unique participants and utterances (sorted for determinism)
    participant_ids = sorted(records["user_id"].unique())
    utterance_ids = sorted(records["utterance_id"].unique())

    n_participants = len(participant_ids)
    n_utterances = len(utterance_ids)

    print(f"  Unique participants: {n_participants}")
    print(f"  Unique utterances:   {n_utterances}")
    print(f"  Total records:       {len(records):,}")

    # --- Scores matrix (continuous, normalized 1-100 -> [0,1]) ---
    print("\n  Building scores matrix ...")
    scores_pivot = records.pivot_table(
        index="user_id",
        columns="utterance_id",
        values="score",
        aggfunc="first",
    )
    # Reindex to ensure consistent ordering
    scores_pivot = scores_pivot.reindex(
        index=participant_ids, columns=utterance_ids
    )
    # Normalize from 1-100 to [0, 1]
    scores_pivot = (scores_pivot - 1.0) / 99.0
    scores_pivot.index.name = "participant"

    n_valid_scores = scores_pivot.notna().sum().sum()
    total_cells = n_participants * n_utterances
    print(f"    Shape: {scores_pivot.shape}")
    print(f"    Valid cells: {n_valid_scores:,} / {total_cells:,} "
          f"({n_valid_scores / total_cells * 100:.2f}%)")

    # --- Chosen matrix (binary) ---
    print("\n  Building chosen matrix ...")
    chosen_pivot = records.pivot_table(
        index="user_id",
        columns="utterance_id",
        values="if_chosen",
        aggfunc="first",
    )
    chosen_pivot = chosen_pivot.reindex(
        index=participant_ids, columns=utterance_ids
    )
    # Convert bool to float
    chosen_pivot = chosen_pivot.astype(float)
    chosen_pivot.index.name = "participant"

    n_valid_chosen = chosen_pivot.notna().sum().sum()
    print(f"    Shape: {chosen_pivot.shape}")
    print(f"    Valid cells: {n_valid_chosen:,} / {total_cells:,} "
          f"({n_valid_chosen / total_cells * 100:.2f}%)")

    # Save
    scores_path = os.path.join(PROCESSED_DIR, "response_matrix_scores.csv")
    chosen_path = os.path.join(PROCESSED_DIR, "response_matrix_chosen.csv")
    scores_pivot.to_csv(scores_path)
    chosen_pivot.to_csv(chosen_path)
    print(f"\n  Saved: {scores_path}")
    print(f"  Saved: {chosen_path}")

    return scores_pivot, chosen_pivot, participant_ids, utterance_ids


def build_utterance_metadata(utt_df, utterance_ids):
    """Build per-utterance metadata."""
    print("\nBuilding utterance metadata ...")

    # Take first occurrence of each utterance_id for metadata
    meta_cols = [
        "utterance_id", "conversation_id", "user_id",
        "turn", "within_turn_id",
        "model_name", "model_provider",
        "user_prompt",
    ]
    available_cols = [c for c in meta_cols if c in utt_df.columns]
    meta_df = utt_df[available_cols].drop_duplicates(
        subset="utterance_id", keep="first"
    )
    meta_df = meta_df.set_index("utterance_id").reindex(utterance_ids).reset_index()

    meta_path = os.path.join(PROCESSED_DIR, "utterance_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path} ({len(meta_df):,} utterances)")

    # Model distribution
    if "model_name" in meta_df.columns:
        print(f"\n  Model distribution:")
        for model, count in meta_df["model_name"].value_counts().head(25).items():
            print(f"    {model:40s}  {count:,}")

    return meta_df


def build_participant_metadata(survey_df, participant_ids):
    """Build per-participant metadata from survey data."""
    print("\nBuilding participant metadata ...")

    demo_cols = [
        "user_id", "age", "gender", "education", "employment_status",
        "marital_status", "english_proficiency",
        "birth_country", "birth_region", "birth_subregion",
        "reside_country", "reside_region", "reside_subregion",
        "lm_familiarity", "lm_frequency_use",
    ]
    available_cols = [c for c in demo_cols if c in survey_df.columns]
    demo_df = survey_df[available_cols].copy()
    demo_df = demo_df.set_index("user_id").reindex(participant_ids).reset_index()
    demo_df = demo_df.rename(columns={"index": "user_id"})

    meta_path = os.path.join(PROCESSED_DIR, "participant_metadata.csv")
    demo_df.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path} ({len(demo_df):,} participants)")

    # Demographics summary
    n_matched = demo_df.dropna(subset=["age"]).shape[0] if "age" in demo_df.columns else 0
    print(f"  Matched survey data: {n_matched}/{len(participant_ids)} participants")

    if "birth_country" in demo_df.columns:
        n_countries = demo_df["birth_country"].dropna().nunique()
        print(f"  Birth countries: {n_countries}")
        print(f"\n  Top birth countries:")
        for country, count in demo_df["birth_country"].value_counts().head(15).items():
            print(f"    {country:30s}  {count:,}")

    if "gender" in demo_df.columns:
        print(f"\n  Gender distribution:")
        for gender, count in demo_df["gender"].value_counts().items():
            print(f"    {gender:20s}  {count:,}")

    return demo_df


def print_statistics(scores_df, chosen_df):
    """Print detailed statistics about the response matrices."""
    print(f"\n{'=' * 60}")
    print(f"  PRISM RESPONSE MATRIX STATISTICS")
    print(f"{'=' * 60}")

    n_participants, n_utterances = scores_df.shape
    total_cells = n_participants * n_utterances

    # Scores statistics
    n_valid = scores_df.notna().sum().sum()
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Matrix dimensions:")
    print(f"    Participants:  {n_participants}")
    print(f"    Utterances:    {n_utterances:,}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({fill_rate * 100:.2f}%)")
    print(f"    Fill rate:     {fill_rate * 100:.2f}%")

    # Score distribution (after normalization to [0,1])
    all_scores = scores_df.values.flatten()
    valid_scores = all_scores[~np.isnan(all_scores)]
    if len(valid_scores) > 0:
        print(f"\n  Score distribution (normalized [0,1]):")
        print(f"    Mean:   {np.mean(valid_scores):.4f}")
        print(f"    Median: {np.median(valid_scores):.4f}")
        print(f"    Std:    {np.std(valid_scores):.4f}")
        print(f"    Min:    {np.min(valid_scores):.4f}")
        print(f"    Max:    {np.max(valid_scores):.4f}")

        # Quartile distribution
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            val = np.quantile(valid_scores, q)
            print(f"    P{int(q * 100):02d}:    {val:.4f}")

    # Chosen distribution
    all_chosen = chosen_df.values.flatten()
    valid_chosen = all_chosen[~np.isnan(all_chosen)]
    if len(valid_chosen) > 0:
        n_chosen = np.sum(valid_chosen == 1.0)
        n_not_chosen = np.sum(valid_chosen == 0.0)
        print(f"\n  Chosen distribution:")
        print(f"    Chosen (1):     {int(n_chosen):,} ({n_chosen / len(valid_chosen) * 100:.1f}%)")
        print(f"    Not chosen (0): {int(n_not_chosen):,} ({n_not_chosen / len(valid_chosen) * 100:.1f}%)")

    # Per-participant statistics
    per_participant_mean = scores_df.mean(axis=1)
    per_participant_count = scores_df.notna().sum(axis=1)

    print(f"\n  Per-participant statistics:")
    print(f"    Ratings per participant:")
    print(f"      Mean:   {per_participant_count.mean():.1f}")
    print(f"      Median: {per_participant_count.median():.1f}")
    print(f"      Min:    {per_participant_count.min()}")
    print(f"      Max:    {per_participant_count.max()}")
    print(f"    Mean score per participant:")
    print(f"      Mean:   {per_participant_mean.mean():.4f}")
    print(f"      Std:    {per_participant_mean.std():.4f}")

    # Per-utterance statistics
    per_utt_mean = scores_df.mean(axis=0)
    per_utt_count = scores_df.notna().sum(axis=0)

    print(f"\n  Per-utterance statistics:")
    print(f"    Raters per utterance:")
    print(f"      Mean:   {per_utt_count.mean():.1f}")
    print(f"      Median: {per_utt_count.median():.1f}")
    print(f"      Min:    {per_utt_count.min()}")
    print(f"      Max:    {per_utt_count.max()}")

    # Inter-participant agreement (sample to avoid memory issues)
    if n_participants > 2:
        print(f"\n  Inter-participant agreement (sampled):")
        sample_size = min(100, n_participants)
        sample_df = scores_df.iloc[:sample_size]
        corr_matrix = sample_df.T.corr()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        corr_vals = upper_tri.values.flatten()
        corr_vals = corr_vals[~np.isnan(corr_vals)]
        if len(corr_vals) > 0:
            print(f"    Mean correlation:   {np.mean(corr_vals):.4f}")
            print(f"    Median correlation: {np.median(corr_vals):.4f}")
            print(f"    Std correlation:    {np.std(corr_vals):.4f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("PRISM Alignment Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download utterance data
    print("STEP 1: Downloading PRISM utterances")
    print("-" * 60)
    utt_df = download_utterances()

    # Step 2: Download survey data
    print("\nSTEP 2: Downloading PRISM survey (demographics)")
    print("-" * 60)
    survey_df = download_survey()

    # Step 3: Build response matrices
    print("\nSTEP 3: Building response matrices")
    print("-" * 60)
    scores_df, chosen_df, participant_ids, utterance_ids = build_response_matrices(utt_df)

    # Step 4: Build utterance metadata
    print("\nSTEP 4: Building utterance metadata")
    print("-" * 60)
    utt_meta_df = build_utterance_metadata(utt_df, utterance_ids)

    # Step 5: Build participant metadata
    print("\nSTEP 5: Building participant metadata")
    print("-" * 60)
    participant_meta_df = build_participant_metadata(survey_df, participant_ids)

    # Step 6: Statistics
    print("\nSTEP 6: Detailed statistics")
    print("-" * 60)
    print_statistics(scores_df, chosen_df)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    n_p, n_u = scores_df.shape
    print(f"  Scores matrix:  {n_p} participants x {n_u:,} utterances")
    print(f"  Chosen matrix:  {n_p} participants x {n_u:,} utterances")
    print(f"  Participant metadata: {len(participant_meta_df)} entries")
    print(f"  Utterance metadata:   {len(utt_meta_df):,} entries")


if __name__ == "__main__":
    main()
