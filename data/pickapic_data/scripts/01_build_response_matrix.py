"""
Build Pick-a-Pic pairwise preference data from yuvalkirstain/pickapic_v2.

Data source:
  - yuvalkirstain/pickapic_v2_no_images on HuggingFace Hub: ~1M human
    preference comparisons over text-to-image model outputs.
  - Users submit text prompts, receive two generated images from different
    models, and indicate which image they prefer (or declare a tie).

  Key columns:
    caption:        Text prompt submitted by the user
    model_0:        Name of the model that generated image 0
    model_1:        Name of the model that generated image 1
    label_0:        Preference for image 0 (1.0=preferred, 0.0=not, 0.5=tie)
    label_1:        Preference for image 1 (1.0=preferred, 0.0=not, 0.5=tie)
    best_image_uid: UID of the preferred image (or "tie"/"")
    user_id:        Anonymous user identifier
    ranking_id:     Unique comparison identifier
    has_label:      Whether the comparison has a valid label

  Models in v2 (14 unique):
    Stable Diffusion 2.1, Dreamlike Photoreal 2.0, Stable Diffusion XL
    variants (Alpha, Beta, etc.), and others.

  Splits:
    train:             ~959K rows
    validation:        ~20.6K rows
    test:              ~20.7K rows
    test_unique:       500 rows
    validation_unique: 500 rows

Outputs:
  - processed/comparison_summary.csv:  Per-comparison metadata (without images)
  - processed/model_pair_stats.csv:    Win-rate matrix between model pairs
  - processed/model_summary.csv:       Per-model aggregate statistics
  - processed/prompt_stats.csv:        Per-prompt comparison counts
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

SRC_REPO = "yuvalkirstain/pickapic_v2_no_images"


def download_data():
    """Download Pick-a-Pic v2 (no images) training split from HuggingFace."""
    from datasets import load_dataset

    print("Downloading Pick-a-Pic v2 (no images) from HuggingFace ...")
    ds = load_dataset(SRC_REPO, split="train", token=HF_TOKEN)
    print(f"  Total rows: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")

    # Convert to DataFrame
    df = ds.to_pandas()
    return df


def filter_labeled(df):
    """Keep only rows with valid preference labels."""
    print("\nFiltering to labeled comparisons ...")
    n_before = len(df)
    df = df[df["has_label"] == True].copy()  # noqa: E712
    n_after = len(df)
    print(f"  Before: {n_before:,}, After: {n_after:,} ({n_before - n_after:,} dropped)")
    return df


def build_comparison_summary(df):
    """Build a summary DataFrame of all comparisons (without image data)."""
    print(f"\n{'='*60}")
    print("Building comparison summary ...")
    print(f"{'='*60}")

    n = len(df)
    print(f"  Total comparisons: {n:,}")

    # Preference distribution
    n_img0_wins = (df["label_0"] == 1.0).sum()
    n_img1_wins = (df["label_1"] == 1.0).sum()
    n_ties = ((df["label_0"] == 0.5) & (df["label_1"] == 0.5)).sum()
    n_other = n - n_img0_wins - n_img1_wins - n_ties

    print(f"\n  Preference distribution:")
    print(f"    Image 0 wins: {n_img0_wins:,} ({n_img0_wins/n*100:.1f}%)")
    print(f"    Image 1 wins: {n_img1_wins:,} ({n_img1_wins/n*100:.1f}%)")
    print(f"    Ties:         {n_ties:,} ({n_ties/n*100:.1f}%)")
    if n_other > 0:
        print(f"    Other:        {n_other:,} ({n_other/n*100:.1f}%)")

    # Unique entities
    n_users = df["user_id"].nunique()
    n_prompts = df["caption"].nunique()
    models_0 = set(df["model_0"].unique())
    models_1 = set(df["model_1"].unique())
    all_models = sorted(models_0 | models_1)

    print(f"\n  Unique users:   {n_users:,}")
    print(f"  Unique prompts: {n_prompts:,}")
    print(f"  Unique models:  {len(all_models)}")
    for m in all_models:
        n_as_0 = (df["model_0"] == m).sum()
        n_as_1 = (df["model_1"] == m).sum()
        print(f"    {m:25s}  as model_0: {n_as_0:,}  as model_1: {n_as_1:,}")

    # Caption length stats
    df["caption_len"] = df["caption"].str.len()
    print(f"\n  Caption length (chars):")
    print(f"    Mean:   {df['caption_len'].mean():.0f}")
    print(f"    Median: {df['caption_len'].median():.0f}")
    print(f"    Min:    {df['caption_len'].min()}")
    print(f"    Max:    {df['caption_len'].max()}")

    # Save summary CSV
    summary_cols = [
        "ranking_id", "user_id", "caption", "model_0", "model_1",
        "label_0", "label_1", "best_image_uid", "caption_len",
    ]
    available_cols = [c for c in summary_cols if c in df.columns]
    summary_df = df[available_cols].copy()
    output_path = os.path.join(PROCESSED_DIR, "comparison_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    return df, all_models


def build_model_pair_stats(df, all_models):
    """Build pairwise win-rate matrix between all model pairs."""
    print(f"\n{'='*60}")
    print("Building model pair statistics ...")
    print(f"{'='*60}")

    rows = []
    for m_a in all_models:
        for m_b in all_models:
            if m_a == m_b:
                continue
            # Cases where m_a is model_0 and m_b is model_1
            mask_ab = (df["model_0"] == m_a) & (df["model_1"] == m_b)
            n_ab = mask_ab.sum()
            if n_ab > 0:
                wins_a_ab = (df.loc[mask_ab, "label_0"] == 1.0).sum()
                wins_b_ab = (df.loc[mask_ab, "label_1"] == 1.0).sum()
                ties_ab = n_ab - wins_a_ab - wins_b_ab
            else:
                wins_a_ab = 0
                wins_b_ab = 0
                ties_ab = 0

            # Cases where m_a is model_1 and m_b is model_0
            mask_ba = (df["model_1"] == m_a) & (df["model_0"] == m_b)
            n_ba = mask_ba.sum()
            if n_ba > 0:
                wins_a_ba = (df.loc[mask_ba, "label_1"] == 1.0).sum()
                wins_b_ba = (df.loc[mask_ba, "label_0"] == 1.0).sum()
                ties_ba = n_ba - wins_a_ba - wins_b_ba
            else:
                wins_a_ba = 0
                wins_b_ba = 0
                ties_ba = 0

            total = n_ab + n_ba
            total_wins_a = wins_a_ab + wins_a_ba
            total_wins_b = wins_b_ab + wins_b_ba
            total_ties = ties_ab + ties_ba
            win_rate_a = total_wins_a / total if total > 0 else np.nan

            rows.append({
                "model_a": m_a,
                "model_b": m_b,
                "n_comparisons": total,
                "wins_a": total_wins_a,
                "wins_b": total_wins_b,
                "ties": total_ties,
                "win_rate_a": win_rate_a,
            })

    pair_df = pd.DataFrame(rows)
    output_path = os.path.join(PROCESSED_DIR, "model_pair_stats.csv")
    pair_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Print top matchups
    pair_df_sorted = pair_df.sort_values("n_comparisons", ascending=False)
    print(f"\n  Top 10 most-compared model pairs:")
    for _, r in pair_df_sorted.head(10).iterrows():
        print(
            f"    {r['model_a']:20s} vs {r['model_b']:20s}  "
            f"n={r['n_comparisons']:,}  win_rate_a={r['win_rate_a']:.3f}"
        )

    return pair_df


def build_model_summary(df, all_models):
    """Build per-model aggregate statistics."""
    print(f"\n{'='*60}")
    print("Building per-model summary ...")
    print(f"{'='*60}")

    rows = []
    for model in all_models:
        # All comparisons involving this model
        mask_as_0 = df["model_0"] == model
        mask_as_1 = df["model_1"] == model

        n_as_0 = mask_as_0.sum()
        n_as_1 = mask_as_1.sum()
        n_total = n_as_0 + n_as_1

        # Wins: label_0=1.0 when model is model_0, label_1=1.0 when model is model_1
        wins_as_0 = (df.loc[mask_as_0, "label_0"] == 1.0).sum()
        wins_as_1 = (df.loc[mask_as_1, "label_1"] == 1.0).sum()
        total_wins = wins_as_0 + wins_as_1

        # Losses
        losses_as_0 = (df.loc[mask_as_0, "label_1"] == 1.0).sum()
        losses_as_1 = (df.loc[mask_as_1, "label_0"] == 1.0).sum()
        total_losses = losses_as_0 + losses_as_1

        total_ties = n_total - total_wins - total_losses
        win_rate = total_wins / n_total if n_total > 0 else np.nan

        rows.append({
            "model": model,
            "n_comparisons": n_total,
            "n_wins": total_wins,
            "n_losses": total_losses,
            "n_ties": total_ties,
            "win_rate": win_rate,
        })

    model_df = pd.DataFrame(rows)
    model_df = model_df.sort_values("win_rate", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    model_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    print(f"\n  Model rankings (by win rate):")
    for _, r in model_df.iterrows():
        print(
            f"    {r['model']:25s}  n={r['n_comparisons']:>8,}  "
            f"win_rate={r['win_rate']:.4f}  "
            f"W/L/T={r['n_wins']:,}/{r['n_losses']:,}/{r['n_ties']:,}"
        )

    return model_df


def build_prompt_stats(df):
    """Build per-prompt comparison counts."""
    print(f"\n{'='*60}")
    print("Building per-prompt statistics ...")
    print(f"{'='*60}")

    prompt_counts = df.groupby("caption").agg(
        n_comparisons=("ranking_id", "count"),
        n_users=("user_id", "nunique"),
        n_models=("model_0", "nunique"),
    ).reset_index()

    prompt_counts = prompt_counts.sort_values("n_comparisons", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "prompt_stats.csv")
    prompt_counts.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    print(f"\n  Prompt comparison distribution:")
    print(f"    Unique prompts:        {len(prompt_counts):,}")
    print(f"    Mean comparisons/prompt: {prompt_counts['n_comparisons'].mean():.1f}")
    print(f"    Median:                {prompt_counts['n_comparisons'].median():.0f}")
    print(f"    Max:                   {prompt_counts['n_comparisons'].max()}")

    return prompt_counts


def main():
    print("Pick-a-Pic v2 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading Pick-a-Pic v2 from HuggingFace")
    print("-" * 60)
    df = download_data()

    # Step 2: Filter to labeled comparisons
    print("\nSTEP 2: Filtering to labeled comparisons")
    print("-" * 60)
    df = filter_labeled(df)

    # Step 3: Build comparison summary
    print("\nSTEP 3: Building comparison summary")
    print("-" * 60)
    df, all_models = build_comparison_summary(df)

    # Step 4: Build model pair stats
    print("\nSTEP 4: Building model pair statistics")
    print("-" * 60)
    pair_df = build_model_pair_stats(df, all_models)

    # Step 5: Build model summary
    print("\nSTEP 5: Building per-model summary")
    print("-" * 60)
    model_df = build_model_summary(df, all_models)

    # Step 6: Build prompt stats
    print("\nSTEP 6: Building per-prompt statistics")
    print("-" * 60)
    prompt_df = build_prompt_stats(df)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total labeled comparisons: {len(df):,}")
    print(f"  Unique models: {len(all_models)}")
    print(f"  Unique prompts: {df['caption'].nunique():,}")
    print(f"  Unique users: {df['user_id'].nunique():,}")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
