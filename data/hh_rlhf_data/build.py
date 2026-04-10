"""
Build HH-RLHF pairwise preference data from Anthropic/hh-rlhf.

Data source:
  - Anthropic/hh-rlhf on HuggingFace Hub: ~161K chosen/rejected conversation
    pairs for RLHF training.  Split into "helpful" (helpful-base,
    helpful-online, helpful-rejection-sampled) and "harmless" (harmless-base)
    subsets.

  The HF repo contains separate JSONL.gz files per subset:
      harmless-base/{train,test}.jsonl.gz
      helpful-base/{train,test}.jsonl.gz
      helpful-online/{train,test}.jsonl.gz
      helpful-rejection-sampled/{train,test}.jsonl.gz

Format:
  - Each sample is a (chosen, rejected) conversation pair where a human
    annotator preferred the chosen conversation.
  - Conversations are multi-turn dialogues in "Human: ... Assistant: ..." format.

Outputs:
  - processed/helpful_summary.csv: Per-pair metadata for helpful subset
  - processed/harmless_summary.csv: Per-pair metadata for harmless subset
  - processed/subset_stats.csv: Aggregate statistics per subset
"""

import gzip
import json
import os
import sys

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

HF_SOURCE = "Anthropic/hh-rlhf"

SUBSET_FILES = {
    "helpful": ["helpful-base", "helpful-online", "helpful-rejection-sampled"],
    "harmless": ["harmless-base"],
}


def count_turns(text: str) -> int:
    """Count the number of conversation turns (Human + Assistant exchanges)."""
    return text.count("\n\nHuman:") + text.count("\n\nAssistant:")


def text_length(text: str) -> int:
    """Return character length of a text."""
    return len(text)


def load_jsonl_gz(path: str) -> list[dict]:
    """Load a gzipped JSONL file and return list of dicts."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_subset(subset_name: str, file_prefixes: list[str]) -> pd.DataFrame:
    """Download and combine all JSONL files for a subset."""
    from huggingface_hub import hf_hub_download

    rows = []
    for prefix in file_prefixes:
        for split_name in ["train", "test"]:
            filename = f"{prefix}/{split_name}.jsonl.gz"
            print(f"    Downloading {filename} ...")

            local_path = hf_hub_download(
                repo_id=HF_SOURCE,
                filename=filename,
                repo_type="dataset",
            )

            jsonl_rows = load_jsonl_gz(local_path)
            print(f"      {split_name}: {len(jsonl_rows)} pairs")
            for i, row in enumerate(jsonl_rows):
                rows.append({
                    "config": prefix,
                    "split": split_name,
                    "pair_idx": i,
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })

    df = pd.DataFrame(rows)
    print(f"  Total {subset_name}: {len(df)} pairs")
    return df


def compute_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pair statistics."""
    df = df.copy()
    df["chosen_len"] = df["chosen"].apply(text_length)
    df["rejected_len"] = df["rejected"].apply(text_length)
    df["chosen_turns"] = df["chosen"].apply(count_turns)
    df["rejected_turns"] = df["rejected"].apply(count_turns)
    df["len_diff"] = df["chosen_len"] - df["rejected_len"]
    df["turn_diff"] = df["chosen_turns"] - df["rejected_turns"]
    return df


def print_subset_stats(df: pd.DataFrame, subset_name: str) -> dict:
    """Print and return aggregate statistics for a subset."""
    print(f"\n{'='*60}")
    print(f"  {subset_name.upper()} SUBSET STATISTICS")
    print(f"{'='*60}")

    n_pairs = len(df)
    print(f"  Total pairs: {n_pairs:,}")

    # Config breakdown
    print(f"\n  Config breakdown:")
    for config, count in df["config"].value_counts().items():
        print(f"    {config}: {count:,}")

    # Split breakdown
    print(f"\n  Split breakdown:")
    for split, count in df["split"].value_counts().items():
        print(f"    {split}: {count:,}")

    # Chosen conversation stats
    print(f"\n  Chosen conversation length (chars):")
    print(f"    Mean:   {df['chosen_len'].mean():.0f}")
    print(f"    Median: {df['chosen_len'].median():.0f}")
    print(f"    Std:    {df['chosen_len'].std():.0f}")
    print(f"    Min:    {df['chosen_len'].min()}")
    print(f"    Max:    {df['chosen_len'].max()}")

    # Rejected conversation stats
    print(f"\n  Rejected conversation length (chars):")
    print(f"    Mean:   {df['rejected_len'].mean():.0f}")
    print(f"    Median: {df['rejected_len'].median():.0f}")
    print(f"    Std:    {df['rejected_len'].std():.0f}")
    print(f"    Min:    {df['rejected_len'].min()}")
    print(f"    Max:    {df['rejected_len'].max()}")

    # Turn counts
    print(f"\n  Chosen turn count:")
    print(f"    Mean:   {df['chosen_turns'].mean():.1f}")
    print(f"    Median: {df['chosen_turns'].median():.0f}")

    print(f"\n  Rejected turn count:")
    print(f"    Mean:   {df['rejected_turns'].mean():.1f}")
    print(f"    Median: {df['rejected_turns'].median():.0f}")

    # Length difference (chosen - rejected)
    print(f"\n  Length difference (chosen - rejected):")
    print(f"    Mean:   {df['len_diff'].mean():.0f}")
    print(f"    Median: {df['len_diff'].median():.0f}")
    chosen_longer = (df["len_diff"] > 0).sum()
    rejected_longer = (df["len_diff"] < 0).sum()
    same_len = (df["len_diff"] == 0).sum()
    print(f"    Chosen longer:   {chosen_longer:,} ({chosen_longer/n_pairs*100:.1f}%)")
    print(f"    Rejected longer: {rejected_longer:,} ({rejected_longer/n_pairs*100:.1f}%)")
    print(f"    Same length:     {same_len:,} ({same_len/n_pairs*100:.1f}%)")

    return {
        "subset": subset_name,
        "n_pairs": n_pairs,
        "n_configs": df["config"].nunique(),
        "n_train": (df["split"] == "train").sum(),
        "n_test": (df["split"] == "test").sum(),
        "chosen_len_mean": df["chosen_len"].mean(),
        "chosen_len_median": df["chosen_len"].median(),
        "rejected_len_mean": df["rejected_len"].mean(),
        "rejected_len_median": df["rejected_len"].median(),
        "chosen_turns_mean": df["chosen_turns"].mean(),
        "rejected_turns_mean": df["rejected_turns"].mean(),
        "pct_chosen_longer": chosen_longer / n_pairs * 100,
    }


def main():
    print("Anthropic HH-RLHF Data Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading HH-RLHF from HuggingFace")
    print("-" * 60)

    all_stats_rows = []

    for subset_name, file_prefixes in SUBSET_FILES.items():
        print(f"\n--- {subset_name} ---")

        # Download
        df = download_subset(subset_name, file_prefixes)

        if df.empty:
            print(f"  WARNING: No data for {subset_name}, skipping")
            continue

        # Compute stats
        df = compute_pair_stats(df)

        # Print stats
        stats = print_subset_stats(df, subset_name)
        all_stats_rows.append(stats)

        # Save summary CSV (without full text for size)
        summary_df = df[["config", "split", "pair_idx",
                         "chosen_len", "rejected_len",
                         "chosen_turns", "rejected_turns",
                         "len_diff", "turn_diff"]].copy()
        summary_path = os.path.join(PROCESSED_DIR, f"{subset_name}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Saved: {summary_path}")

    # Step 2: Save aggregate stats
    print(f"\n\n{'='*60}")
    print("STEP 2: Saving aggregate statistics")
    print("-" * 60)

    stats_df = pd.DataFrame(all_stats_rows)
    stats_path = os.path.join(PROCESSED_DIR, "subset_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    total = sum(r["n_pairs"] for r in all_stats_rows)
    print(f"  Total preference pairs: {total:,}")
    for row in all_stats_rows:
        print(f"    {row['subset']}: {row['n_pairs']:,} pairs")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
