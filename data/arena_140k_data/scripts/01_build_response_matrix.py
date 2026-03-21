"""
Build Arena 140K comparison summary from pairwise human preferences.

Data source:
  - lmarena-ai/arena-human-preference-140k on HuggingFace
  - Format: ~140K pairwise comparisons with model_a, model_b, winner columns

Processing:
  1. Download parquet from HuggingFace
  2. Save raw CSV
  3. Build processed comparison summary CSV (model-pair win/loss/tie counts)

Outputs:
  - raw/arena_140k_raw.csv: Full raw data
  - processed/response_matrix.csv: Pairwise comparison summary
    (rows = model pairs, columns = win_a, win_b, tie counts + rates)
"""

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


def download_data():
    """Download the Arena 140K dataset from HuggingFace."""
    raw_csv = os.path.join(RAW_DIR, "arena_140k_raw.csv")

    if os.path.exists(raw_csv) and os.path.getsize(raw_csv) > 1000:
        print(f"  Raw CSV already exists: {raw_csv}")
        return pd.read_csv(raw_csv)

    print("  Downloading lmarena-ai/arena-human-preference-140k from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "lmarena-ai/arena-human-preference-140k",
            split="train",
            token=os.environ.get("HF_TOKEN"),
        )
        df = ds.to_pandas()
    except ImportError:
        print("  'datasets' library not available, trying direct parquet download...")
        import urllib.request

        url = (
            "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k"
            "/resolve/main/data/train-00000-of-00001.parquet"
        )
        parquet_path = os.path.join(RAW_DIR, "train.parquet")
        if not os.path.exists(parquet_path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(parquet_path, "wb") as f:
                    f.write(resp.read())
        df = pd.read_parquet(parquet_path)

    df.to_csv(raw_csv, index=False)
    print(f"  Saved raw CSV: {raw_csv} ({len(df):,} rows)")
    return df


def build_response_matrix(df):
    """Build pairwise comparison summary from raw arena data."""
    print("\nBuilding pairwise comparison summary...")

    # Identify relevant columns
    model_a_col = None
    model_b_col = None
    winner_col = None

    for col in df.columns:
        cl = col.lower()
        if "model_a" in cl:
            model_a_col = col
        elif "model_b" in cl:
            model_b_col = col
        elif "winner" in cl:
            winner_col = col

    if model_a_col is None or model_b_col is None or winner_col is None:
        # Fall back to positional guessing
        print(f"  Columns found: {list(df.columns)}")
        raise ValueError(
            "Could not identify model_a, model_b, winner columns. "
            f"Found columns: {list(df.columns)}"
        )

    print(f"  Using columns: model_a={model_a_col}, model_b={model_b_col}, winner={winner_col}")

    # Get unique models
    all_models = sorted(set(df[model_a_col].unique()) | set(df[model_b_col].unique()))
    print(f"  Unique models: {len(all_models)}")

    # Winner value distribution
    print(f"\n  Winner column values:")
    for val, count in df[winner_col].value_counts().items():
        print(f"    {val}: {count:,} ({count/len(df)*100:.1f}%)")

    # Build pairwise summary
    rows = []
    pair_counts = df.groupby([model_a_col, model_b_col, winner_col]).size().reset_index(name="count")

    pair_groups = df.groupby([model_a_col, model_b_col])
    for (ma, mb), group in pair_groups:
        winner_counts = group[winner_col].value_counts().to_dict()

        # Determine win counts based on winner values
        # Common formats: "model_a"/"model_b"/"tie" or "A"/"B"/"tie"
        win_a = 0
        win_b = 0
        tie = 0
        for wval, cnt in winner_counts.items():
            wstr = str(wval).lower().strip()
            if wstr in ("model_a", "a"):
                win_a += cnt
            elif wstr in ("model_b", "b"):
                win_b += cnt
            elif "tie" in wstr:
                tie += cnt
            else:
                # Try to match model names
                if wstr == str(ma).lower():
                    win_a += cnt
                elif wstr == str(mb).lower():
                    win_b += cnt
                else:
                    tie += cnt

        total = win_a + win_b + tie
        rows.append({
            "model_a": ma,
            "model_b": mb,
            "win_a": win_a,
            "win_b": win_b,
            "tie": tie,
            "total": total,
            "win_rate_a": win_a / total if total > 0 else 0.0,
            "win_rate_b": win_b / total if total > 0 else 0.0,
            "tie_rate": tie / total if total > 0 else 0.0,
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["model_a", "model_b"]).reset_index(drop=True)

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Saved comparison summary: {output_path}")

    return summary_df


def print_statistics(df, summary_df):
    """Print detailed statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"  ARENA 140K STATISTICS")
    print(f"{'='*60}")

    print(f"\n  Raw data:")
    print(f"    Total comparisons:  {len(df):,}")
    print(f"    Columns:            {list(df.columns)}")

    # Model counts
    model_a_col = [c for c in df.columns if "model_a" in c.lower()][0]
    model_b_col = [c for c in df.columns if "model_b" in c.lower()][0]
    all_models = sorted(set(df[model_a_col].unique()) | set(df[model_b_col].unique()))
    print(f"    Unique models:      {len(all_models)}")

    # Per-model comparison counts
    model_counts = {}
    for m in all_models:
        count = ((df[model_a_col] == m) | (df[model_b_col] == m)).sum()
        model_counts[m] = count

    counts_series = pd.Series(model_counts).sort_values(ascending=False)
    print(f"\n  Per-model comparison counts:")
    print(f"    Min:    {counts_series.min():,} ({counts_series.idxmin()})")
    print(f"    Max:    {counts_series.max():,} ({counts_series.idxmax()})")
    print(f"    Mean:   {counts_series.mean():,.0f}")
    print(f"    Median: {counts_series.median():,.0f}")

    print(f"\n  Top 10 most compared models:")
    for model, count in counts_series.head(10).items():
        print(f"    {model:45s}  {count:,}")

    print(f"\n  Pairwise summary:")
    print(f"    Unique model pairs:   {len(summary_df):,}")
    print(f"    Mean comparisons/pair: {summary_df['total'].mean():.1f}")
    print(f"    Max comparisons/pair:  {summary_df['total'].max():,}")

    # Win rate distribution
    print(f"\n  Win rate distribution (model_a):")
    print(f"    Mean:   {summary_df['win_rate_a'].mean():.3f}")
    print(f"    Std:    {summary_df['win_rate_a'].std():.3f}")
    print(f"    Median: {summary_df['win_rate_a'].median():.3f}")

    print(f"\n  Tie rate distribution:")
    print(f"    Mean:   {summary_df['tie_rate'].mean():.3f}")
    print(f"    Std:    {summary_df['tie_rate'].std():.3f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("Arena 140K Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download
    print("STEP 1: Downloading Arena 140K dataset")
    print("-" * 60)
    df = download_data()

    # Step 2: Build comparison summary
    print("\nSTEP 2: Building pairwise comparison summary")
    print("-" * 60)
    summary_df = build_response_matrix(df)

    # Step 3: Statistics
    print("\nSTEP 3: Detailed statistics")
    print("-" * 60)
    print_statistics(df, summary_df)


if __name__ == "__main__":
    main()
