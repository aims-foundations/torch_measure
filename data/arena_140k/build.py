"""
Build Arena 140K response matrix from pairwise human preferences.

Data source:
  - lmarena-ai/arena-human-preference-140k on HuggingFace
  - Format: ~140K pairwise comparisons with model_a, model_b, winner columns,
    plus language, is_code, and category_tag metadata per battle.

Processing:
  1. Download parquet from HuggingFace
  2. Save raw CSV
  3. Build a model x category response matrix where each column is a
     category bucket (language × code flag × primary criterion) and the
     value is the model's win rate in that bucket.
  4. Also save a pairwise comparison summary for downstream analysis.

Because prompts in Arena are effectively unique (~1 battle per prompt),
we aggregate to category buckets to get per-item model responses.

Outputs:
  - raw/arena_140k_raw.csv: Full raw data
  - processed/response_matrix.csv: Models (rows) x category buckets (columns),
    values = win rate in [0,1]
  - processed/pair_summary.csv: Per (model_a, model_b) win/loss/tie counts
"""

from pathlib import Path
import os
import sys

import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


def _score_from_winner(winner_val, ma, mb):
    """Return (score_a, score_b) for a single battle.

    Each battle contributes 1.0 to the winner, 0.0 to the loser, 0.5 to each
    on ties. Unknown verdicts are treated as ties.
    """
    wstr = str(winner_val).lower().strip()
    if wstr in ("model_a", "a") or wstr == str(ma).lower():
        return 1.0, 0.0
    if wstr in ("model_b", "b") or wstr == str(mb).lower():
        return 0.0, 1.0
    return 0.5, 0.5


def _parse_primary_criterion(cat_tag):
    """Extract a single coarse criterion label from the category_tag field."""
    if isinstance(cat_tag, dict):
        crit = cat_tag.get("criteria_v0.1")
        if isinstance(crit, dict):
            for k, v in crit.items():
                if v is True:
                    return k
    return "other"


def _parse_category_tag(val):
    """Parse category_tag field, which may be a dict, JSON string, or None."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        import ast
        try:
            obj = ast.literal_eval(val)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def build_response_matrix(df):
    """Build model x category-bucket response matrix.

    Prompts in Arena are effectively unique so we bucket them by
    (language, is_code, primary_criterion) and compute each model's win rate
    within each bucket.
    """
    print("\nBuilding response matrix (models x category buckets)...")

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
        print(f"  Columns found: {list(df.columns)}")
        raise ValueError(
            "Could not identify model_a, model_b, winner columns. "
            f"Found columns: {list(df.columns)}"
        )

    print(f"  Using columns: model_a={model_a_col}, model_b={model_b_col}, winner={winner_col}")

    all_models = sorted(set(df[model_a_col].unique()) | set(df[model_b_col].unique()))
    print(f"  Unique models: {len(all_models)}")

    # Winner distribution
    print("\n  Winner column values:")
    for val, count in df[winner_col].value_counts().items():
        print(f"    {val}: {count:,} ({count/len(df)*100:.1f}%)")

    # Parse category buckets for each battle
    print("\n  Parsing category buckets...")
    lang_col = "language" if "language" in df.columns else None
    code_col = "is_code" if "is_code" in df.columns else None
    cat_col = "category_tag" if "category_tag" in df.columns else None

    if cat_col is not None:
        df["_criterion"] = df[cat_col].apply(
            lambda v: _parse_primary_criterion(_parse_category_tag(v))
        )
    else:
        df["_criterion"] = "other"

    df["_lang"] = df[lang_col].fillna("unknown") if lang_col else "unknown"
    df["_code"] = df[code_col].fillna(False).astype(bool).map(
        {True: "code", False: "text"}
    ) if code_col else "text"

    def bucket_row(row):
        return f"{row['_lang']}__{row['_code']}__{row['_criterion']}"

    df["_bucket"] = df.apply(bucket_row, axis=1)

    # Accumulate per-(model, bucket): wins + total
    print("  Accumulating per-(model, bucket) stats...")
    wins_a = {}
    wins_b = {}
    for _, row in df.iterrows():
        ma = row[model_a_col]
        mb = row[model_b_col]
        bucket = row["_bucket"]
        sa, sb = _score_from_winner(row[winner_col], ma, mb)
        wins_a[(ma, bucket)] = wins_a.get((ma, bucket), [0.0, 0])
        wins_a[(ma, bucket)][0] += sa
        wins_a[(ma, bucket)][1] += 1
        wins_b[(mb, bucket)] = wins_b.get((mb, bucket), [0.0, 0])
        wins_b[(mb, bucket)][0] += sb
        wins_b[(mb, bucket)][1] += 1

    # Merge the two dicts
    combined = {}
    for (m, b), (s, n) in wins_a.items():
        combined[(m, b)] = [s, n]
    for (m, b), (s, n) in wins_b.items():
        if (m, b) in combined:
            combined[(m, b)][0] += s
            combined[(m, b)][1] += n
        else:
            combined[(m, b)] = [s, n]

    # Only keep buckets with >= MIN_TOTAL battles overall
    MIN_BUCKET_TOTAL = 10
    bucket_totals = {}
    for (m, b), (s, n) in combined.items():
        bucket_totals[b] = bucket_totals.get(b, 0) + n
    keep_buckets = sorted([b for b, t in bucket_totals.items() if t >= MIN_BUCKET_TOTAL])
    print(f"  Kept {len(keep_buckets)} / {len(bucket_totals)} buckets "
          f"with >= {MIN_BUCKET_TOTAL} total battles")

    # Build matrix
    matrix = pd.DataFrame(
        index=all_models, columns=keep_buckets, dtype=float
    )
    MIN_CELL = 3  # require at least 3 battles per (model, bucket)
    for (m, b), (s, n) in combined.items():
        if b in keep_buckets and n >= MIN_CELL:
            matrix.loc[m, b] = s / n

    matrix.index.name = "Model"
    n_models, n_buckets = matrix.shape
    fill = matrix.notna().sum().sum() / (n_models * n_buckets)
    print(f"  Matrix: {n_models} models x {n_buckets} buckets "
          f"(fill rate {fill*100:.1f}%)")

    # Also save the pairwise summary for reference
    print("\n  Building pairwise summary (for reference)...")
    rows = []
    pair_groups = df.groupby([model_a_col, model_b_col])
    for (ma, mb), group in pair_groups:
        win_a = 0
        win_b = 0
        tie = 0
        for _, r in group.iterrows():
            sa, sb = _score_from_winner(r[winner_col], ma, mb)
            if sa > sb:
                win_a += 1
            elif sb > sa:
                win_b += 1
            else:
                tie += 1
        total = win_a + win_b + tie
        rows.append({
            "model_a": ma, "model_b": mb,
            "win_a": win_a, "win_b": win_b, "tie": tie, "total": total,
            "win_rate_a": win_a / total if total > 0 else 0.0,
            "win_rate_b": win_b / total if total > 0 else 0.0,
            "tie_rate": tie / total if total > 0 else 0.0,
        })
    summary_df = pd.DataFrame(rows).sort_values(
        ["model_a", "model_b"]
    ).reset_index(drop=True)
    summary_path = os.path.join(PROCESSED_DIR, "pair_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved pair summary: {summary_path}")

    # Save response matrix
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix.to_csv(output_path)
    print(f"\n  Saved response matrix: {output_path}")

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

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
