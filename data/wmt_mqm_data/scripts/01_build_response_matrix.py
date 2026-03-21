"""
Build WMT MQM response matrices from expert human translation quality evaluations.

Data source:
  - RicardoRei/wmt-mqm-human-evaluation on HuggingFace Hub
    WMT 2020-2022 Multidimensional Quality Metrics (MQM) annotations.

Structure:
  - Years: 2020, 2021, 2022
  - Language pairs: en-de, zh-en (2020); en-de, en-ru, zh-en (2021-2022)
  - Each row is one (system, segment) annotation with an MQM score
  - Scores are continuous (typically <= 0; lower = more errors; 0 = perfect)
  - Multiple annotators per segment in 2020 (mean=3), single annotator in 2021-2022

Score semantics:
  - MQM scores penalize translation errors by severity and category
  - 0.0 means no errors detected
  - Negative values indicate errors (e.g. -5 = 5 error penalty points)
  - Note: en-ru 2021 uses a different scoring scale (0-100, higher is better)

Outputs per (year, language_pair):
  - response_matrix.csv: Systems (rows) x segments (columns), mean MQM score
  - item_metadata.csv: Per-segment metadata (domain, source text, reference)
  - system_summary.csv: Per-system aggregate statistics
"""

import hashlib
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

SRC_REPO = "RicardoRei/wmt-mqm-human-evaluation"

YEAR_LP_COMBOS = [
    (2020, "en-de"),
    (2020, "zh-en"),
    (2021, "en-de"),
    (2021, "en-ru"),
    (2021, "zh-en"),
    (2022, "en-de"),
    (2022, "en-ru"),
    (2022, "zh-en"),
]


def _src_hash(src_text: str) -> str:
    """Create a short deterministic ID from source text."""
    return hashlib.sha256(src_text.encode("utf-8")).hexdigest()[:16]


def download_wmt_mqm():
    """Download the full WMT MQM dataset from HuggingFace."""
    print("Downloading WMT MQM dataset from HuggingFace ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(SRC_REPO, split="train")
        df = ds.to_pandas()
        print(f"  Loaded {len(df)} rows")
        print(f"  Years: {sorted(df['year'].unique())}")
        print(f"  Language pairs: {sorted(df['lp'].unique())}")

        # Save raw data
        raw_path = os.path.join(RAW_DIR, "wmt_mqm_full.csv")
        df.to_csv(raw_path, index=False)
        print(f"  Saved raw data: {raw_path}")
        return df
    except Exception as e:
        print(f"  Failed to download: {e}")
        # Try loading from cache
        raw_path = os.path.join(RAW_DIR, "wmt_mqm_full.csv")
        if os.path.exists(raw_path):
            print(f"  Loading from cache: {raw_path}")
            return pd.read_csv(raw_path)
        raise


def build_response_matrix(df, year, lp):
    """Build response matrix for one (year, language_pair) combination."""
    subset = df[(df["year"] == year) & (df["lp"] == lp)].copy()

    if len(subset) == 0:
        print(f"  WARNING: No data for year={year}, lp={lp}")
        return None, None, None

    # Create stable segment IDs from source text
    subset["seg_id"] = subset["src"].apply(_src_hash)

    # Aggregate: mean score per (system, segment) across annotators
    agg = subset.groupby(["system", "seg_id"]).agg(
        score=("score", "mean"),
        annotators=("annotators", "first"),
        domain=("domain", "first"),
        src=("src", "first"),
        ref=("ref", "first"),
    ).reset_index()

    # Pivot into matrix: systems (rows) x segments (columns)
    pivot = agg.pivot(index="system", columns="seg_id", values="score")
    pivot = pivot.sort_index()
    pivot = pivot[sorted(pivot.columns)]

    system_names = list(pivot.index)
    seg_ids = list(pivot.columns)
    n_systems = len(system_names)
    n_segments = len(seg_ids)

    # Create response matrix DataFrame
    matrix_df = pivot.copy()
    matrix_df.index.name = "system"

    # Statistics
    matrix_vals = matrix_df.values
    total_cells = n_systems * n_segments
    n_valid = np.sum(~np.isnan(matrix_vals))
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"  Systems:       {n_systems}")
    print(f"  Segments:      {n_segments}")
    print(f"  Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:     {fill_rate*100:.1f}%")

    # Score statistics
    valid_scores = matrix_vals[~np.isnan(matrix_vals)]
    print(f"\n  Score stats:")
    print(f"    Mean:   {np.mean(valid_scores):.4f}")
    print(f"    Std:    {np.std(valid_scores):.4f}")
    print(f"    Min:    {np.min(valid_scores):.4f}")
    print(f"    Max:    {np.max(valid_scores):.4f}")
    print(f"    Median: {np.median(valid_scores):.4f}")

    # Per-system stats
    per_system_mean = np.nanmean(matrix_vals, axis=1)
    best_idx = np.argmax(per_system_mean)
    worst_idx = np.argmin(per_system_mean)
    print(f"\n  Per-system mean score:")
    print(f"    Best:   {per_system_mean[best_idx]:.4f} ({system_names[best_idx]})")
    print(f"    Worst:  {per_system_mean[worst_idx]:.4f} ({system_names[worst_idx]})")
    print(f"    Median: {np.median(per_system_mean):.4f}")

    # Build item metadata
    seg_info = agg.drop_duplicates(subset="seg_id").set_index("seg_id")
    item_rows = []
    for seg_id in seg_ids:
        if seg_id in seg_info.index:
            row = seg_info.loc[seg_id]
            item_rows.append({
                "seg_id": seg_id,
                "domain": row["domain"],
                "src": row["src"],
                "ref": row["ref"],
                "mean_score": pivot[seg_id].mean(),
                "n_systems_rated": pivot[seg_id].notna().sum(),
            })
        else:
            item_rows.append({
                "seg_id": seg_id,
                "domain": "",
                "src": "",
                "ref": "",
                "mean_score": np.nan,
                "n_systems_rated": 0,
            })

    item_meta_df = pd.DataFrame(item_rows)

    # Build system summary
    sys_rows = []
    for i, sys_name in enumerate(system_names):
        sys_scores = matrix_vals[i, :]
        valid_mask = ~np.isnan(sys_scores)
        sys_rows.append({
            "system": sys_name,
            "mean_score": np.nanmean(sys_scores),
            "std_score": np.nanstd(sys_scores),
            "n_segments": int(np.sum(valid_mask)),
            "min_score": np.nanmin(sys_scores) if valid_mask.any() else np.nan,
            "max_score": np.nanmax(sys_scores) if valid_mask.any() else np.nan,
        })
    sys_summary_df = pd.DataFrame(sys_rows)
    sys_summary_df = sys_summary_df.sort_values("mean_score", ascending=False)

    return matrix_df, item_meta_df, sys_summary_df


def main():
    print("WMT MQM Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Downloading WMT MQM data")
    print("-" * 60)
    df = download_wmt_mqm()

    # Step 2: Build response matrices per (year, lp)
    print("\nSTEP 2: Building response matrices")
    print("-" * 60)

    all_results = {}

    for year, lp in YEAR_LP_COMBOS:
        lp_underscore = lp.replace("-", "_")
        key = f"{year}_{lp_underscore}"
        print(f"\n{'='*60}")
        print(f"  Building: wmt_mqm/{key}")
        print(f"{'='*60}")

        matrix_df, item_meta_df, sys_summary_df = build_response_matrix(df, year, lp)

        if matrix_df is None:
            continue

        # Create output directory for this split
        split_dir = os.path.join(PROCESSED_DIR, key)
        os.makedirs(split_dir, exist_ok=True)

        # Save response matrix
        matrix_path = os.path.join(split_dir, "response_matrix.csv")
        matrix_df.to_csv(matrix_path)
        print(f"\n  Saved: {matrix_path}")

        # Save item metadata
        item_meta_path = os.path.join(split_dir, "item_metadata.csv")
        item_meta_df.to_csv(item_meta_path, index=False)
        print(f"  Saved: {item_meta_path}")

        # Save system summary
        sys_summary_path = os.path.join(split_dir, "system_summary.csv")
        sys_summary_df.to_csv(sys_summary_path, index=False)
        print(f"  Saved: {sys_summary_path}")

        all_results[key] = {
            "matrix": matrix_df,
            "item_meta": item_meta_df,
            "sys_summary": sys_summary_df,
        }

    # Step 3: Domain breakdown
    print(f"\n{'='*60}")
    print("STEP 3: Domain breakdown")
    print(f"{'='*60}")

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        domains = sorted(year_df["domain"].unique())
        print(f"\n  Year {year}: domains = {domains}")
        for domain in domains:
            n = len(year_df[year_df["domain"] == domain])
            print(f"    {domain}: {n} rows")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows in source dataset: {len(df)}")
    print(f"  Splits built: {len(all_results)}")
    print()
    for key in sorted(all_results.keys()):
        matrix = all_results[key]["matrix"]
        n_sys, n_seg = matrix.shape
        nan_pct = matrix.isna().sum().sum() / (n_sys * n_seg) * 100
        print(f"  wmt_mqm/{key}: {n_sys} systems x {n_seg} segments ({nan_pct:.1f}% missing)")

    print(f"\n  All output files:")
    for root, dirs, files in os.walk(PROCESSED_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_kb = os.path.getsize(fpath) / 1024
            rel = os.path.relpath(fpath, PROCESSED_DIR)
            print(f"    {rel:55s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
