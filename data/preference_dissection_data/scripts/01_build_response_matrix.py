"""
Build Preference Dissection response matrix from per-judge binary preferences.

Data source:
  - GAIR/preference-dissection on HuggingFace (gated, requires accepted access)
  - 33 judges x 5,240 pairs, binary preferences
  - Requires HF_TOKEN with accepted access to the gated dataset

Processing:
  1. Download dataset from HuggingFace (with HF_TOKEN auth)
  2. Extract per-judge binary preferences for each pair
  3. Build response_matrix.csv (judges x pairs, binary 0/1)

Outputs:
  - raw/preference_dissection_raw.parquet: Cached raw data
  - processed/response_matrix.csv: Judges (rows) x pairs (columns), binary {0,1}
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
    """Download preference-dissection dataset from HuggingFace (gated)."""
    cache_path = os.path.join(RAW_DIR, "preference_dissection_raw.parquet")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached data: {cache_path}")
        return pd.read_parquet(cache_path)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set. This is a gated dataset requiring accepted access.")
        print("  Set HF_TOKEN environment variable with a token that has access to GAIR/preference-dissection.")

    print("  Downloading GAIR/preference-dissection from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "GAIR/preference-dissection",
        token=hf_token,
    )

    # Combine all splits
    all_dfs = []
    for split_name in ds:
        split_df = ds[split_name].to_pandas()
        split_df["_split"] = split_name
        all_dfs.append(split_df)
        print(f"    Split '{split_name}': {len(split_df):,} rows")
        print(f"    Columns: {list(split_df.columns)}")

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(cache_path)
    print(f"  Cached: {cache_path} ({len(df):,} rows)")

    return df


def extract_preferences(df):
    """Extract per-judge binary preferences from the dataset."""
    print("\n  Extracting per-judge preferences...")
    print(f"  Available columns: {list(df.columns)}")

    records = []

    # The dataset structure may vary. Common patterns:
    # 1. Each row = one pair, columns for each judge's preference
    # 2. Each row = one judgment (judge, pair_id, preference)

    # Check if judge preferences are in separate columns
    # Look for columns that might be judge names/IDs with binary values
    potential_judge_cols = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        # Check if column has binary-like values
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 5:
            vals_set = set()
            for v in unique_vals:
                try:
                    vals_set.add(float(v))
                except (ValueError, TypeError):
                    vals_set.add(v)
            if vals_set <= {0, 1, 0.0, 1.0, True, False}:
                potential_judge_cols.append(col)

    # Also check for nested preference structures
    has_preferences_col = "preferences" in df.columns or "annotations" in df.columns

    pair_id_col = None
    for col in ["id", "pair_id", "instance_id", "index"]:
        if col in df.columns:
            pair_id_col = col
            break

    if pair_id_col is None:
        # Use row index as pair ID
        df["_pair_id"] = df.index.astype(str)
        pair_id_col = "_pair_id"

    if len(potential_judge_cols) > 5:
        # Columns are judge names
        print(f"  Found {len(potential_judge_cols)} potential judge columns")
        print(f"  Judge columns: {potential_judge_cols[:10]}{'...' if len(potential_judge_cols) > 10 else ''}")

        for _, row in df.iterrows():
            pair_id = str(row[pair_id_col])
            for judge_col in potential_judge_cols:
                val = row[judge_col]
                if pd.notna(val):
                    try:
                        pref = int(float(val))
                        records.append({
                            "judge": judge_col,
                            "pair_id": pair_id,
                            "preference": pref,
                        })
                    except (ValueError, TypeError):
                        pass

    elif has_preferences_col:
        # Preferences are nested
        pref_col = "preferences" if "preferences" in df.columns else "annotations"
        print(f"  Parsing nested '{pref_col}' column...")

        for _, row in df.iterrows():
            pair_id = str(row[pair_id_col])
            prefs = row[pref_col]

            if isinstance(prefs, dict):
                for judge, pref in prefs.items():
                    try:
                        pref_val = int(float(pref))
                        records.append({
                            "judge": str(judge),
                            "pair_id": pair_id,
                            "preference": pref_val,
                        })
                    except (ValueError, TypeError):
                        pass
            elif isinstance(prefs, list):
                for i, pref in enumerate(prefs):
                    if isinstance(pref, dict):
                        judge = pref.get("judge", pref.get("model", f"judge_{i}"))
                        pref_val = pref.get("preference", pref.get("label", None))
                        if pref_val is not None:
                            try:
                                records.append({
                                    "judge": str(judge),
                                    "pair_id": pair_id,
                                    "preference": int(float(pref_val)),
                                })
                            except (ValueError, TypeError):
                                pass
                    else:
                        try:
                            records.append({
                                "judge": f"judge_{i}",
                                "pair_id": pair_id,
                                "preference": int(float(pref)),
                            })
                        except (ValueError, TypeError):
                            pass
    else:
        # Try to find judge and preference columns explicitly
        judge_col = None
        pref_col = None
        for col in df.columns:
            cl = col.lower()
            if "judge" in cl or "annotator" in cl or "evaluator" in cl:
                judge_col = col
            elif "preference" in cl or "label" in cl or "choice" in cl:
                pref_col = col

        if judge_col and pref_col:
            print(f"  Using columns: judge={judge_col}, preference={pref_col}")
            for _, row in df.iterrows():
                pair_id = str(row[pair_id_col])
                judge = str(row[judge_col])
                try:
                    pref = int(float(row[pref_col]))
                    records.append({
                        "judge": judge,
                        "pair_id": pair_id,
                        "preference": pref,
                    })
                except (ValueError, TypeError):
                    pass
        else:
            print(f"  WARNING: Could not determine data structure.")
            print(f"  Columns: {list(df.columns)}")
            print(f"  First row sample: {df.iloc[0].to_dict()}")
            # Fallback: try all non-metadata columns as judges
            exclude = {pair_id_col, "_split", "_pair_id"}
            judge_cols = [c for c in df.columns if c not in exclude]
            print(f"  Fallback: treating all {len(judge_cols)} remaining columns as judges")
            for _, row in df.iterrows():
                pair_id = str(row[pair_id_col])
                for jc in judge_cols:
                    val = row[jc]
                    if pd.notna(val):
                        try:
                            pref = int(float(val))
                            records.append({
                                "judge": jc,
                                "pair_id": pair_id,
                                "preference": pref,
                            })
                        except (ValueError, TypeError):
                            pass

    print(f"  Extracted {len(records):,} preference records")
    prefs_df = pd.DataFrame(records)

    if len(prefs_df) > 0:
        print(f"  Unique judges: {prefs_df['judge'].nunique()}")
        print(f"  Unique pairs:  {prefs_df['pair_id'].nunique()}")
        print(f"  Preference value distribution:")
        for val, count in prefs_df["preference"].value_counts().sort_index().items():
            print(f"    {val}: {count:,}")

    return prefs_df


def build_response_matrix(prefs_df):
    """Build response matrix (judges x pairs)."""
    print("\nBuilding response matrix...")

    matrix_df = prefs_df.pivot_table(
        index="judge",
        columns="pair_id",
        values="preference",
        aggfunc="first",  # Should be unique, but take first if duplicates
    )
    matrix_df.index.name = "Model"  # Convention: rows are "subjects" (here, judges)

    n_judges, n_pairs = matrix_df.shape
    print(f"  Matrix: {n_judges} judges x {n_pairs:,} pairs")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(output_path)
    print(f"  Saved: {output_path}")

    return matrix_df


def print_statistics(prefs_df, matrix_df):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print(f"  PREFERENCE DISSECTION STATISTICS")
    print(f"{'='*60}")

    n_judges, n_pairs = matrix_df.shape
    total_cells = n_judges * n_pairs
    n_valid = matrix_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    print(f"\n  Matrix dimensions:")
    print(f"    Judges:        {n_judges}")
    print(f"    Pairs:         {n_pairs:,}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Valid cells:   {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"    Missing cells: {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"    Fill rate:     {fill_rate*100:.1f}%")

    # Preference distribution
    all_prefs = matrix_df.values.flatten()
    valid_prefs = all_prefs[~np.isnan(all_prefs.astype(float))]
    if len(valid_prefs) > 0:
        print(f"\n  Preference distribution:")
        unique_vals, counts = np.unique(valid_prefs, return_counts=True)
        for val, count in zip(unique_vals, counts):
            pct = count / len(valid_prefs) * 100
            bar = "#" * int(pct / 2)
            print(f"    {int(val)}: {count:8,} ({pct:5.1f}%) {bar}")

    # Per-judge stats
    per_judge_mean = matrix_df.mean(axis=1).sort_values(ascending=False)
    per_judge_coverage = matrix_df.notna().sum(axis=1)

    print(f"\n  Per-judge statistics:")
    print(f"    Mean preference range: [{per_judge_mean.min():.3f}, {per_judge_mean.max():.3f}]")

    print(f"\n  All judges (sorted by mean preference):")
    for judge in per_judge_mean.index:
        mean_pref = per_judge_mean[judge]
        coverage = per_judge_coverage[judge]
        print(f"    {judge:50s}  mean={mean_pref:.3f}  coverage={coverage:,}")

    # Per-pair agreement
    per_pair_mean = matrix_df.mean(axis=0)
    per_pair_std = matrix_df.std(axis=0)

    print(f"\n  Per-pair agreement:")
    print(f"    Mean preference:  {per_pair_mean.mean():.3f}")
    print(f"    Mean std:         {per_pair_std.mean():.3f}")

    # Pairs with high/low agreement
    if len(per_pair_std.dropna()) > 0:
        high_agree = (per_pair_std < 0.1).sum()
        low_agree = (per_pair_std > 0.4).sum()
        print(f"    High agreement (std < 0.1): {high_agree:,} pairs")
        print(f"    Low agreement (std > 0.4):  {low_agree:,} pairs")

    # Inter-judge correlation
    print(f"\n  Inter-judge correlation (Pearson):")
    corr_matrix = matrix_df.T.corr()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    corr_vals = upper_tri.values.flatten()
    corr_vals = corr_vals[~np.isnan(corr_vals)]
    if len(corr_vals) > 0:
        print(f"    Mean:   {np.mean(corr_vals):.3f}")
        print(f"    Median: {np.median(corr_vals):.3f}")
        print(f"    Min:    {np.min(corr_vals):.3f}")
        print(f"    Max:    {np.max(corr_vals):.3f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("Preference Dissection Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Check for token
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set. GAIR/preference-dissection is a gated dataset.")
        print("You need accepted access + a valid HF token to download it.")
        print()

    # Step 1: Download
    print("STEP 1: Downloading Preference Dissection dataset")
    print("-" * 60)
    df = download_data()

    # Step 2: Extract preferences
    print("\nSTEP 2: Extracting per-judge preferences")
    print("-" * 60)
    prefs_df = extract_preferences(df)

    if len(prefs_df) == 0:
        print("  ERROR: No preferences extracted. Check dataset format.")
        print(f"  Dataset shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  First row: {df.iloc[0].to_dict()}")
        sys.exit(1)

    # Step 3: Build matrix
    print("\nSTEP 3: Building response matrix")
    print("-" * 60)
    matrix_df = build_response_matrix(prefs_df)

    # Step 4: Statistics
    print("\nSTEP 4: Detailed statistics")
    print("-" * 60)
    print_statistics(prefs_df, matrix_df)


if __name__ == "__main__":
    main()
