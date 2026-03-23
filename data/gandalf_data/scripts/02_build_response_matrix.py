"""
02_build_response_matrix.py — Gandalf Ignore Instructions dataset exploration and processing.

Loads HuggingFace dataset (saved via save_to_disk) from raw/gandalf_ignore_instructions/.
Expected: ~279K prompt injection attempts across 7 difficulty levels (Gandalf game).
Summarizes by level, saves labeled data.
"""

import os
import sys
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def list_raw_contents():
    """Recursively list files in raw/ (excluding .git), print summary."""
    print("=" * 60)
    print("FILES IN raw/")
    print("=" * 60)
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), RAW_DIR)
            all_files.append(rel)
    for f in sorted(all_files)[:50]:
        print(f"  {f}")
    if len(all_files) > 50:
        print(f"  ... and {len(all_files) - 50} more files")
    print(f"\nTotal files: {len(all_files)}")
    return all_files


def try_load_hf_dataset(path):
    """Try loading a HuggingFace DatasetDict or Dataset from disk."""
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        print(f"\nLoaded HF dataset from: {path}")
        print(f"  Type: {type(ds).__name__}")
        if hasattr(ds, "keys"):
            print(f"  Splits: {list(ds.keys())}")
            for split_name in ds:
                print(f"  {split_name}: {len(ds[split_name])} rows")
                print(f"    Columns: {ds[split_name].column_names}")
                sample = ds[split_name][0]
                for k, v in sample.items():
                    print(f"      {k}: {str(v)[:120]}")
        else:
            print(f"  Rows: {len(ds)}")
            print(f"  Columns: {ds.column_names}")
        return ds
    except Exception as e:
        print(f"Failed to load HF dataset from {path}: {e}")
        return None


def main():
    print("Gandalf Ignore Instructions Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    # Find HF datasets
    hf_candidates = []
    for root, dirs, files in os.walk(RAW_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        if "dataset_dict.json" in files or "dataset_info.json" in files:
            hf_candidates.append(root)

    print(f"\nHF dataset candidates: {hf_candidates}")

    ds = None
    for candidate in hf_candidates:
        ds = try_load_hf_dataset(candidate)
        if ds is not None:
            break

    if ds is None:
        print("\nERROR: Could not load any HuggingFace dataset.")
        return

    # Convert to pandas
    print("\n" + "=" * 60)
    print("CONVERTING TO PANDAS")
    print("=" * 60)

    dfs = {}
    if hasattr(ds, "keys"):
        for split_name in ds:
            dfs[split_name] = ds[split_name].to_pandas()
            print(f"\n{split_name}: {len(dfs[split_name])} rows")
            print(f"  Columns: {list(dfs[split_name].columns)}")
            print(f"  Dtypes:\n{dfs[split_name].dtypes}")
            print(f"  Sample:\n{dfs[split_name].head(3)}")
    else:
        dfs["all"] = ds.to_pandas()

    df_all = pd.concat(dfs.values(), ignore_index=True)
    print(f"\nCombined: {len(df_all)} rows x {len(df_all.columns)} columns")
    print(f"Columns: {list(df_all.columns)}")

    # Column analysis
    print("\n" + "=" * 60)
    print("COLUMN ANALYSIS")
    print("=" * 60)
    for col in df_all.columns:
        nunique = df_all[col].nunique()
        dtype = df_all[col].dtype
        print(f"\n  {col} (dtype={dtype}, nunique={nunique})")
        if nunique <= 30:
            vc = df_all[col].value_counts().head(20)
            print(f"    Value counts:\n{vc.to_string()}")
        elif dtype in ("float64", "int64", "float32", "int32"):
            print(f"    Stats: {df_all[col].describe().to_dict()}")
        else:
            print(f"    Sample values: {df_all[col].dropna().head(5).tolist()}")

    # Detect level column
    level_cols = [c for c in df_all.columns if any(kw in c.lower() for kw in ["level", "difficulty", "stage", "tier"])]
    success_cols = [c for c in df_all.columns if any(kw in c.lower() for kw in ["success", "label", "result", "outcome", "score", "pass", "win"])]

    print(f"\nDetected level columns: {level_cols}")
    print(f"Detected success/label columns: {success_cols}")

    # Summarize by level
    print("\n" + "=" * 60)
    print("SUMMARY BY LEVEL")
    print("=" * 60)

    if level_cols:
        level_col = level_cols[0]
        level_summary = df_all.groupby(level_col).agg(
            count=pd.NamedAgg(column=level_col, aggfunc="count"),
        ).reset_index()

        # Add success rates if we have a success column
        if success_cols:
            success_col = success_cols[0]
            try:
                level_success = df_all.groupby(level_col)[success_col].agg(["mean", "sum"]).reset_index()
                level_success.columns = [level_col, "success_rate", "successful_count"]
                level_summary = level_summary.merge(level_success, on=level_col, how="left")
            except Exception as e:
                print(f"  Could not compute success rate: {e}")

        print(level_summary.to_string(index=False))
        level_summary.to_csv(PROCESSED_DIR / "summary_by_level.csv", index=False)
        print(f"\n  -> Saved to processed/summary_by_level.csv")
    else:
        print("  No level column detected. Summarizing all categorical columns instead.")
        for col in df_all.select_dtypes(include=["object", "category"]).columns:
            if df_all[col].nunique() <= 20:
                print(f"\n  {col}:")
                vc = df_all[col].value_counts()
                print(vc.to_string())

    # Per-split summaries
    print("\n" + "=" * 60)
    print("PER-SPLIT SUMMARY")
    print("=" * 60)
    for split_name, split_df in dfs.items():
        print(f"\n--- {split_name} ({len(split_df)} rows) ---")
        if level_cols:
            vc = split_df[level_cols[0]].value_counts().sort_index()
            print(f"  {level_cols[0]} distribution:\n{vc.to_string()}")

    # Check for text length distributions (prompts tend to have interesting length patterns)
    text_cols = [c for c in df_all.columns if df_all[c].dtype == "object" and df_all[c].str.len().mean() > 20]
    if text_cols:
        print("\n" + "=" * 60)
        print("TEXT LENGTH STATISTICS")
        print("=" * 60)
        for col in text_cols[:3]:
            lengths = df_all[col].str.len()
            print(f"\n  {col}: mean={lengths.mean():.1f}, median={lengths.median():.1f}, "
                  f"min={lengths.min()}, max={lengths.max()}")
            if level_cols:
                len_by_level = df_all.groupby(level_cols[0])[col].apply(lambda x: x.str.len().mean())
                print(f"  Mean length by {level_cols[0]}:\n{len_by_level.to_string()}")

    # Save labeled data
    out_path = PROCESSED_DIR / "gandalf_combined.csv"
    # Truncate very long text columns for the CSV
    df_save = df_all.copy()
    for col in text_cols:
        df_save[col] = df_save[col].str[:500]
    df_save.to_csv(out_path, index=False)
    print(f"\n  -> Saved combined data to processed/gandalf_combined.csv ({len(df_save)} rows)")

    # Save overall summary
    summary_rows = [
        {"metric": "total_rows", "value": len(df_all)},
        {"metric": "total_columns", "value": len(df_all.columns)},
        {"metric": "n_splits", "value": len(dfs)},
    ]
    for col in df_all.columns:
        summary_rows.append({"metric": f"nunique_{col}", "value": df_all[col].nunique()})
    if level_cols:
        summary_rows.append({"metric": "n_levels", "value": df_all[level_cols[0]].nunique()})

    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
