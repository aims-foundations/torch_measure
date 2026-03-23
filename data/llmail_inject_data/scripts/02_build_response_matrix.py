"""
02_build_response_matrix.py — LLMail-Inject dataset exploration and processing.

Loads HuggingFace dataset (saved via save_to_disk) from raw/llmail_inject/.
Expected: ~127K prompt injection attacks across 40 difficulty levels, multiple LLMs and defenses.
Builds: attack success rate by level, by LLM, by defense. Saves cross-tabulations.
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
                print(f"    First row sample:")
                row = ds[split_name][0]
                for k, v in row.items():
                    val_str = str(v)[:120]
                    print(f"      {k}: {val_str}")
        else:
            print(f"  Rows: {len(ds)}")
            print(f"  Columns: {ds.column_names}")
        return ds
    except Exception as e:
        print(f"Failed to load HF dataset from {path}: {e}")
        return None


def main():
    print("LLMail-Inject Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    # Try to find HF dataset directories (contain dataset_dict.json or dataset_info.json)
    hf_candidates = []
    for root, dirs, files in os.walk(RAW_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        if "dataset_dict.json" in files or "dataset_info.json" in files:
            hf_candidates.append(root)

    print(f"\nHF dataset candidates: {hf_candidates}")

    # Load the main dataset
    ds = None
    for candidate in hf_candidates:
        ds = try_load_hf_dataset(candidate)
        if ds is not None:
            break

    if ds is None:
        print("\nERROR: Could not load any HuggingFace dataset. Trying CSV/JSON fallback...")
        # Try loading any CSV or JSON files
        for f in all_files:
            fpath = RAW_DIR / f
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(fpath)
                    print(f"\nLoaded CSV: {f} ({len(df)} rows)")
                    print(f"  Columns: {list(df.columns)}")
                    print(f"  Dtypes:\n{df.dtypes}")
                    print(f"  Sample:\n{df.head(3)}")
                except Exception as e:
                    print(f"  Failed to load {f}: {e}")
            elif f.endswith(".json") and "dataset_info" not in f and "dataset_dict" not in f:
                try:
                    df = pd.read_json(fpath)
                    print(f"\nLoaded JSON: {f} ({len(df)} rows)")
                    print(f"  Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"  Failed to load {f}: {e}")
        print("\nNo usable data found. Exiting.")
        return

    # Convert to pandas DataFrames
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
            print(f"  Sample rows:\n{dfs[split_name].head(3)}")
    else:
        dfs["all"] = ds.to_pandas()
        print(f"\nAll data: {len(dfs['all'])} rows")
        print(f"  Columns: {list(dfs['all'].columns)}")
        print(f"  Dtypes:\n{dfs['all'].dtypes}")

    # Combine all splits into one DataFrame for analysis
    df_all = pd.concat(dfs.values(), ignore_index=True)
    print(f"\nCombined DataFrame: {len(df_all)} rows x {len(df_all.columns)} columns")
    print(f"Columns: {list(df_all.columns)}")

    # Discover key columns dynamically
    print("\n" + "=" * 60)
    print("COLUMN ANALYSIS")
    print("=" * 60)
    for col in df_all.columns:
        nunique = df_all[col].nunique()
        dtype = df_all[col].dtype
        print(f"\n  {col} (dtype={dtype}, nunique={nunique})")
        if nunique <= 50:
            vc = df_all[col].value_counts()
            print(f"    Value counts:\n{vc.to_string()}")
        else:
            print(f"    Sample values: {df_all[col].dropna().head(5).tolist()}")

    # Try to identify success/outcome, level, model, defense columns
    success_cols = [c for c in df_all.columns if any(kw in c.lower() for kw in ["success", "outcome", "result", "label", "pass", "win", "score"])]
    level_cols = [c for c in df_all.columns if any(kw in c.lower() for kw in ["level", "difficulty", "stage", "phase"])]
    model_cols = [c for c in df_all.columns if any(kw in c.lower() for kw in ["model", "llm", "agent"])]
    defense_cols = [c for c in df_all.columns if any(kw in c.lower() for kw in ["defense", "guard", "filter", "protection"])]

    print(f"\nDetected success columns: {success_cols}")
    print(f"Detected level columns: {level_cols}")
    print(f"Detected model columns: {model_cols}")
    print(f"Detected defense columns: {defense_cols}")

    # Build cross-tabulations
    print("\n" + "=" * 60)
    print("CROSS-TABULATIONS")
    print("=" * 60)

    # 1. Success rate by level
    if level_cols and success_cols:
        level_col = level_cols[0]
        success_col = success_cols[0]
        try:
            level_summary = df_all.groupby(level_col)[success_col].agg(["mean", "count", "sum"]).reset_index()
            level_summary.columns = [level_col, "success_rate", "total_attacks", "successful_attacks"]
            print(f"\nAttack success rate by {level_col}:")
            print(level_summary.to_string(index=False))
            level_summary.to_csv(PROCESSED_DIR / "success_rate_by_level.csv", index=False)
            print(f"  -> Saved to processed/success_rate_by_level.csv")
        except Exception as e:
            print(f"  Error computing level summary: {e}")

    # 2. Success rate by model
    if model_cols and success_cols:
        model_col = model_cols[0]
        success_col = success_cols[0]
        try:
            model_summary = df_all.groupby(model_col)[success_col].agg(["mean", "count", "sum"]).reset_index()
            model_summary.columns = [model_col, "success_rate", "total_attacks", "successful_attacks"]
            print(f"\nAttack success rate by {model_col}:")
            print(model_summary.to_string(index=False))
            model_summary.to_csv(PROCESSED_DIR / "success_rate_by_model.csv", index=False)
            print(f"  -> Saved to processed/success_rate_by_model.csv")
        except Exception as e:
            print(f"  Error computing model summary: {e}")

    # 3. Success rate by defense
    if defense_cols and success_cols:
        defense_col = defense_cols[0]
        success_col = success_cols[0]
        try:
            defense_summary = df_all.groupby(defense_col)[success_col].agg(["mean", "count", "sum"]).reset_index()
            defense_summary.columns = [defense_col, "success_rate", "total_attacks", "successful_attacks"]
            print(f"\nAttack success rate by {defense_col}:")
            print(defense_summary.to_string(index=False))
            defense_summary.to_csv(PROCESSED_DIR / "success_rate_by_defense.csv", index=False)
            print(f"  -> Saved to processed/success_rate_by_defense.csv")
        except Exception as e:
            print(f"  Error computing defense summary: {e}")

    # 4. Full cross-tab: level x model (if both exist)
    if level_cols and model_cols and success_cols:
        level_col = level_cols[0]
        model_col = model_cols[0]
        success_col = success_cols[0]
        try:
            cross = pd.crosstab(
                df_all[level_col],
                df_all[model_col],
                values=df_all[success_col],
                aggfunc="mean",
            )
            print(f"\nCross-tab: {level_col} x {model_col} (mean success rate):")
            print(cross.to_string())
            cross.to_csv(PROCESSED_DIR / "crosstab_level_x_model.csv")
            print(f"  -> Saved to processed/crosstab_level_x_model.csv")
        except Exception as e:
            print(f"  Error computing cross-tab: {e}")

    # 5. Full cross-tab: model x defense
    if model_cols and defense_cols and success_cols:
        model_col = model_cols[0]
        defense_col = defense_cols[0]
        success_col = success_cols[0]
        try:
            cross = pd.crosstab(
                df_all[model_col],
                df_all[defense_col],
                values=df_all[success_col],
                aggfunc="mean",
            )
            print(f"\nCross-tab: {model_col} x {defense_col} (mean success rate):")
            print(cross.to_string())
            cross.to_csv(PROCESSED_DIR / "crosstab_model_x_defense.csv")
            print(f"  -> Saved to processed/crosstab_model_x_defense.csv")
        except Exception as e:
            print(f"  Error computing cross-tab: {e}")

    # Save overall summary CSV
    summary = {
        "metric": ["total_rows", "total_columns", "n_splits"],
        "value": [len(df_all), len(df_all.columns), len(dfs)],
    }
    for col in df_all.columns:
        summary["metric"].append(f"nunique_{col}")
        summary["value"].append(df_all[col].nunique())
        if df_all[col].dtype in ("float64", "int64", "float32", "int32"):
            summary["metric"].append(f"mean_{col}")
            summary["value"].append(df_all[col].mean())

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"\n  -> Saved overall summary to processed/summary_statistics.csv")

    # Save the combined data as a flat CSV for downstream use
    out_path = PROCESSED_DIR / "llmail_inject_combined.csv"
    df_all.to_csv(out_path, index=False)
    print(f"  -> Saved combined data to processed/llmail_inject_combined.csv ({len(df_all)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
