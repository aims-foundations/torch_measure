"""
02_build_response_matrix.py — CoT Unfaithfulness dataset exploration and processing.

Loads from raw/cot-unfaithfulness (git clone of milesaturpin/cot-unfaithfulness).
Expected structure:
  - data/bbq/data.json — BBQ bias benchmark questions
  - data/bbq/few_shot_prompts.json — few-shot prompt examples
  - results/bbq-verify-explanations.csv — verification of CoT explanations
  - results/bbq-qual-analysis.csv — qualitative analysis
  - results/bbh-verify-explanations.csv — BBH verification results
  - results/bbh-qual-analysis.csv — BBH qualitative analysis
  - results/bbq_samples/*.json — raw model outputs per condition
Summarizes CoT unfaithfulness rates across models and conditions.
"""

import json
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
    for f in sorted(all_files)[:80]:
        print(f"  {f}")
    if len(all_files) > 80:
        print(f"  ... and {len(all_files) - 80} more files")
    print(f"\nTotal files: {len(all_files)}")
    return all_files


def find_data_files(base_dir, extensions=(".json", ".csv", ".jsonl", ".parquet")):
    """Find data files recursively, excluding .git."""
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                data_files.append(os.path.join(root, f))
    return sorted(data_files)


def main():
    print("CoT Unfaithfulness Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "cot-unfaithfulness"
    if not repo_dir.exists():
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "unfaith" in candidate.name.lower():
                repo_dir = candidate
                break

    if not repo_dir.exists():
        print(f"\nERROR: Repo not found at {repo_dir}")
        return

    print(f"\nRepo directory: {repo_dir}")
    print(f"  Contents: {sorted(os.listdir(repo_dir))}")

    # Find all data files
    data_files = find_data_files(repo_dir)
    print(f"\nData files: {len(data_files)}")
    for f in data_files:
        rel = os.path.relpath(f, repo_dir)
        size_kb = os.path.getsize(f) / 1024
        print(f"  {rel} ({size_kb:.1f} KB)")

    # Load the results CSVs (primary analysis data)
    print("\n" + "=" * 60)
    print("RESULTS CSV FILES")
    print("=" * 60)

    results_dir = repo_dir / "results"
    results_dfs = {}

    if results_dir.exists():
        csv_files = sorted(results_dir.glob("*.csv"))
        print(f"CSV files: {[f.name for f in csv_files]}")

        for cf in csv_files:
            print(f"\n--- {cf.name} ---")
            try:
                df = pd.read_csv(cf)
                results_dfs[cf.stem] = df
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Dtypes:\n{df.dtypes.to_string()}")
                print(f"  Sample:\n{df.head(5).to_string()}")

                # Analyze each column
                for col in df.columns:
                    nunique = df[col].nunique()
                    dtype = df[col].dtype
                    if nunique <= 20:
                        vc = df[col].value_counts()
                        print(f"\n  {col} (nunique={nunique}):")
                        print(f"{vc.to_string()}")
                    elif dtype in ("float64", "int64"):
                        print(f"\n  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
                              f"min={df[col].min():.4f}, max={df[col].max():.4f}")

            except Exception as e:
                print(f"  Error loading: {e}")

    # Load BBQ data
    print("\n" + "=" * 60)
    print("BBQ BENCHMARK DATA")
    print("=" * 60)

    bbq_data_file = repo_dir / "data" / "bbq" / "data.json"
    if bbq_data_file.exists():
        try:
            with open(bbq_data_file) as f:
                bbq_data = json.load(f)

            if isinstance(bbq_data, list):
                print(f"BBQ data: list of {len(bbq_data)} items")
                if bbq_data:
                    print(f"  First item keys: {list(bbq_data[0].keys()) if isinstance(bbq_data[0], dict) else type(bbq_data[0])}")
                    try:
                        df_bbq = pd.DataFrame(bbq_data)
                        print(f"  As DataFrame: {df_bbq.shape}")
                        print(f"  Columns: {list(df_bbq.columns)}")
                        print(f"  Dtypes:\n{df_bbq.dtypes.to_string()}")
                        print(f"  Sample:\n{df_bbq.head(3).to_string()}")

                        # Look for category/bias columns
                        for col in df_bbq.columns:
                            if df_bbq[col].nunique() <= 30 and df_bbq[col].dtype == "object":
                                vc = df_bbq[col].value_counts()
                                print(f"\n  {col} distribution:")
                                print(f"{vc.to_string()}")

                        results_dfs["bbq_data"] = df_bbq
                    except Exception as e:
                        print(f"  Error converting to DataFrame: {e}")
            elif isinstance(bbq_data, dict):
                print(f"BBQ data: dict with {len(bbq_data)} keys")
                for k in list(bbq_data.keys())[:10]:
                    v = bbq_data[k]
                    print(f"  {k}: {type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
        except Exception as e:
            print(f"Error loading BBQ data: {e}")

    # Load few-shot prompts
    fewshot_file = repo_dir / "data" / "bbq" / "few_shot_prompts.json"
    if fewshot_file.exists():
        try:
            with open(fewshot_file) as f:
                fewshot_data = json.load(f)
            print(f"\nFew-shot prompts:")
            if isinstance(fewshot_data, dict):
                print(f"  Keys: {list(fewshot_data.keys())}")
                for k, v in list(fewshot_data.items())[:3]:
                    print(f"  {k}: {str(v)[:200]}")
            elif isinstance(fewshot_data, list):
                print(f"  {len(fewshot_data)} prompts")
        except Exception as e:
            print(f"  Error: {e}")

    # Load model sample outputs
    print("\n" + "=" * 60)
    print("MODEL SAMPLE OUTPUTS")
    print("=" * 60)

    samples_dir = results_dir / "bbq_samples" if results_dir.exists() else None
    sample_summary = []

    if samples_dir and samples_dir.exists():
        sample_files = sorted(samples_dir.glob("*.json"))
        print(f"Sample files: {len(sample_files)}")

        for sf in sample_files:
            print(f"\n--- {sf.name} ---")
            # Parse filename for conditions
            # Format: 20230302-001518-bbq-<model>-explicitnonbias<True/False>-fewshot<True/False>.json
            name_parts = sf.stem.split("-")
            print(f"  Name parts: {name_parts}")

            try:
                with open(sf) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    print(f"  Records: {len(data)}")
                    if data and isinstance(data[0], dict):
                        print(f"  Keys: {list(data[0].keys())}")
                        sample = data[0]
                        for k, v in sample.items():
                            print(f"    {k}: {str(v)[:120]}")

                        df_sample = pd.DataFrame(data)

                        # Extract condition info from filename
                        model_name = "unknown"
                        explicit_nonbias = "unknown"
                        fewshot = "unknown"
                        for part in name_parts:
                            if "explicitnonbias" in part.lower():
                                explicit_nonbias = part.replace("explicitnonbias", "")
                            elif "fewshot" in part.lower():
                                fewshot = part.replace("fewshot", "")
                        # Model name is between bbq- and -explicitnonbias
                        try:
                            bbq_idx = name_parts.index("bbq")
                            # Find the part containing 'explicitnonbias'
                            exp_idx = next(i for i, p in enumerate(name_parts) if "explicitnonbias" in p.lower())
                            model_name = "-".join(name_parts[bbq_idx + 1:exp_idx])
                        except (ValueError, StopIteration):
                            pass

                        sample_summary.append({
                            "file": sf.name,
                            "model": model_name,
                            "explicit_nonbias": explicit_nonbias,
                            "fewshot": fewshot,
                            "n_records": len(data),
                            "n_columns": len(df_sample.columns),
                        })

                        # Analyze label columns
                        label_cols = [c for c in df_sample.columns if any(kw in c.lower() for kw in
                                      ["answer", "correct", "bias", "label", "unfaith", "faith"])]
                        for col in label_cols:
                            if df_sample[col].nunique() <= 20:
                                print(f"\n  {col}:")
                                print(f"{df_sample[col].value_counts().to_string()}")

                elif isinstance(data, dict):
                    print(f"  Dict with {len(data)} keys")

            except Exception as e:
                print(f"  Error: {e}")

    if sample_summary:
        sample_df = pd.DataFrame(sample_summary)
        print(f"\n\nSample Summary Table:")
        print(sample_df.to_string(index=False))
        sample_df.to_csv(PROCESSED_DIR / "model_samples_summary.csv", index=False)
        print(f"\n  -> Saved to processed/model_samples_summary.csv")

    # Also check for BBH data files
    print("\n" + "=" * 60)
    print("BBH DATA (if available)")
    print("=" * 60)

    bbh_dir = repo_dir / "data" / "bbh"
    if bbh_dir.exists():
        bbh_files = find_data_files(bbh_dir)
        print(f"BBH data files: {len(bbh_files)}")
        for f in bbh_files[:10]:
            rel = os.path.relpath(f, repo_dir)
            print(f"  {rel}")
    else:
        print("  data/bbh/ not found (BBH data may be loaded at runtime)")

    # Save all results CSVs to processed/
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    for name, df in results_dfs.items():
        out_path = PROCESSED_DIR / f"cot_unfaith_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> Saved {name} ({len(df)} rows) to processed/cot_unfaith_{name}.csv")

    # Build cross-tabulation of unfaithfulness across conditions
    print("\n" + "=" * 60)
    print("UNFAITHFULNESS CROSS-TABULATION")
    print("=" * 60)

    # Check verify-explanations CSVs for unfaithfulness rates
    for key in ["bbq-verify-explanations", "bbh-verify-explanations"]:
        if key in results_dfs:
            df = results_dfs[key]
            print(f"\n--- {key} ---")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            # Look for model and unfaithfulness columns
            model_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["model", "agent"])]
            unfaith_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                           ["unfaith", "faith", "correct", "accurate", "consistent", "rate", "score"])]
            condition_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                             ["condition", "bias", "explicit", "fewshot", "setting"])]

            print(f"  Model columns: {model_cols}")
            print(f"  Unfaithfulness columns: {unfaith_cols}")
            print(f"  Condition columns: {condition_cols}")

            # Build cross-tab if possible
            if model_cols and unfaith_cols:
                for ucol in unfaith_cols:
                    try:
                        if df[ucol].dtype in ("float64", "int64"):
                            summary = df.groupby(model_cols[0])[ucol].agg(["mean", "std", "count"]).reset_index()
                            print(f"\n  {ucol} by {model_cols[0]}:")
                            print(summary.to_string(index=False))
                    except Exception as e:
                        print(f"  Error: {e}")

    # Overall summary
    summary_rows = [
        {"metric": "n_result_csv_files", "value": len([f for f in data_files if f.endswith(".csv")])},
        {"metric": "n_sample_files", "value": len(sample_summary)},
        {"metric": "n_total_data_files", "value": len(data_files)},
    ]
    for name, df in results_dfs.items():
        summary_rows.append({"metric": f"rows_{name}", "value": len(df)})

    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"\n  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
