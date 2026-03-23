"""
02_build_response_matrix.py — FaithCoT-BENCH dataset exploration and processing.

Loads from raw/FaithCoT-BENCH (git clone of the FaithCoT-BENCH repo).
Expected: faithfulness annotations for chain-of-thought reasoning.
The repo may contain a zip file (faithcot.zip) with the actual data.
Discovers CSV/JSON data files, summarizes faithfulness annotations, saves catalog.
"""

import json
import os
import sys
import zipfile
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


def find_data_files(base_dir, extensions=(".csv", ".json", ".jsonl", ".tsv", ".parquet")):
    """Find data files recursively, excluding .git."""
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                data_files.append(os.path.join(root, f))
    return sorted(data_files)


def extract_zip_if_needed(zip_path, extract_to):
    """Extract a zip file if not already extracted."""
    if not os.path.exists(zip_path):
        return None
    # Check if already extracted
    marker = extract_to / ".extracted"
    if marker.exists():
        print(f"  Zip already extracted to {extract_to}")
        return extract_to
    print(f"  Extracting {zip_path} to {extract_to} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        marker.touch()
        print(f"  Extracted {len(os.listdir(extract_to))} items")
        return extract_to
    except Exception as e:
        print(f"  Failed to extract zip: {e}")
        return None


def load_data_file(fpath):
    """Try to load a single data file, return DataFrame or None."""
    fpath = str(fpath)
    try:
        if fpath.endswith(".csv"):
            df = pd.read_csv(fpath)
            return df
        elif fpath.endswith(".tsv"):
            df = pd.read_csv(fpath, sep="\t")
            return df
        elif fpath.endswith(".jsonl"):
            records = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return pd.DataFrame(records) if records else None
        elif fpath.endswith(".json"):
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Could be a dict of lists or a single record
                if all(isinstance(v, list) for v in data.values()):
                    return pd.DataFrame(data)
                else:
                    # Try to interpret as dataset with splits
                    for key, val in data.items():
                        if isinstance(val, list) and len(val) > 0:
                            print(f"    Found key '{key}' with {len(val)} items")
                    return None
        elif fpath.endswith(".parquet"):
            return pd.read_parquet(fpath)
    except Exception as e:
        print(f"  Error loading {fpath}: {e}")
    return None


def main():
    print("FaithCoT-BENCH Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "FaithCoT-BENCH"
    if not repo_dir.exists():
        # Try other common names
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "faith" in candidate.name.lower():
                repo_dir = candidate
                break

    print(f"\nRepo directory: {repo_dir}")

    # Check for zip files and extract if needed
    zip_files = list(repo_dir.glob("*.zip")) if repo_dir.exists() else []
    print(f"Zip files found: {[str(z) for z in zip_files]}")

    extracted_dir = None
    for zf in zip_files:
        extract_to = PROCESSED_DIR / "extracted"
        extracted_dir = extract_zip_if_needed(str(zf), extract_to)

    # Find all data files in repo and extracted directory
    search_dirs = [repo_dir]
    if extracted_dir and Path(extracted_dir).exists():
        search_dirs.append(Path(extracted_dir))

    all_data_files = []
    for search_dir in search_dirs:
        if search_dir and search_dir.exists():
            found = find_data_files(search_dir)
            all_data_files.extend(found)
            print(f"\nData files in {search_dir}:")
            for f in found[:30]:
                rel = os.path.relpath(f, RAW_DIR)
                size_kb = os.path.getsize(f) / 1024
                print(f"  {rel} ({size_kb:.1f} KB)")
            if len(found) > 30:
                print(f"  ... and {len(found) - 30} more")

    if not all_data_files:
        print("\nNo CSV/JSON/JSONL data files found. Listing all non-git files for reference:")
        for f in all_files:
            if not f.startswith(".git"):
                print(f"  {f}")
        print("\nExiting — no data to process.")
        return

    # Load and explore each data file
    print("\n" + "=" * 60)
    print("LOADING DATA FILES")
    print("=" * 60)

    file_catalog = []
    loaded_dfs = {}

    for fpath in all_data_files:
        rel = os.path.relpath(fpath, RAW_DIR)
        print(f"\n--- {rel} ---")
        df = load_data_file(fpath)
        if df is not None and len(df) > 0:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Dtypes:\n{df.dtypes.to_string()}")
            print(f"  Sample:\n{df.head(2).to_string()}")

            file_catalog.append({
                "file": rel,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": "|".join(df.columns),
                "size_kb": os.path.getsize(fpath) / 1024,
            })
            loaded_dfs[rel] = df
        else:
            print(f"  Could not load as DataFrame")
            file_catalog.append({
                "file": rel,
                "rows": 0,
                "columns": 0,
                "column_names": "",
                "size_kb": os.path.getsize(fpath) / 1024,
            })

    # Save file catalog
    catalog_df = pd.DataFrame(file_catalog)
    catalog_df.to_csv(PROCESSED_DIR / "file_catalog.csv", index=False)
    print(f"\n  -> Saved file catalog to processed/file_catalog.csv ({len(catalog_df)} files)")

    # Analyze faithfulness-related columns across all loaded DataFrames
    print("\n" + "=" * 60)
    print("FAITHFULNESS ANALYSIS")
    print("=" * 60)

    faith_keywords = ["faith", "correct", "label", "faithful", "unfaithful", "consistency",
                       "accuracy", "error", "halluc", "score", "verdict", "judge"]

    for rel, df in loaded_dfs.items():
        faith_cols = [c for c in df.columns if any(kw in c.lower() for kw in faith_keywords)]
        if faith_cols:
            print(f"\n--- {rel} ---")
            print(f"  Faithfulness-related columns: {faith_cols}")
            for col in faith_cols:
                nunique = df[col].nunique()
                print(f"\n  {col} (nunique={nunique}):")
                if nunique <= 20:
                    vc = df[col].value_counts()
                    print(f"{vc.to_string()}")
                elif df[col].dtype in ("float64", "int64"):
                    print(f"    Stats: {df[col].describe().to_dict()}")

    # Try to build a summary across datasets/tasks
    task_cols = [c for c in set().union(*(df.columns for df in loaded_dfs.values()))
                 if any(kw in c.lower() for kw in ["task", "dataset", "benchmark", "category", "domain"])]
    print(f"\nTask/dataset columns found across files: {task_cols}")

    # Pick the largest loaded DataFrame for deeper analysis
    if loaded_dfs:
        largest_key = max(loaded_dfs, key=lambda k: len(loaded_dfs[k]))
        largest_df = loaded_dfs[largest_key]
        print(f"\nLargest file: {largest_key} ({len(largest_df)} rows)")
        print(f"  Columns: {list(largest_df.columns)}")

        # Save it as the main processed output
        out_path = PROCESSED_DIR / "faithcot_main.csv"
        largest_df.to_csv(out_path, index=False)
        print(f"  -> Saved to processed/faithcot_main.csv")

    # Overall summary
    summary_rows = [
        {"metric": "total_data_files", "value": len(all_data_files)},
        {"metric": "loadable_files", "value": len(loaded_dfs)},
        {"metric": "total_rows_all_files", "value": sum(len(df) for df in loaded_dfs.values())},
    ]
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
