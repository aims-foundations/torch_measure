"""
02_build_response_matrix.py — BELLS dataset exploration and processing.

BELLS (Benchmarks for the Evaluation of LLM Safeguards) contains 5+ sub-datasets:
  - hallucination, jailbreak, prompt injection, machiavelli, tensor_trust, etc.
Loads JSONL files from raw/bells_datasets/ and/or raw/BELLS/ repo.
Counts traces per dataset, summarizes labels.
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


def find_jsonl_files(base_dir):
    """Find all JSONL files recursively."""
    results = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if f.endswith(".jsonl"):
                results.append(os.path.join(root, f))
    return sorted(results)


def find_data_files(base_dir, extensions=(".jsonl", ".json", ".csv", ".parquet")):
    """Find data files recursively, excluding .git."""
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                data_files.append(os.path.join(root, f))
    return sorted(data_files)


def load_jsonl(fpath, max_lines=None):
    """Load a JSONL file, return list of dicts."""
    records = []
    try:
        with open(fpath) as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"  Error reading {fpath}: {e}")
    return records


def main():
    print("BELLS Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    # Search for JSONL files in multiple locations
    search_paths = [
        RAW_DIR / "bells_datasets",
        RAW_DIR / "BELLS",
    ]
    # Also add any other directories in raw/
    for d in RAW_DIR.iterdir():
        if d.is_dir() and d not in search_paths:
            search_paths.append(d)

    print("\n" + "=" * 60)
    print("SEARCHING FOR DATA FILES")
    print("=" * 60)

    all_data_files = []
    for sp in search_paths:
        if sp.exists():
            found = find_data_files(sp)
            if found:
                print(f"\n{sp.name}/:")
                for f in found[:20]:
                    rel = os.path.relpath(f, RAW_DIR)
                    size_kb = os.path.getsize(f) / 1024
                    print(f"  {rel} ({size_kb:.1f} KB)")
                if len(found) > 20:
                    print(f"  ... and {len(found) - 20} more")
                all_data_files.extend(found)

    # Specifically look for the BELLS benchmark data directories
    bells_repo = RAW_DIR / "BELLS"
    benchmark_dirs = []
    if bells_repo.exists():
        src_benchmarks = bells_repo / "src" / "benchmarks"
        if src_benchmarks.exists():
            benchmark_dirs = [d for d in src_benchmarks.iterdir() if d.is_dir() and d.name != "benchmark-template"]
            print(f"\nBELLS benchmark modules: {[d.name for d in benchmark_dirs]}")

    # Try to load JSONL files
    print("\n" + "=" * 60)
    print("LOADING JSONL DATA")
    print("=" * 60)

    jsonl_files = [f for f in all_data_files if f.endswith(".jsonl")]
    dataset_summaries = []

    for fpath in jsonl_files:
        rel = os.path.relpath(fpath, RAW_DIR)
        print(f"\n--- {rel} ---")
        records = load_jsonl(fpath)
        if not records:
            print("  Empty or unreadable")
            continue

        print(f"  Records: {len(records)}")
        # Examine structure of first record
        sample = records[0]
        print(f"  Keys: {list(sample.keys())}")
        for k, v in sample.items():
            vtype = type(v).__name__
            vstr = str(v)[:150]
            print(f"    {k} ({vtype}): {vstr}")

        # Convert to DataFrame for analysis
        try:
            df = pd.DataFrame(records)
            print(f"  DataFrame shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Dtypes:\n{df.dtypes.to_string()}")

            # Look for dataset/category labels
            label_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                          ["dataset", "label", "category", "type", "failure_type", "source", "benchmark"])]
            print(f"  Label columns: {label_cols}")

            for col in label_cols:
                if df[col].nunique() <= 50:
                    vc = df[col].value_counts()
                    print(f"\n  {col} distribution:")
                    print(f"{vc.to_string()}")

            dataset_summaries.append({
                "file": rel,
                "n_records": len(records),
                "n_columns": len(df.columns),
                "columns": "|".join(df.columns),
                "label_columns": "|".join(label_cols),
            })

            # Save individual dataset
            dataset_name = Path(fpath).stem
            df.to_csv(PROCESSED_DIR / f"bells_{dataset_name}.csv", index=False)
            print(f"  -> Saved to processed/bells_{dataset_name}.csv")

        except Exception as e:
            print(f"  Error converting to DataFrame: {e}")
            dataset_summaries.append({
                "file": rel,
                "n_records": len(records),
                "n_columns": len(records[0].keys()) if records else 0,
                "columns": "|".join(records[0].keys()) if records else "",
                "label_columns": "",
            })

    # Also try loading JSON files (non-JSONL)
    json_files = [f for f in all_data_files if f.endswith(".json") and not f.endswith(".jsonl")]
    for fpath in json_files[:10]:
        rel = os.path.relpath(fpath, RAW_DIR)
        print(f"\n--- {rel} (JSON) ---")
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"  List with {len(data)} items")
                if data:
                    print(f"  First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
            elif isinstance(data, dict):
                print(f"  Dict with {len(data)} keys: {list(data.keys())[:10]}")
        except Exception as e:
            print(f"  Error: {e}")

    # BELLS benchmark sub-dataset catalog
    print("\n" + "=" * 60)
    print("BELLS BENCHMARK SUB-DATASETS")
    print("=" * 60)

    known_subdatasets = [
        "hallucinations", "hallucinations_mcq", "jailbreakbench", "hf_jailbreak_prompts",
        "bipia", "tensor_trust", "machiavelli", "wildjailbreak", "dan",
        "guardrail_attacks", "lmsys_normal",
    ]
    for subds in known_subdatasets:
        subds_dir = bells_repo / "src" / "benchmarks" / subds if bells_repo.exists() else None
        if subds_dir and subds_dir.exists():
            py_files = list(subds_dir.glob("*.py"))
            data_files_in_dir = find_data_files(subds_dir)
            print(f"\n  {subds}:")
            print(f"    Python files: {[f.name for f in py_files]}")
            if data_files_in_dir:
                print(f"    Data files: {[os.path.basename(f) for f in data_files_in_dir]}")
            else:
                print(f"    No embedded data files (data likely loaded at runtime)")

    # Save catalog
    if dataset_summaries:
        catalog_df = pd.DataFrame(dataset_summaries)
        catalog_df.to_csv(PROCESSED_DIR / "dataset_catalog.csv", index=False)
        print(f"\n  -> Saved dataset catalog to processed/dataset_catalog.csv")

    # Overall summary
    summary_rows = [
        {"metric": "total_jsonl_files", "value": len(jsonl_files)},
        {"metric": "total_data_files", "value": len(all_data_files)},
        {"metric": "total_records_loaded", "value": sum(d.get("n_records", 0) for d in dataset_summaries)},
        {"metric": "n_benchmark_modules", "value": len(benchmark_dirs)},
        {"metric": "known_subdatasets", "value": len(known_subdatasets)},
    ]
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
