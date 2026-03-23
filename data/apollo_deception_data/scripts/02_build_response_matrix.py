"""
02_build_response_matrix.py — Apollo Deception Detection dataset exploration and processing.

Loads from raw/deception-detection (git clone of ApolloResearch/deception-detection).
Expected probe datasets:
  - AI Liar (how_to_catch_an_ai_liar/)
  - Roleplaying (roleplaying/)
  - Insider Trading (insider_trading/)
  - Sandbagging (sandbagging_v2/)
  - AI Audit (ai_audit/)
  - Werewolf (werewolf/)
  - Geometry of Truth (geometry_of_truth/)
  - TruthfulQA (truthfulqa/)
  - Rollouts (rollouts/)
Summarizes honest/deceptive/ambiguous labels per dataset.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

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


def load_file(fpath):
    """Load a data file (JSON, CSV, YAML, JSONL). Returns raw data."""
    fpath = str(fpath)
    try:
        if fpath.endswith(".csv"):
            return pd.read_csv(fpath)
        elif fpath.endswith(".jsonl"):
            records = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif fpath.endswith(".json"):
            with open(fpath) as f:
                return json.load(f)
        elif fpath.endswith(".yaml") or fpath.endswith(".yml"):
            with open(fpath) as f:
                return yaml.safe_load(f)
    except Exception as e:
        print(f"  Error loading {fpath}: {e}")
    return None


def summarize_data(data, label=""):
    """Print a summary of loaded data."""
    if isinstance(data, pd.DataFrame):
        print(f"  {label}DataFrame: {data.shape}")
        print(f"    Columns: {list(data.columns)}")
        print(f"    Dtypes:\n{data.dtypes.to_string()}")
        print(f"    Sample:\n{data.head(3).to_string()}")
        return data
    elif isinstance(data, list):
        print(f"  {label}List: {len(data)} items")
        if data and isinstance(data[0], dict):
            print(f"    Keys: {list(data[0].keys())}")
            for k, v in list(data[0].items())[:6]:
                print(f"      {k}: {str(v)[:100]}")
            try:
                df = pd.DataFrame(data)
                print(f"    As DataFrame: {df.shape}")
                return df
            except Exception:
                pass
        elif data:
            print(f"    First item type: {type(data[0]).__name__}")
            print(f"    First item: {str(data[0])[:200]}")
        return data
    elif isinstance(data, dict):
        print(f"  {label}Dict: {len(data)} keys")
        for k in list(data.keys())[:10]:
            v = data[k]
            vtype = type(v).__name__
            vlen = len(v) if hasattr(v, "__len__") else "N/A"
            print(f"    {k} ({vtype}, len={vlen}): {str(v)[:80]}")
        return data
    else:
        print(f"  {label}{type(data).__name__}: {str(data)[:200]}")
        return data


def main():
    print("Apollo Deception Detection Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "deception-detection"
    if not repo_dir.exists():
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "deception" in candidate.name.lower():
                repo_dir = candidate
                break

    data_dir = repo_dir / "data"
    if not data_dir.exists():
        print(f"\nERROR: data directory not found at {data_dir}")
        return

    # List all sub-datasets
    print("\n" + "=" * 60)
    print("SUB-DATASETS IN data/")
    print("=" * 60)

    subdatasets = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Sub-datasets: {subdatasets}")

    dataset_catalog = []

    for subds_name in subdatasets:
        subds_dir = data_dir / subds_name
        print(f"\n{'=' * 60}")
        print(f"SUB-DATASET: {subds_name}")
        print(f"{'=' * 60}")

        # List files
        all_subds_files = []
        for root, dirs, files in os.walk(subds_dir):
            dirs[:] = [d for d in dirs if d != ".git"]
            for f in files:
                all_subds_files.append(os.path.join(root, f))

        data_extensions = (".json", ".csv", ".jsonl", ".yaml", ".yml")
        data_files = [f for f in all_subds_files if any(f.endswith(ext) for ext in data_extensions)]
        print(f"  Total files: {len(all_subds_files)}")
        print(f"  Data files: {len(data_files)}")
        for f in data_files:
            rel = os.path.relpath(f, data_dir)
            size_kb = os.path.getsize(f) / 1024
            print(f"    {rel} ({size_kb:.1f} KB)")

        # Load and explore each data file
        n_records_total = 0
        label_counts = {}

        for fpath in data_files:
            fname = os.path.basename(fpath)
            print(f"\n  --- {fname} ---")
            data = load_file(fpath)
            if data is None:
                continue

            result = summarize_data(data)

            # Try to find deception-related labels
            df = None
            if isinstance(result, pd.DataFrame):
                df = result
            elif isinstance(result, list) and result and isinstance(result[0], dict):
                try:
                    df = pd.DataFrame(result)
                except Exception:
                    pass

            if df is not None:
                n_records_total += len(df)

                # Search for deception/honesty labels
                label_keywords = ["decepti", "honest", "lie", "label", "truth", "deception",
                                  "class", "verdict", "grade", "score", "type", "category"]
                label_cols = [c for c in df.columns if any(kw in c.lower() for kw in label_keywords)]

                if label_cols:
                    print(f"  Label columns: {label_cols}")
                    for col in label_cols:
                        if df[col].nunique() <= 30:
                            vc = df[col].value_counts()
                            print(f"\n    {col} distribution:")
                            print(f"{vc.to_string()}")
                            for val, count in vc.items():
                                key = f"{subds_name}|{col}|{val}"
                                label_counts[key] = label_counts.get(key, 0) + count

                # Save as CSV
                out_name = f"apollo_{subds_name}_{Path(fpath).stem}.csv"
                try:
                    df.to_csv(PROCESSED_DIR / out_name, index=False)
                    print(f"  -> Saved to processed/{out_name}")
                except Exception as e:
                    print(f"  Could not save CSV: {e}")

        dataset_catalog.append({
            "subdataset": subds_name,
            "n_data_files": len(data_files),
            "n_total_files": len(all_subds_files),
            "n_records": n_records_total,
        })

    # Save dataset catalog
    print("\n" + "=" * 60)
    print("DATASET CATALOG")
    print("=" * 60)

    catalog_df = pd.DataFrame(dataset_catalog)
    print(catalog_df.to_string(index=False))
    catalog_df.to_csv(PROCESSED_DIR / "dataset_catalog.csv", index=False)
    print(f"\n  -> Saved to processed/dataset_catalog.csv")

    # Save label distribution
    if label_counts:
        print("\n" + "=" * 60)
        print("LABEL DISTRIBUTION ACROSS ALL DATASETS")
        print("=" * 60)
        label_rows = []
        for key, count in sorted(label_counts.items()):
            parts = key.split("|")
            label_rows.append({
                "subdataset": parts[0],
                "column": parts[1] if len(parts) > 1 else "",
                "value": parts[2] if len(parts) > 2 else "",
                "count": count,
            })
        label_df = pd.DataFrame(label_rows)
        print(label_df.to_string(index=False))
        label_df.to_csv(PROCESSED_DIR / "label_distribution.csv", index=False)
        print(f"\n  -> Saved to processed/label_distribution.csv")

    # Also explore rollouts directory specifically
    rollouts_dir = data_dir / "rollouts"
    if rollouts_dir.exists():
        print("\n" + "=" * 60)
        print("ROLLOUTS ANALYSIS")
        print("=" * 60)
        rollout_files = list(rollouts_dir.glob("*.json"))
        print(f"Rollout files: {len(rollout_files)}")

        rollout_summary = []
        for rf in rollout_files[:15]:
            try:
                with open(rf) as f:
                    data = json.load(f)
                n_items = len(data) if isinstance(data, list) else 1
                print(f"  {rf.name}: {n_items} items")
                rollout_summary.append({
                    "file": rf.name,
                    "n_items": n_items,
                    "type": type(data).__name__,
                })
            except Exception as e:
                print(f"  {rf.name}: Error - {e}")

        if rollout_summary:
            pd.DataFrame(rollout_summary).to_csv(PROCESSED_DIR / "rollouts_summary.csv", index=False)

    # Overall summary
    summary_rows = [
        {"metric": "n_subdatasets", "value": len(subdatasets)},
        {"metric": "total_records", "value": sum(d["n_records"] for d in dataset_catalog)},
        {"metric": "n_unique_labels", "value": len(label_counts)},
    ]
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"\n  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
