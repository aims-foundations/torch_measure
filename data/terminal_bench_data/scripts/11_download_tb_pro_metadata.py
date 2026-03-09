"""
Download Terminal-Bench Pro task metadata from HuggingFace.

TB Pro has 400 tasks (200 public + 200 private) across 8 domains.
Only the public 200 are accessible via HuggingFace.

Dataset: alibabagroup/terminal-bench-pro
"""

import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_BENCHMARK_DIR / "raw")


def main():
    print("Loading Terminal-Bench Pro from HuggingFace "
          "(alibabagroup/terminal-bench-pro)...")
    try:
        ds = load_dataset("alibabagroup/terminal-bench-pro", split="train")
    except Exception as e:
        print(f"Failed to load with split='train': {e}")
        print("Trying without split specification...")
        ds = load_dataset("alibabagroup/terminal-bench-pro")
        print(f"Available splits: {list(ds.keys())}")
        ds = ds[list(ds.keys())[0]]

    print(f"Loaded {len(ds)} rows")
    print(f"Columns: {ds.column_names}")

    # Extract metadata (skip large archive/binary columns)
    records = []
    for row in ds:
        record = {}
        for col in ds.column_names:
            val = row.get(col)
            # Skip binary/large columns
            if isinstance(val, (bytes, bytearray)):
                record[f"{col}_size_bytes"] = len(val) if val else 0
            elif col == "archive":
                record["has_archive"] = val is not None
            else:
                record[col] = val
        records.append(record)

    df = pd.DataFrame(records)

    # Save full metadata
    csv_path = f"{OUTPUT_DIR}/tb_pro_task_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} with {len(df)} tasks")

    # Save as JSON (preserves nested structures)
    json_path = f"{OUTPUT_DIR}/tb_pro_task_metadata.json"
    json_records = []
    for row in ds:
        rec = {}
        for col in ds.column_names:
            val = row.get(col)
            if isinstance(val, (bytes, bytearray)):
                rec[f"{col}_size_bytes"] = len(val) if val else 0
            elif col == "archive":
                continue  # skip binary archive
            else:
                rec[col] = val
        json_records.append(rec)

    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2, default=str)
    print(f"Saved {json_path}")

    # Print summary statistics
    print(f"\n=== Summary ===")
    print(f"Total tasks (public): {len(df)}")

    # Check for common metadata columns
    for col in ["category", "difficulty", "domain", "tags"]:
        if col in df.columns:
            print(f"\n{col.title()} distribution:")
            print(df[col].value_counts().to_string())

    # Check task_id patterns
    if "task_id" in df.columns:
        print(f"\nSample task IDs:")
        for tid in df["task_id"].head(10):
            print(f"  {tid}")

    # Check for instruction/description column
    for col in ["instruction", "base_description", "description"]:
        if col in df.columns:
            lengths = df[col].str.len()
            print(f"\n{col} length stats:")
            print(f"  Min: {lengths.min()}, Max: {lengths.max()}, "
                  f"Mean: {lengths.mean():.0f}, Median: {lengths.median():.0f}")
            break

    # Check overlap with TB 2.0
    tb2_path = f"{OUTPUT_DIR}/task_metadata.csv"
    try:
        tb2 = pd.read_csv(tb2_path)
        tb2_ids = set(tb2["task_id"].values)
        pro_ids = set(df["task_id"].values) if "task_id" in df.columns else set()
        overlap = tb2_ids & pro_ids
        print(f"\n=== Overlap with TB 2.0 ===")
        print(f"TB 2.0 tasks: {len(tb2_ids)}")
        print(f"TB Pro tasks (public): {len(pro_ids)}")
        print(f"Overlap: {len(overlap)}")
        print(f"New in Pro: {len(pro_ids - tb2_ids)}")
        if overlap:
            print(f"Shared tasks: {sorted(overlap)[:10]}...")
    except FileNotFoundError:
        print(f"\nCould not find TB 2.0 metadata at {tb2_path} for overlap check")


if __name__ == "__main__":
    main()
