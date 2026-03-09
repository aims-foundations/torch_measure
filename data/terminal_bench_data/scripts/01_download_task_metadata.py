"""
Download Terminal-Bench task metadata from HuggingFace.

Extracts: task_id, category, difficulty, base_description, tags,
max_agent_timeout_sec, max_test_timeout_sec for all 89 tasks.
"""

import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_BENCHMARK_DIR / "raw")


def main():
    print("Loading Terminal-Bench dataset from HuggingFace (ia03/terminal-bench)...")
    ds = load_dataset("ia03/terminal-bench", split="test")
    print(f"Loaded {len(ds)} rows")

    # Print column names
    print(f"Columns: {ds.column_names}")

    # Extract metadata (skip the large archive column)
    records = []
    for row in ds:
        record = {
            "task_id": row.get("task_id"),
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "base_description": row.get("base_description"),
            "tags": row.get("tags"),
            "max_agent_timeout_sec": row.get("max_agent_timeout_sec"),
            "max_test_timeout_sec": row.get("max_test_timeout_sec"),
            "archive_bytes": row.get("archive_bytes"),
            "n_files": row.get("n_files"),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Save full metadata
    df.to_csv(f"{OUTPUT_DIR}/task_metadata.csv", index=False)
    print(f"\nSaved task_metadata.csv with {len(df)} tasks")

    # Save as JSON too (preserves lists like tags)
    with open(f"{OUTPUT_DIR}/task_metadata.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved task_metadata.json")

    # Print summary statistics
    print(f"\n=== Summary ===")
    print(f"Total tasks: {len(df)}")
    print(f"\nDifficulty distribution:")
    print(df["difficulty"].value_counts().to_string())
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nDescription length stats:")
    desc_lens = df["base_description"].str.len()
    print(f"  Min: {desc_lens.min()}, Max: {desc_lens.max()}, "
          f"Mean: {desc_lens.mean():.0f}, Median: {desc_lens.median():.0f}")


if __name__ == "__main__":
    main()
