"""
Populate a SQLite database from the merged all_benchmarks.pt file.

Creates benchmarks/benchmark_responses.sqlite with a single `responses` table
and useful indexes.

Usage:
    python benchmarks/populate_sqlite.py
    python benchmarks/populate_sqlite.py --input benchmarks/pt/all_benchmarks.pt
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "pt" / "all_benchmarks.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "benchmark_responses.sqlite"


def main():
    parser = argparse.ArgumentParser(description="Populate SQLite from merged .pt")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        print("Run merge_benchmarks.py first.")
        return

    print(f"Loading {args.input} ...")
    df = torch.load(args.input, weights_only=False)

    # Convert categoricals to plain strings for SQLite
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str)

    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Remove existing DB
    if args.output.exists():
        args.output.unlink()

    print(f"Writing to {args.output} ...")
    conn = sqlite3.connect(str(args.output))

    df.to_sql("responses", conn, index=False, if_exists="replace", chunksize=50_000)

    # Create indexes for common query patterns
    print("Creating indexes ...")
    conn.execute("CREATE INDEX idx_dataset ON responses(dataset_name)")
    conn.execute("CREATE INDEX idx_test_taker ON responses(test_taker)")
    conn.execute("CREATE INDEX idx_item ON responses(item)")
    conn.execute("CREATE INDEX idx_dataset_taker ON responses(dataset_name, test_taker)")

    # Write summary as metadata
    summary = {
        "total_rows": len(df),
        "benchmarks": sorted(df["dataset_name"].unique().tolist()),
        "n_benchmarks": df["dataset_name"].nunique(),
        "n_test_takers": df["test_taker"].nunique(),
        "n_items": df["item"].nunique(),
    }

    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    import json
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("summary", json.dumps(summary, indent=2)),
    )
    conn.commit()

    # Print summary
    print(f"\n  Benchmarks:   {summary['n_benchmarks']}")
    print(f"  Test takers:  {summary['n_test_takers']:,}")
    print(f"  Items:        {summary['n_items']:,}")
    print(f"  Total rows:   {summary['total_rows']:,}")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\n  DB size: {size_mb:.1f} MB")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
