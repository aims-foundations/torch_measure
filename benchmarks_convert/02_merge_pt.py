"""
Merge all individual benchmark .pt files into a single combined .pt file.

Reads every .pt in benchmarks/pt/ (except all_benchmarks.pt itself),
concatenates them, re-categorises string columns across the merged set,
and saves as benchmarks/pt/all_benchmarks.pt.

Usage:
    python benchmarks/merge_benchmarks.py
"""

from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PT_DIR = SCRIPT_DIR / "pt"
OUTPUT_PATH = PT_DIR / "all_benchmarks.pt"

STRING_COLS = [
    "dataset_name",
    "item_text",
    "test_taker",
    "item",
    "test_condition",
    "benchmark_url",
    "category",
]


def main():
    pt_files = sorted(PT_DIR.glob("*.pt"))
    pt_files = [f for f in pt_files if f.name != "all_benchmarks.pt"]

    if not pt_files:
        print("No .pt files found in", PT_DIR)
        return

    print(f"Found {len(pt_files)} benchmark files:\n")

    frames = []
    for f in pt_files:
        df = torch.load(f, weights_only=False)
        print(f"  {f.stem:30s}  {len(df):>10,} rows")
        # Convert categoricals to plain strings before concat
        for col in STRING_COLS:
            if col in df.columns and df[col].dtype.name == "category":
                df[col] = df[col].astype(str)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  {'TOTAL':30s}  {len(merged):>10,} rows")

    # Re-apply categorical dtype across merged set
    for col in STRING_COLS:
        if col in merged.columns:
            merged[col] = merged[col].astype("category")

    merged["response"] = merged["response"].astype("float32")

    torch.save(merged, OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {OUTPUT_PATH.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
