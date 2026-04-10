#!/usr/bin/env python3
"""
Compute statistics over the three benchmark lists in reproduce.py.

Counts:
  - Number of benchmarks, response matrices, and variants
  - Total items (unique and summed across variants)
  - Total cells (subject × item response values)
  - Shape of each benchmark's primary matrix
  - Binary vs continuous distribution

Usage:
    python data/scripts/dataset_stats.py              # summary of all three lists
    python data/scripts/dataset_stats.py --list BENCHMARKS
    python data/scripts/dataset_stats.py --top 20     # show top N by item count
    python data/scripts/dataset_stats.py --full       # print every benchmark
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def _parse_list(source: str, list_name: str) -> list[str]:
    """Extract a benchmark name list from reproduce.py source."""
    # Match `BENCHMARKS = [ ... ]` followed by `#` (the next comment block)
    pattern = rf"{list_name}\s*=\s*\[(.*?)\]\s*\n\s*(?:#|\Z)"
    m = re.search(pattern, source, re.DOTALL)
    if not m:
        return []
    return re.findall(r'"(\w+)"', m.group(1))


def load_benchmark_lists() -> dict[str, list[str]]:
    """Read BENCHMARKS, BENCHMARKS_AGGREGATE, and BENCHMARKS_PENDING from reproduce.py."""
    source = (BASE_DIR / "reproduce.py").read_text()
    return {
        "BENCHMARKS": _parse_list(source, "BENCHMARKS"),
        "BENCHMARKS_AGGREGATE": _parse_list(source, "BENCHMARKS_AGGREGATE"),
        "BENCHMARKS_PENDING": _parse_list(source, "BENCHMARKS_PENDING"),
    }


def _classify(df: pd.DataFrame) -> str:
    """Return 'binary' if all non-NaN values are 0 or 1, else 'continuous'."""
    vals = df.values.flatten()
    vals = vals[~pd.isna(vals)]
    if len(vals) == 0:
        return "—"
    try:
        return "binary" if set(vals.tolist()).issubset({0.0, 1.0}) else "continuous"
    except TypeError:
        return "—"


def collect_stats(benchmark_names: list[str]) -> dict:
    """Collect statistics over a set of benchmarks.

    Returns a dict with:
      - benchmarks: list of per-benchmark dicts (name, shape, matrices, items_total, is_binary, kinds)
      - n_benchmarks: number of benchmarks with at least one valid matrix
      - n_matrices: total matrix variants across all benchmarks
      - items_primary: sum of largest variant's item count per benchmark
      - items_summed: sum of all items across all variants
      - cells_total: total number of (subject, item) cells across all matrices
      - binary_count: number of matrices where all cells are 0 or 1
      - continuous_count: number of matrices with non-binary values
    """
    per_bench = []
    n_matrices = 0
    items_summed = 0
    cells_total = 0
    binary_count = 0
    continuous_count = 0

    for bench in sorted(benchmark_names):
        rms = sorted((BASE_DIR / bench / "processed").glob("response_matrix*.csv"))
        if not rms:
            continue

        max_items = 0
        bench_items_total = 0
        bench_matrices = []
        for rm in rms:
            try:
                df = pd.read_csv(rm, index_col=0)
                df = df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                continue
            n_rows, n_cols = df.shape
            if n_rows < 2 or n_cols < 2:
                continue

            kind = _classify(df)
            if kind == "binary":
                binary_count += 1
            elif kind == "continuous":
                continuous_count += 1

            n_matrices += 1
            items_summed += n_cols
            cells_total += n_rows * n_cols
            max_items = max(max_items, n_cols)
            bench_items_total += n_cols

            variant = rm.stem.removeprefix("response_matrix").lstrip("_") or ""
            name = f"{bench}_{variant}" if variant else bench
            bench_matrices.append({
                "name": name,
                "shape": (n_rows, n_cols),
                "kind": kind,
            })

        if bench_matrices:
            per_bench.append({
                "name": bench,
                "primary_items": max_items,
                "total_items": bench_items_total,
                "n_matrices": len(bench_matrices),
                "matrices": bench_matrices,
            })

    items_primary = sum(b["primary_items"] for b in per_bench)

    return {
        "benchmarks": per_bench,
        "n_benchmarks": len(per_bench),
        "n_matrices": n_matrices,
        "items_primary": items_primary,
        "items_summed": items_summed,
        "cells_total": cells_total,
        "binary_count": binary_count,
        "continuous_count": continuous_count,
    }


def print_summary(label: str, stats: dict, top: int = 0, full: bool = False):
    """Print a formatted summary of collected stats."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"Benchmarks with at least one valid matrix : {stats['n_benchmarks']:>10,}")
    print(f"Total response matrices (with variants)   : {stats['n_matrices']:>10,}")
    print(f"Total items (largest variant per benchmark): {stats['items_primary']:>10,}")
    print(f"Total items (summed across all variants)  : {stats['items_summed']:>10,}")
    print(f"Total cells (subject × item values)       : {stats['cells_total']:>10,}")
    if stats["binary_count"] or stats["continuous_count"]:
        print(f"Matrices: binary={stats['binary_count']}, continuous={stats['continuous_count']}")

    if full:
        print(f"\nAll {stats['n_benchmarks']} benchmarks:")
        for b in sorted(stats["benchmarks"], key=lambda x: -x["primary_items"]):
            print(f"  {b['name']:30s} {b['primary_items']:>10,} items  "
                  f"({b['n_matrices']} variants)")
    elif top > 0:
        print(f"\nTop {top} benchmarks by item count:")
        for b in sorted(stats["benchmarks"], key=lambda x: -x["primary_items"])[:top]:
            print(f"  {b['name']:30s} {b['primary_items']:>10,}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics over benchmark lists in reproduce.py."
    )
    parser.add_argument(
        "--list", choices=["BENCHMARKS", "BENCHMARKS_AGGREGATE", "BENCHMARKS_PENDING", "all"],
        default="all", help="Which list to summarize",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Show top N benchmarks by item count (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show every benchmark instead of top N",
    )
    args = parser.parse_args()

    all_lists = load_benchmark_lists()
    if args.list == "all":
        targets = all_lists
    else:
        targets = {args.list: all_lists[args.list]}

    for label, names in targets.items():
        if not names:
            continue
        stats = collect_stats(names)
        print_summary(label, stats, top=args.top, full=args.full)


if __name__ == "__main__":
    main()
