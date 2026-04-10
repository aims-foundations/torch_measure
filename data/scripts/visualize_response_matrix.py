#!/usr/bin/env python3
"""
Visualize all response matrices across benchmarks.

Discovers every response_matrix*.csv under data/*_data/processed/ and produces
a standard set of plots for each:
  1. Full heatmap (models x items, sorted by accuracy / difficulty)
  2. Model accuracy bar chart
  3. Item difficulty histogram
  4. Model–model correlation heatmap (clustered)

Usage:
    python data/scripts/visualize_response_matrix.py              # all benchmarks
    python data/scripts/visualize_response_matrix.py bfcl_data    # one benchmark
    python data/scripts/visualize_response_matrix.py bfcl_data swebench_data  # several
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", font_scale=0.9)

BASE_DIR = Path(__file__).resolve().parent.parent  # data/


# ── helpers ──────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    """Save a figure as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _label(csv_path: Path) -> str:
    """Derive a human-readable label from the CSV path.

    e.g.  bfcl/processed/response_matrix.csv                -> "bfcl"
          bigcodebench/processed/response_matrix_instruct.csv -> "bigcodebench / instruct"
    """
    bench = csv_path.parent.parent.name
    stem = csv_path.stem  # e.g. "response_matrix_instruct"
    variant = stem.removeprefix("response_matrix").lstrip("_")
    if variant:
        return f"{bench} / {variant}"
    return bench


# ── plots ────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: pd.DataFrame, label: str, out_path: Path):
    """Full heatmap: models (rows) x items (columns)."""
    df = matrix.copy()
    orig_n_models, orig_n_items = df.shape

    # Sort rows by mean score (best on top), columns by difficulty (easiest left)
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]
    df = df[df.mean(axis=0).sort_values(ascending=False).index]

    # Downsample to keep figure sizes reasonable (max ~1M cells in the plot).
    # Use linspace sampling to preserve the top-to-bottom and left-to-right ordering.
    MAX_ROWS = 1000
    MAX_COLS = 2000
    if len(df) > MAX_ROWS:
        idx = np.linspace(0, len(df) - 1, MAX_ROWS, dtype=int)
        df = df.iloc[idx]
    if df.shape[1] > MAX_COLS:
        idx = np.linspace(0, df.shape[1] - 1, MAX_COLS, dtype=int)
        df = df.iloc[:, idx]
    n_models, n_items = df.shape

    height = max(6, min(16, n_models * 0.18))
    width = max(10, min(20, n_items * 0.02 + 4))
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(
        df.values, ax=ax,
        cmap=sns.color_palette("RdYlGn", as_cmap=True),
        vmin=0, vmax=1,
        cbar_kws={"label": "Score", "shrink": 0.4},
        xticklabels=False,
        yticklabels=False,  # always off: too many rows for readable labels anyway
    )
    ax.set_ylabel(f"Model (sorted by mean score) — {orig_n_models} total")
    ax.set_xlabel(f"Items (sorted by difficulty) — {orig_n_items} total")
    note = "" if (n_models, n_items) == (orig_n_models, orig_n_items) else " [downsampled]"
    ax.set_title(f"{label}  ({orig_n_models} x {orig_n_items}){note}", fontweight="bold")

    _save(fig, out_path)
    print(f"    {out_path.name}  ({orig_n_models} x {orig_n_items})")


# ── main ─────────────────────────────────────────────────────────────────

def visualize_one(csv_path: Path):
    """Load a single response matrix CSV and save a heatmap next to it.

    Output path: same directory as the CSV, same stem but .png extension.
      processed/response_matrix.csv            -> processed/response_matrix.png
      processed/response_matrix_instruct.csv   -> processed/response_matrix_instruct.png
    """
    label = _label(csv_path)
    out_path = csv_path.with_suffix(".png")

    print(f"  [{label}]  {csv_path.relative_to(BASE_DIR)}")

    matrix = pd.read_csv(csv_path, index_col=0)
    if matrix.empty:
        print("    SKIPPED (empty matrix)")
        return

    # Coerce all columns to numeric, dropping non-numeric metadata columns
    # (e.g. scienceagentbench's `domain`, `subtask_categories` cols mixed in).
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.dropna(axis=1, how="all")  # drop fully-NaN columns
    matrix = matrix.dropna(axis=0, how="all")  # drop fully-NaN rows
    if matrix.empty or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        print("    SKIPPED (matrix too small after coercion)")
        return

    plot_heatmap(matrix, label, out_path)


def discover(benchmarks: list[str] | None = None) -> list[Path]:
    """Find all response_matrix*.csv files, optionally filtered by benchmark."""
    pattern = "*/processed/response_matrix*.csv"
    all_csvs = sorted(BASE_DIR.glob(pattern))
    if benchmarks:
        all_csvs = [p for p in all_csvs
                     if p.parent.parent.name in benchmarks
                     or p.parent.parent.name in benchmarks]
    return all_csvs


def main():
    parser = argparse.ArgumentParser(
        description="Visualize response matrices for all (or selected) benchmarks."
    )
    parser.add_argument(
        "benchmarks", nargs="*", default=None,
        help="Optional benchmark names to process (e.g. bfcl_data or bfcl). "
             "If omitted, processes all benchmarks.",
    )
    args = parser.parse_args()

    csvs = discover(args.benchmarks or None)
    if not csvs:
        print("No response_matrix*.csv files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csvs)} response matrices across "
          f"{len({p.parent.parent.name for p in csvs})} benchmarks.\n")

    for csv_path in csvs:
        try:
            visualize_one(csv_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
