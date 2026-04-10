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
from scipy.cluster.hierarchy import leaves_list, linkage

sns.set_theme(style="white", font_scale=0.9)

BASE_DIR = Path(__file__).resolve().parent.parent  # data/


# ── helpers ──────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    """Save a figure as both PDF and PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _label(csv_path: Path) -> str:
    """Derive a human-readable label from the CSV path.

    e.g.  bfcl_data/processed/response_matrix.csv           -> "bfcl"
          bigcodebench_data/processed/response_matrix_instruct.csv -> "bigcodebench / instruct"
    """
    bench = csv_path.parent.parent.name
    stem = csv_path.stem  # e.g. "response_matrix_instruct"
    variant = stem.removeprefix("response_matrix").lstrip("_")
    if variant:
        return f"{bench} / {variant}"
    return bench


def _prefix(csv_path: Path) -> str:
    """Filename prefix for saving figures (no spaces, filesystem-safe)."""
    bench = csv_path.parent.parent.name
    variant = csv_path.stem.removeprefix("response_matrix").lstrip("_")
    if variant:
        return f"{bench}_{variant}"
    return bench


# ── plots ────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: pd.DataFrame, label: str, fig_dir: Path, prefix: str):
    """Full heatmap: models (rows) x items (columns)."""
    df = matrix.copy()
    n_models, n_items = df.shape

    # Sort rows by mean score (best on top), columns by difficulty (easiest left)
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]
    df = df[df.mean(axis=0).sort_values(ascending=False).index]

    height = max(6, n_models * 0.18)
    width = max(10, min(28, n_items * 0.02 + 4))
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(
        df.values, ax=ax,
        cmap=sns.color_palette("RdYlGn", as_cmap=True),
        vmin=0, vmax=1,
        cbar_kws={"label": "Score", "shrink": 0.4},
        xticklabels=False,
        yticklabels=True,
    )
    ax.set_ylabel("Model (sorted by mean score)")
    ax.set_xlabel(f"Items ({n_items})")
    ax.set_title(f"{label}  ({n_models} models x {n_items} items)", fontweight="bold")
    plt.yticks(fontsize=max(3, min(7, 300 / n_models)))

    _save(fig, fig_dir / f"{prefix}_heatmap")
    print(f"    heatmap  ({n_models} x {n_items})")


def plot_model_accuracy(matrix: pd.DataFrame, label: str, fig_dir: Path, prefix: str):
    """Horizontal bar chart of per-model mean scores."""
    scores = matrix.mean(axis=1).sort_values(ascending=True)
    n = len(scores)

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.2)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), scores.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(scores.index, fontsize=max(4, min(9, 200 / n)))
    ax.set_xlabel("Mean Score")
    ax.set_title(f"{label} — Model Scores", fontweight="bold")
    ax.set_xlim(0, min(1.05, scores.max() * 1.15))

    for i, v in enumerate(scores.values):
        ax.text(v + scores.max() * 0.01, i, f"{v:.3f}", va="center",
                fontsize=max(3, min(7, 200 / n)))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _save(fig, fig_dir / f"{prefix}_model_accuracy")
    print(f"    model_accuracy  ({n} models)")


def plot_item_difficulty(matrix: pd.DataFrame, label: str, fig_dir: Path, prefix: str):
    """Histogram of per-item mean scores (difficulty distribution)."""
    item_scores = matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(item_scores, bins=40, color=sns.color_palette("Set2")[0],
            edgecolor="white", linewidth=0.5)
    ax.axvline(item_scores.median(), color="red", linestyle="--", alpha=0.7,
               label=f"median = {item_scores.median():.3f}")
    ax.set_xlabel("Item Mean Score (across models)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} — Item Difficulty Distribution ({len(item_scores)} items)",
                 fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _save(fig, fig_dir / f"{prefix}_item_difficulty")
    print(f"    item_difficulty  ({len(item_scores)} items)")


def plot_model_correlation(matrix: pd.DataFrame, label: str, fig_dir: Path, prefix: str):
    """Clustered model–model Pearson correlation heatmap."""
    n_models = matrix.shape[0]
    if n_models < 3:
        print("    model_correlation  SKIPPED (< 3 models)")
        return

    corr = matrix.T.corr()

    # Hierarchical clustering for ordering
    try:
        link = linkage(corr.values, method="ward")
        order = leaves_list(link)
        corr = corr.iloc[order, order]
    except Exception:
        pass  # fall back to original order

    size = max(6, min(16, n_models * 0.25))
    fig, ax = plt.subplots(figsize=(size, size))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.3, vmax=1,
        mask=mask, square=True,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    fontsize = max(4, min(9, 200 / n_models))
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_title(f"{label} — Model Correlation (clustered)", fontweight="bold")
    plt.tight_layout()

    _save(fig, fig_dir / f"{prefix}_model_correlation")
    print(f"    model_correlation  ({n_models} models)")


# ── main ─────────────────────────────────────────────────────────────────

def visualize_one(csv_path: Path):
    """Load a single response matrix CSV and produce all plots."""
    label = _label(csv_path)
    prefix = _prefix(csv_path)
    fig_dir = csv_path.parent.parent / "figures"

    print(f"  [{label}]  {csv_path.relative_to(BASE_DIR)}")

    matrix = pd.read_csv(csv_path, index_col=0)
    if matrix.empty:
        print("    SKIPPED (empty matrix)")
        return

    plot_heatmap(matrix, label, fig_dir, prefix)
    plot_model_accuracy(matrix, label, fig_dir, prefix)
    plot_item_difficulty(matrix, label, fig_dir, prefix)
    plot_model_correlation(matrix, label, fig_dir, prefix)


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
