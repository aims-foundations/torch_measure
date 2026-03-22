#!/usr/bin/env python3
"""
Visualize the African NLP benchmark collection.

Produces:
1. Dataset composition bar chart (items per dataset per language)
2. Language coverage heatmap (languages x datasets)
3. Task type distribution pie chart
4. Text length distributions by dataset
5. Label distribution by dataset
6. Language family coverage map
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BENCHMARK_DIR / "processed"
FIG_DIR = _BENCHMARK_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=0.9)

# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_data():
    """Load all processed data."""
    meta = pd.read_csv(DATA_DIR / "task_metadata.csv")
    summary = pd.read_csv(DATA_DIR / "summary_stats.csv")
    return meta, summary


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Dataset composition
# ──────────────────────────────────────────────────────────────────────

def plot_dataset_composition(meta):
    """Bar chart showing number of items per dataset."""
    counts = meta.groupby("source_dataset").size().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(counts))
    bars = ax.barh(counts.index, counts.values, color=colors)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", ha="left", va="center", fontsize=9)

    ax.set_xlabel("Number of Items")
    ax.set_title("African NLP Benchmark Collection: Items per Dataset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"dataset_composition.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved dataset_composition.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Language coverage heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_language_coverage(meta):
    """Heatmap showing which languages are covered by which datasets."""
    # Create pivot: language x dataset with item counts
    pivot = meta.groupby(["language", "source_dataset"]).size().unstack(fill_value=0)

    # Sort languages by total count
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot = pivot.drop("total", axis=1)

    # Limit to top 30 languages for readability
    if len(pivot) > 30:
        pivot = pivot.head(30)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.35)))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    # Use log scale for better visibility
    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=1, vmax=pivot.values.max() + 1)

    # Replace 0 with NaN for display
    display_df = pivot.replace(0, np.nan)

    sns.heatmap(display_df, cmap=cmap, norm=norm, annot=pivot.values,
                fmt="d", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Number of items (log scale)"})

    ax.set_title("Language Coverage Across African NLP Benchmarks")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Language")
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"language_coverage.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved language_coverage.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Task type distribution
# ──────────────────────────────────────────────────────────────────────

def plot_task_distribution(meta):
    """Pie chart showing task type distribution."""
    task_counts = meta["task_type"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    colors = sns.color_palette("Set2", len(task_counts))
    wedges, texts, autotexts = axes[0].pie(
        task_counts.values, labels=task_counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90
    )
    axes[0].set_title("Distribution by Task Type")

    # Bar chart of languages per task
    task_lang = meta.groupby("task_type")["language"].nunique().sort_values(ascending=True)
    axes[1].barh(task_lang.index, task_lang.values, color=colors[:len(task_lang)])
    axes[1].set_xlabel("Number of Languages")
    axes[1].set_title("Language Diversity per Task Type")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"task_distribution.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_distribution.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 4: Text length distributions
# ──────────────────────────────────────────────────────────────────────

def plot_text_lengths(meta):
    """Box plot of text lengths by dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = meta["source_dataset"].unique()
    colors = sns.color_palette("Set2", len(datasets))

    # Cap n_chars at 99th percentile for visibility
    cap = meta["n_chars"].quantile(0.99)
    plot_data = meta[meta["n_chars"] <= cap]

    sns.boxplot(data=plot_data, x="source_dataset", y="n_chars",
                palette="Set2", ax=ax)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Text Length (characters)")
    ax.set_title("Text Length Distribution by Dataset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"text_lengths.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved text_lengths.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 5: Label distribution
# ──────────────────────────────────────────────────────────────────────

def plot_label_distribution(meta):
    """Stacked bar chart of label distributions per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    datasets = meta["source_dataset"].unique()
    for i, ds in enumerate(datasets[:4]):
        ax = axes[i]
        subset = meta[meta["source_dataset"] == ds]
        label_counts = subset["label"].value_counts().head(10)

        colors = sns.color_palette("Set3", len(label_counts))
        bars = ax.barh(label_counts.index.astype(str), label_counts.values, color=colors)
        ax.set_title(f"{ds} — Label Distribution (top 10)")
        ax.set_xlabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Handle remaining datasets if any
    if len(datasets) > 4:
        # Remove the 4th subplot and replace with the 5th dataset
        pass

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"label_distribution.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved label_distribution.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 6: Split distribution
# ──────────────────────────────────────────────────────────────────────

def plot_split_distribution(meta):
    """Grouped bar chart of split distribution by dataset."""
    pivot = meta.groupby(["source_dataset", "split"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, colormap="Set2")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Items")
    ax.set_title("Data Split Distribution by Dataset")
    ax.legend(title="Split")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"split_distribution.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved split_distribution.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 7: Summary stats overview
# ──────────────────────────────────────────────────────────────────────

def plot_summary_overview(summary):
    """Overview plot from summary statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Items per language-dataset pair
    top_pairs = summary.nlargest(20, "n_items")
    labels = [f"{row['language']} ({row['source_dataset']})" for _, row in top_pairs.iterrows()]
    colors = sns.color_palette("husl", len(top_pairs))

    axes[0].barh(labels, top_pairs["n_items"].values, color=colors)
    axes[0].set_xlabel("Number of Items")
    axes[0].set_title("Top 20 Language-Dataset Pairs by Size")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Average text length by dataset
    avg_chars = summary.groupby("source_dataset")["avg_n_chars"].mean().sort_values()
    colors2 = sns.color_palette("Set2", len(avg_chars))
    axes[1].barh(avg_chars.index, avg_chars.values, color=colors2)
    axes[1].set_xlabel("Average Text Length (characters)")
    axes[1].set_title("Average Text Length by Dataset")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"summary_overview.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_overview.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Visualizing African NLP Benchmark Collection")
    print("=" * 70)

    meta, summary = load_data()

    print(f"\nLoaded {len(meta)} items, {len(summary)} summary rows\n")

    plot_dataset_composition(meta)
    plot_language_coverage(meta)
    plot_task_distribution(meta)
    plot_text_lengths(meta)
    plot_label_distribution(meta)
    plot_split_distribution(meta)
    plot_summary_overview(summary)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
