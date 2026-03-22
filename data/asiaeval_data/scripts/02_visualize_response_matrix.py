#!/usr/bin/env python3
"""
Visualize the AsiaEval benchmark collection metadata.

Produces:
1. Dataset coverage heatmap (languages x datasets)
2. Language distribution bar chart
3. Task type distribution
4. Label distribution per dataset
5. Dataset-language item count heatmap
6. Split distribution pie chart
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BENCHMARK_DIR / "processed"
FIG_DIR = _BENCHMARK_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=0.9)


def load_data():
    """Load processed AsiaEval data."""
    metadata = pd.read_csv(DATA_DIR / "task_metadata.csv")
    print(f"Loaded task_metadata.csv: {len(metadata)} rows")
    print(f"  Datasets: {sorted(metadata['source_dataset'].unique())}")
    print(f"  Languages: {metadata['language'].nunique()}")
    print(f"  Task types: {sorted(metadata['task_type'].unique())}")
    return metadata


def plot_dataset_language_heatmap(df):
    """Heatmap showing item counts per (dataset, language) pair."""
    pivot = df.groupby(["source_dataset", "language"]).size().unstack(fill_value=0)

    # Sort languages by total items
    lang_totals = pivot.sum(axis=0).sort_values(ascending=False)
    pivot = pivot[lang_totals.index]

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.5), 6))

    # Use log scale for better visibility since counts vary a lot
    from matplotlib.colors import LogNorm
    data = pivot.values.astype(float)
    data[data == 0] = np.nan  # mask zeros

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cmap.set_bad("white")

    sns.heatmap(
        pivot.where(pivot > 0),
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 7},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Number of Items", "shrink": 0.6},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(
        f"AsiaEval — Dataset x Language Coverage ({len(df)} total items)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Dataset")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap_dataset_language.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "heatmap_dataset_language.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_dataset_language.pdf/png")


def plot_language_distribution(df):
    """Horizontal bar chart of items per language."""
    lang_counts = df.groupby("language").size().sort_values(ascending=True)
    n = len(lang_counts)

    fig, ax = plt.subplots(figsize=(10, max(8, n * 0.35)))
    colors = sns.color_palette("viridis", n)
    bars = ax.barh(range(n), lang_counts.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(lang_counts.index, fontsize=8)
    ax.set_xlabel("Number of Items")
    ax.set_title(
        f"AsiaEval — Items per Language ({n} languages, {len(df)} total items)",
        fontsize=14, fontweight="bold",
    )

    for i, (lang, count) in enumerate(lang_counts.items()):
        ax.text(count + max(lang_counts) * 0.01, i, f"{count:,}",
                va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "language_distribution.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "language_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved language_distribution.pdf/png")


def plot_task_type_distribution(df):
    """Stacked bar chart: task types per dataset."""
    pivot = df.groupby(["source_dataset", "task_type"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               colormap="Set2", edgecolor="white", linewidth=0.5)
    ax.set_title("AsiaEval — Task Type Distribution by Dataset",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Items")
    ax.legend(title="Task Type", bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task_type_distribution.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "task_type_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_type_distribution.pdf/png")


def plot_label_distribution(df):
    """Per-dataset label distribution (top labels)."""
    datasets = sorted(df["source_dataset"].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        subset = df[df["source_dataset"] == ds_name]
        label_counts = subset["label"].value_counts().head(10)
        colors = sns.color_palette("pastel", len(label_counts))
        ax.barh(range(len(label_counts)), label_counts.values, color=colors)
        ax.set_yticks(range(len(label_counts)))
        ax.set_yticklabels(label_counts.index, fontsize=7)
        ax.set_xlabel("Count")
        ax.set_title(f"{ds_name}\n(n={len(subset):,})", fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle("AsiaEval — Label Distribution per Dataset",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_distribution.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved label_distribution.pdf/png")


def plot_split_distribution(df):
    """Show how items are distributed across splits."""
    split_counts = df.groupby(["source_dataset", "split"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    split_counts.plot(kind="bar", ax=ax, colormap="Paired",
                      edgecolor="white", linewidth=0.5)
    ax.set_title("AsiaEval — Split Distribution per Dataset",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Items")
    ax.legend(title="Split", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "split_distribution.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "split_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved split_distribution.pdf/png")


def plot_coverage_summary(df):
    """Summary table/figure showing overall dataset coverage statistics."""
    stats = []
    for ds_name, grp in df.groupby("source_dataset"):
        stats.append({
            "Dataset": ds_name,
            "Items": len(grp),
            "Languages": grp["language"].nunique(),
            "Task Types": grp["task_type"].nunique(),
            "Labels": grp["label"].nunique(),
            "Splits": grp["split"].nunique(),
        })
    stats_df = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(stats_df.columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(stats_df) + 1):
        color = "#D6E4F0" if i % 2 == 0 else "white"
        for j in range(len(stats_df.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("AsiaEval — Dataset Coverage Summary",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "coverage_summary.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "coverage_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved coverage_summary.pdf/png")


def main():
    df = load_data()
    print(f"\nGenerating figures in {FIG_DIR}/\n")

    plot_dataset_language_heatmap(df)
    plot_language_distribution(df)
    plot_task_type_distribution(df)
    plot_label_distribution(df)
    plot_split_distribution(df)
    plot_coverage_summary(df)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
