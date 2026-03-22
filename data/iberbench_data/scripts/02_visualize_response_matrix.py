#!/usr/bin/env python3
"""
Visualize the IberBench & Latin American Spanish NLP benchmark collection.

Produces:
1. Label distribution bar charts per variety (grouped by task)
2. Dataset size comparison chart
3. Language variety comparison across tasks

All figures saved to figures/ directory.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BENCHMARK_DIR / "processed"
FIG_DIR = _BENCHMARK_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# Task-specific color palettes
TASK_COLORS = {
    "irony_detection": "#E74C3C",
    "sentiment_analysis": "#3498DB",
    "aggressiveness_detection": "#E67E22",
    "lgbtphobia_detection": "#9B59B6",
}

VARIETY_ORDER = [
    "spain", "mexico", "cuba", "peru", "costa_rica", "uruguay",
]


def load_data():
    """Load the task metadata."""
    meta_path = DATA_DIR / "task_metadata.csv"
    df = pd.read_csv(meta_path)
    print(f"Loaded task_metadata.csv: {df.shape[0]} items, "
          f"{df['source_dataset'].nunique()} source datasets")
    return df


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Label distribution per source dataset
# ──────────────────────────────────────────────────────────────────────
def plot_label_distributions(df):
    """Bar charts showing label distribution for each source dataset."""
    datasets = sorted(df["source_dataset"].unique())
    n_datasets = len(datasets)

    # Layout: 4 columns, as many rows as needed
    n_cols = 4
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["source_dataset"] == ds_name]
        task = subset["task"].iloc[0]
        variety = subset["language_variety"].iloc[0]

        # Count labels
        label_counts = subset["label"].value_counts().sort_index()
        colors = plt.cm.Set2(np.linspace(0, 1, len(label_counts)))

        bars = ax.bar(
            range(len(label_counts)),
            label_counts.values,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(label_counts)))
        ax.set_xticklabels(label_counts.index, rotation=45, ha="right",
                           fontsize=8)
        ax.set_ylabel("Count")

        # Title with dataset info
        short_name = ds_name.replace("iberbench_", "").replace("_", " ").title()
        ax.set_title(f"{short_name}\n({variety}, n={len(subset)})", fontsize=10)

        # Add count labels on bars
        for bar, val in zip(bars, label_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=7)

    # Hide unused axes
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Label Distribution per Source Dataset",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_distributions.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(FIG_DIR / "label_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved label_distributions.pdf/png")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Dataset size comparison
# ──────────────────────────────────────────────────────────────────────
def plot_dataset_sizes(df):
    """Horizontal bar chart comparing dataset sizes, colored by task."""
    ds_sizes = (
        df.groupby(["source_dataset", "task", "language_variety"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(ds_sizes) * 0.5)))

    # Color by task
    colors = [TASK_COLORS.get(t, "#95A5A6") for t in ds_sizes["task"]]

    bars = ax.barh(
        range(len(ds_sizes)),
        ds_sizes["count"].values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Y-axis labels: dataset name + variety
    labels = []
    for _, row in ds_sizes.iterrows():
        name = row["source_dataset"].replace("iberbench_", "")
        labels.append(f"{name} ({row['language_variety']})")
    ax.set_yticks(range(len(ds_sizes)))
    ax.set_yticklabels(labels, fontsize=9)

    # Add count labels
    for i, (bar, val) in enumerate(zip(bars, ds_sizes["count"].values)):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)

    ax.set_xlabel("Number of Items")
    ax.set_title("Dataset Size Comparison",
                 fontsize=14, fontweight="bold")

    # Legend for tasks
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=task.replace("_", " ").title())
        for task, color in TASK_COLORS.items()
        if task in ds_sizes["task"].values
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "dataset_sizes.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "dataset_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved dataset_sizes.pdf/png")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Language variety comparison across tasks
# ──────────────────────────────────────────────────────────────────────
def plot_variety_task_heatmap(df):
    """Heatmap showing item counts per (variety, task) combination."""
    # Build pivot table
    pivot = (
        df.groupby(["language_variety", "task"])
        .size()
        .reset_index(name="count")
        .pivot(index="language_variety", columns="task", values="count")
        .fillna(0)
        .astype(int)
    )

    # Reorder varieties
    order = [v for v in VARIETY_ORDER if v in pivot.index]
    remaining = [v for v in pivot.index if v not in order]
    pivot = pivot.loc[order + remaining]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot as a heatmap using imshow
    data = pivot.values
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    # Ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in pivot.columns],
        rotation=45, ha="right", fontsize=10,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [v.replace("_", " ").title() for v in pivot.index],
        fontsize=10,
    )

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            text = f"{int(val):,}" if val > 0 else "-"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_title("Items per Language Variety and Task",
                 fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Number of Items")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "variety_task_heatmap.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(FIG_DIR / "variety_task_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved variety_task_heatmap.pdf/png")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Label balance comparison across varieties (IroSvA + TASS)
# ──────────────────────────────────────────────────────────────────────
def plot_label_balance_by_variety(df):
    """Stacked bar charts comparing label proportions across varieties
    for tasks that span multiple varieties (IroSvA, TASS-2020)."""
    multi_variety_tasks = (
        df.groupby("task")["language_variety"]
        .nunique()
        .loc[lambda x: x > 1]
        .index.tolist()
    )

    if not multi_variety_tasks:
        print("  No multi-variety tasks found. Skipping label balance plot.")
        return

    n_tasks = len(multi_variety_tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, sorted(multi_variety_tasks)):
        task_df = df[df["task"] == task]

        # Only include IberBench source datasets for this comparison
        # (they share the same label schema within each task)
        task_df = task_df[task_df["source_dataset"].str.startswith("iberbench_")]

        varieties = [v for v in VARIETY_ORDER
                     if v in task_df["language_variety"].unique()]
        labels_sorted = sorted(task_df["label"].unique())

        # Build proportions
        proportions = {}
        for variety in varieties:
            v_df = task_df[task_df["language_variety"] == variety]
            total = len(v_df)
            proportions[variety] = {
                lbl: len(v_df[v_df["label"] == lbl]) / total
                for lbl in labels_sorted
            }

        # Stacked bars
        x = np.arange(len(varieties))
        width = 0.6
        bottom = np.zeros(len(varieties))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels_sorted)))

        for lbl_idx, lbl in enumerate(labels_sorted):
            vals = [proportions[v].get(lbl, 0) for v in varieties]
            ax.bar(x, vals, width, bottom=bottom, label=f"Label {lbl}",
                   color=colors[lbl_idx], edgecolor="white", linewidth=0.5)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [v.replace("_", " ").title() for v in varieties],
            rotation=45, ha="right",
        )
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{task.replace('_', ' ').title()}\nLabel Balance by Variety",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Label Balance Across Language Varieties",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_balance_by_variety.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(FIG_DIR / "label_balance_by_variety.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved label_balance_by_variety.pdf/png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("IberBench Visualization")
    print("=" * 40 + "\n")

    df = load_data()
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_label_distributions(df)
    plot_dataset_sizes(df)
    plot_variety_task_heatmap(df)
    plot_label_balance_by_variety(df)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
