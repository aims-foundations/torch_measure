"""
Visualize the cultural evaluation data collection.

Produces:
  1. Items per dataset (bar chart)
  2. Language distribution (top 25, horizontal bar)
  3. Country/region distribution (top 25, horizontal bar)
  4. Task type distribution (pie chart)
  5. Language x Dataset heatmap (top 20 languages)
  6. Region x Dataset heatmap (top 20 regions)

Reads from ../processed/task_metadata.csv
Saves figures to ../figures/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(_BENCHMARK_DIR / "processed")
FIG_DIR = str(_BENCHMARK_DIR / "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="white", font_scale=0.9)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
meta_path = os.path.join(DATA_DIR, "task_metadata.csv")
if not os.path.exists(meta_path):
    print(f"ERROR: {meta_path} not found. Run 01_build_response_matrix.py first.")
    sys.exit(1)

df = pd.read_csv(meta_path)
print(f"Loaded {len(df)} items from {meta_path}")
print(f"Columns: {list(df.columns)}")
print(f"Datasets: {df['source_dataset'].unique()}")
print(f"Languages: {df['language'].nunique()}")
print(f"Regions: {df['country_region'].nunique()}")


# ============================================================================
# 1. Items per dataset (bar chart)
# ============================================================================
def plot_items_per_dataset():
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df["source_dataset"].value_counts().sort_values(ascending=True)
    colors = sns.color_palette("Set2", len(counts))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Items")
    ax.set_title("Items per Source Dataset", fontsize=13, fontweight="bold")

    # Add count labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + max(counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10)

    ax.set_xlim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "items_per_dataset.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 2. Language distribution (top 25)
# ============================================================================
def plot_language_distribution():
    fig, ax = plt.subplots(figsize=(10, 8))
    lang_counts = df["language"].value_counts().head(25).sort_values(ascending=True)
    colors = sns.color_palette("viridis", len(lang_counts))
    bars = ax.barh(lang_counts.index, lang_counts.values, color=colors)
    ax.set_xlabel("Number of Items")
    ax.set_title("Top 25 Languages", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, lang_counts.values):
        ax.text(bar.get_width() + max(lang_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_xlim(0, max(lang_counts.values) * 1.15)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "language_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 3. Country/region distribution (top 25)
# ============================================================================
def plot_region_distribution():
    fig, ax = plt.subplots(figsize=(10, 8))
    # Filter out empty regions
    region_series = df["country_region"].replace("", np.nan).dropna()
    region_counts = region_series.value_counts().head(25).sort_values(ascending=True)

    if region_counts.empty:
        print("  SKIP: No region data available for plotting.")
        return

    colors = sns.color_palette("magma", len(region_counts))
    bars = ax.barh(region_counts.index, region_counts.values, color=colors)
    ax.set_xlabel("Number of Items")
    ax.set_title("Top 25 Countries / Regions", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, region_counts.values):
        ax.text(bar.get_width() + max(region_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_xlim(0, max(region_counts.values) * 1.15)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "region_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 4. Task type distribution (pie)
# ============================================================================
def plot_task_type_distribution():
    fig, ax = plt.subplots(figsize=(8, 6))
    tt_counts = df["task_type"].value_counts()
    colors = sns.color_palette("pastel", len(tt_counts))
    wedges, texts, autotexts = ax.pie(
        tt_counts.values,
        labels=tt_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax.set_title("Task Type Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "task_type_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 5. Language x Dataset heatmap
# ============================================================================
def plot_language_dataset_heatmap():
    top_langs = df["language"].value_counts().head(20).index.tolist()
    subset = df[df["language"].isin(top_langs)]
    cross = pd.crosstab(subset["language"], subset["source_dataset"])

    # Reorder by total count
    cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        cross,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Item Count"},
    )
    ax.set_title("Items by Language x Dataset (Top 20 Languages)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Language")
    ax.set_xlabel("Source Dataset")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "language_dataset_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 6. Region x Dataset heatmap
# ============================================================================
def plot_region_dataset_heatmap():
    region_series = df["country_region"].replace("", np.nan).dropna()
    if region_series.empty:
        print("  SKIP: No region data for heatmap.")
        return

    df_clean = df[df["country_region"].replace("", np.nan).notna()]
    top_regions = df_clean["country_region"].value_counts().head(20).index.tolist()
    subset = df_clean[df_clean["country_region"].isin(top_regions)]
    cross = pd.crosstab(subset["country_region"], subset["source_dataset"])
    cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        cross,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Item Count"},
    )
    ax.set_title("Items by Region x Dataset (Top 20 Regions)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Country / Region")
    ax.set_xlabel("Source Dataset")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "region_dataset_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 7. Split distribution per dataset (stacked bar)
# ============================================================================
def plot_split_distribution():
    cross = pd.crosstab(df["source_dataset"], df["split"])
    fig, ax = plt.subplots(figsize=(10, 5))
    cross.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
    ax.set_title("Split Distribution per Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Source Dataset")
    ax.set_ylabel("Number of Items")
    ax.legend(title="Split", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "split_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("Cultural Evaluation Data - Visualization")
    print("=" * 70)

    plot_items_per_dataset()
    plot_language_distribution()
    plot_region_distribution()
    plot_task_type_distribution()
    plot_language_dataset_heatmap()
    plot_region_dataset_heatmap()
    plot_split_distribution()

    print("\n" + "=" * 70)
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
