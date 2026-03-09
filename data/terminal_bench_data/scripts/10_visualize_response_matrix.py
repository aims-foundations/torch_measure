"""
Visualize the Terminal-Bench response matrix.

Produces:
1. Full heatmap of resolution rates (agent-model × task)
2. Task difficulty bar chart
3. Model accuracy bar chart
4. Correlation / clustering analysis
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(_BENCHMARK_DIR / "processed")
FIG_DIR = str(_BENCHMARK_DIR / "figures")

import os
from pathlib import Path
os.makedirs(FIG_DIR, exist_ok=True)

# Consistent style
sns.set_theme(style="white", font_scale=0.9)


def load_data():
    rate = pd.read_csv(f"{DATA_DIR}/resolution_rate_matrix.csv", index_col=0)
    summary = pd.read_csv(f"{DATA_DIR}/agent_model_summary.csv")
    meta = pd.read_csv(f"{DATA_DIR}/tasks_complete_metadata.csv")
    return rate, summary, meta


def plot_full_heatmap(rate, summary):
    """Large heatmap: agent-model combos (rows) × tasks (columns)."""
    # Keep only full-coverage combos
    full = rate.dropna(thresh=89).copy()

    # Sort rows by overall accuracy (descending)
    row_acc = full.mean(axis=1).sort_values(ascending=False)
    full = full.loc[row_acc.index]

    # Sort columns by task difficulty (hardest left)
    col_diff = full.mean(axis=0).sort_values()
    full = full[col_diff.index]

    fig, ax = plt.subplots(figsize=(28, 20))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        full, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.1, linecolor="white",
        cbar_kws={"label": "Resolution Rate", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Task (sorted by difficulty, hardest left)")
    ax.set_ylabel("Agent | Model (sorted by accuracy, best top)")
    ax.set_title("Terminal-Bench 2.0 Response Matrix", fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_clustered_heatmap(rate):
    """Hierarchically clustered heatmap (both axes)."""
    full = rate.dropna(thresh=89).copy().fillna(0)

    # Cluster rows and columns
    row_link = linkage(full.values, method="ward", metric="euclidean")
    col_link = linkage(full.values.T, method="ward", metric="euclidean")
    row_order = leaves_list(row_link)
    col_order = leaves_list(col_link)

    clustered = full.iloc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(28, 20))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        clustered, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.1, linecolor="white",
        cbar_kws={"label": "Resolution Rate", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Task (clustered)")
    ax.set_ylabel("Agent | Model (clustered)")
    ax.set_title("Terminal-Bench 2.0 — Hierarchically Clustered Response Matrix",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_clustered.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_clustered.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_clustered.pdf/png")


def plot_task_difficulty(rate, meta):
    """Horizontal bar chart of per-task resolution rate."""
    full = rate.dropna(thresh=89)
    task_rates = full.mean(axis=0).sort_values()

    # Merge with category for coloring
    cat_map = dict(zip(meta["task_name"], meta["category"]))
    categories = [cat_map.get(t, "unknown") for t in task_rates.index]
    unique_cats = sorted(set(categories))
    cat_colors = dict(zip(unique_cats, sns.color_palette("tab20", len(unique_cats))))
    bar_colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 22))
    bars = ax.barh(range(len(task_rates)), task_rates.values * 100, color=bar_colors)
    ax.set_yticks(range(len(task_rates)))
    ax.set_yticklabels(task_rates.index, fontsize=7)
    ax.set_xlabel("Average Resolution Rate (%)")
    ax.set_title("Task Difficulty (avg across all agent-model combos)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.invert_yaxis()

    # Legend for categories
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=cat_colors[c], label=c) for c in unique_cats]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              title="Category", title_fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(summary):
    """Bar chart of best agent per model."""
    full = summary[summary["n_tasks"] == 89].copy()
    best = full.loc[full.groupby("model_name")["avg_resolution_rate"].idxmax()]
    best = best.sort_values("avg_resolution_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette("viridis", len(best))
    bars = ax.barh(range(len(best)), best["avg_resolution_rate"].values * 100,
                   color=colors)
    ax.set_yticks(range(len(best)))
    labels = [f"{row['model_name']} ({row['agent_name']})"
              for _, row in best.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Resolution Rate (%)")
    ax.set_title("Best Agent per Model on Terminal-Bench 2.0",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 85)

    # Add value labels
    for i, (_, row) in enumerate(best.iterrows()):
        ax.text(row["avg_resolution_rate"] * 100 + 0.5, i,
                f"{row['avg_resolution_rate']*100:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_task_model_correlation(rate):
    """Task-task correlation matrix showing which tasks co-vary."""
    full = rate.dropna(thresh=89).fillna(0)

    # Task-task correlation
    task_corr = full.corr()

    # Cluster for visualization
    link = linkage(task_corr.values, method="ward")
    order = leaves_list(link)
    task_corr = task_corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(task_corr, dtype=bool), k=1)
    sns.heatmap(
        task_corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.5, vmax=1,
        mask=mask,
        square=True,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("Task-Task Correlation (clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_correlation.pdf/png")


def plot_category_breakdown(rate, meta):
    """Box plot of resolution rates grouped by task category."""
    full = rate.dropna(thresh=89)
    task_rates = full.mean(axis=0).reset_index()
    task_rates.columns = ["task_name", "avg_rate"]

    cat_map = dict(zip(meta["task_name"], meta["category"]))
    task_rates["category"] = task_rates["task_name"].map(cat_map)
    task_rates = task_rates.dropna(subset=["category"])

    # Order categories by median resolution rate
    cat_order = (task_rates.groupby("category")["avg_rate"]
                 .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=task_rates, x="category", y="avg_rate", order=cat_order,
                ax=ax, palette="Set2", showfliers=True)
    sns.stripplot(data=task_rates, x="category", y="avg_rate", order=cat_order,
                  ax=ax, color="black", size=4, alpha=0.5, jitter=True)
    ax.set_ylabel("Average Resolution Rate")
    ax.set_xlabel("")
    ax.set_title("Resolution Rate by Task Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_breakdown.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved category_breakdown.pdf/png")


def main():
    rate, summary, meta = load_data()
    print(f"Loaded: {rate.shape[0]} combos × {rate.shape[1]} tasks")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_full_heatmap(rate, summary)
    plot_clustered_heatmap(rate)
    plot_task_difficulty(rate, meta)
    plot_model_accuracy(summary)
    plot_task_model_correlation(rate)
    plot_category_breakdown(rate, meta)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
