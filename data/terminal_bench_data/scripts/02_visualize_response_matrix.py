"""
Visualize the Terminal-Bench response matrices.

Produces:
1. Full heatmap (128 agents x ~100 tasks, binary majority)
2. Category-level heatmap (agents x categories)
3. Task difficulty distribution by category
4. Agent accuracy bar chart
5. Category breakdown box plot
6. Category-category correlation (clustered)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(_BENCHMARK_DIR / "processed")
FIG_DIR = str(_BENCHMARK_DIR / "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="white", font_scale=0.9)


def load_data():
    binary = pd.read_csv(f"{DATA_DIR}/binary_majority_matrix.csv", index_col=0)
    resolution = pd.read_csv(f"{DATA_DIR}/resolution_rate_matrix.csv", index_col=0)
    task_meta = pd.read_csv(f"{DATA_DIR}/all_tasks_metadata.csv")
    return binary, resolution, task_meta


def plot_full_heatmap(binary):
    df = binary.select_dtypes(include=[np.number]).copy()
    n_a, n_t = df.shape
    row_acc = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_acc.index]
    col_diff = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_diff.index]

    fig, ax = plt.subplots(figsize=(20, 18))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Solved (1) / Failed (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=True)
    ax.set_ylabel(f"Agents ({n_a}, sorted by accuracy)")
    ax.set_xlabel(f"Tasks ({n_t}, sorted by difficulty)")
    ax.set_title(f"Terminal-Bench Response Matrix ({n_a} agents x {n_t} tasks)",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(fontsize=3.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_category_heatmap(binary, task_meta):
    """Agents x categories heatmap (mean resolution rate per category)."""
    num_cols = binary.select_dtypes(include=[np.number]).columns
    cat_map = dict(zip(task_meta["task_id"], task_meta["category"]))
    categories = {cat_map.get(t, "unknown") for t in num_cols}
    categories = sorted(categories)

    cat_matrix = pd.DataFrame(index=binary.index, columns=categories, dtype=float)
    for cat in categories:
        cols = [c for c in num_cols if cat_map.get(c) == cat]
        if cols:
            cat_matrix[cat] = binary[cols].mean(axis=1)

    row_acc = cat_matrix.mean(axis=1).sort_values(ascending=False)
    cat_matrix = cat_matrix.loc[row_acc.index]
    col_diff = cat_matrix.mean(axis=0).sort_values()
    cat_matrix = cat_matrix[col_diff.index]

    top = cat_matrix.head(40)
    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(top, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.3, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 5},
                cbar_kws={"label": "Solve Rate", "shrink": 0.5})
    ax.set_xlabel("Category")
    ax.set_ylabel("Agent (top 40 by overall accuracy)")
    ax.set_title(f"Terminal-Bench — Per-Category Solve Rate",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_category.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_category.pdf/png")
    return cat_matrix


def plot_task_difficulty(binary, task_meta):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    item_rates = binary[num_cols].mean(axis=0)
    cat_map = dict(zip(task_meta["task_id"], task_meta["category"]))

    cat_rates = {}
    for t in num_cols:
        cat = cat_map.get(t, "unknown")
        cat_rates.setdefault(cat, []).append(item_rates[t])

    cat_order = sorted(cat_rates, key=lambda c: np.median(cat_rates[c]))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab20", len(cat_order))
    bins = np.linspace(0, 1, 26)
    for cat, color in zip(reversed(cat_order), reversed(colors)):
        ax.hist(cat_rates[cat], bins=bins, alpha=0.7, label=cat,
                color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Task Solve Rate (across agents)")
    ax.set_ylabel("Count")
    ax.set_title("Terminal-Bench — Task Difficulty by Category",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(binary):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    agent_acc = binary[num_cols].mean(axis=1).sort_values(ascending=True)
    n = len(agent_acc)
    fig, ax = plt.subplots(figsize=(10, max(14, n * 0.14)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), agent_acc.values * 100, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(agent_acc.index, fontsize=3.5)
    ax.set_xlabel("Solve Rate (%)")
    ax.set_title(f"Terminal-Bench — Agent Accuracy ({n} agents)",
                 fontsize=14, fontweight="bold")
    for i, (a, v) in enumerate(agent_acc.items()):
        ax.text(v * 100 + 0.3, i, f"{v*100:.1f}%", va="center", fontsize=3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_category_correlation(cat_matrix):
    corr = cat_matrix.dropna(how="all").corr()
    link = linkage(corr.fillna(0).values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.3, vmax=1,
                mask=mask, square=True, linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.6})
    ax.set_title("Terminal-Bench — Category-Category Correlation",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved category_correlation.pdf/png")


def main():
    print("Visualizing Terminal-Bench response matrices...")
    binary, resolution, task_meta = load_data()
    print(f"Loaded: {binary.shape[0]} agents x {binary.shape[1]} tasks")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(binary)
    cat_matrix = plot_category_heatmap(binary, task_meta)
    plot_task_difficulty(binary, task_meta)
    plot_model_accuracy(binary)
    plot_category_correlation(cat_matrix)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
