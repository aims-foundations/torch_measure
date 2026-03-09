"""
Visualize the BFCL response matrix.

Produces:
1. Full heatmap (93 models × 4,751 items, sorted by accuracy/difficulty)
2. Category-level heatmap (93 models × 22 categories, more readable)
3. Task difficulty distribution by category
4. Model accuracy bar chart
5. Category breakdown box plot
6. Task-task correlation at category level
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
    matrix = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    return matrix


def extract_category(task_id):
    """Extract category from global task ID like 'exec_simple::exec_simple_42'."""
    return task_id.split("::")[0]


def plot_full_heatmap(matrix):
    """Full heatmap of 93 models × 4,751 items (no tick labels — too dense)."""
    df = matrix.copy()

    # Sort rows by overall accuracy (best on top)
    row_acc = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_acc.index]

    # Sort columns: first by category, then by difficulty within category
    categories = [extract_category(c) for c in df.columns]
    col_diff = df.mean(axis=0)
    col_order = sorted(
        range(len(df.columns)),
        key=lambda i: (categories[i], col_diff.iloc[i])
    )
    df = df.iloc[:, col_order]
    sorted_cats = [categories[i] for i in col_order]

    fig, ax = plt.subplots(figsize=(24, 12))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.5},
        xticklabels=False, yticklabels=True,
    )
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_xlabel(f"Tasks ({len(df.columns)} items, grouped by category)")
    ax.set_title("BFCL v3 Response Matrix (93 models × 4,751 items)",
                 fontsize=16, fontweight="bold")
    plt.yticks(fontsize=5)

    # Add category separators
    unique_cats = []
    cat_positions = []
    prev_cat = None
    for i, cat in enumerate(sorted_cats):
        if cat != prev_cat:
            if prev_cat is not None:
                ax.axvline(i, color="black", linewidth=0.5, alpha=0.7)
            cat_positions.append(i)
            unique_cats.append(cat)
            prev_cat = cat

    # Add category labels at bottom
    for i, (cat, pos) in enumerate(zip(unique_cats, cat_positions)):
        next_pos = cat_positions[i + 1] if i + 1 < len(cat_positions) else len(sorted_cats)
        mid = (pos + next_pos) / 2
        if (next_pos - pos) > 30:  # only label categories wide enough
            ax.text(mid, len(df) + 1, cat, ha="center", va="top",
                    fontsize=5, rotation=90)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_category_heatmap(matrix):
    """Category-level heatmap: 93 models × 22 categories (mean accuracy)."""
    categories = [extract_category(c) for c in matrix.columns]
    unique_cats = sorted(set(categories))

    # Compute per-model per-category accuracy
    cat_matrix = pd.DataFrame(index=matrix.index, columns=unique_cats, dtype=float)
    for cat in unique_cats:
        cols = [c for c, ct in zip(matrix.columns, categories) if ct == cat]
        cat_matrix[cat] = matrix[cols].mean(axis=1)

    # Sort rows by overall accuracy
    row_acc = cat_matrix.mean(axis=1).sort_values(ascending=False)
    cat_matrix = cat_matrix.loc[row_acc.index]

    # Sort columns by overall difficulty
    col_diff = cat_matrix.mean(axis=0).sort_values()
    cat_matrix = cat_matrix[col_diff.index]

    fig, ax = plt.subplots(figsize=(14, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        cat_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 6},
        cbar_kws={"label": "Accuracy", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Category (sorted by difficulty, hardest left)")
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_title("BFCL v3 — Per-Category Accuracy (93 models × 22 categories)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_category.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_category.pdf/png")

    return cat_matrix


def plot_task_difficulty(matrix):
    """Histogram of per-item pass rates, colored by category."""
    categories = [extract_category(c) for c in matrix.columns]
    item_rates = matrix.mean(axis=0)

    # Group by category
    cat_rates = {}
    for col, cat in zip(matrix.columns, categories):
        cat_rates.setdefault(cat, []).append(item_rates[col])

    # Sort categories by median difficulty
    cat_order = sorted(cat_rates.keys(),
                       key=lambda c: np.median(cat_rates[c]))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab20", len(cat_order))
    cat_colors = dict(zip(cat_order, colors))

    # Stacked histogram
    bins = np.linspace(0, 1, 41)
    for cat in reversed(cat_order):  # reverse so hardest categories draw last
        ax.hist(cat_rates[cat], bins=bins, alpha=0.7, label=cat,
                color=cat_colors[cat], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Item Pass Rate (across 93 models)")
    ax.set_ylabel("Count")
    ax.set_title("BFCL v3 — Item Difficulty Distribution by Category",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(matrix):
    """Horizontal bar chart of model accuracy."""
    model_acc = matrix.mean(axis=1).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 16))
    colors = sns.color_palette("viridis", len(model_acc))
    ax.barh(range(len(model_acc)), model_acc.values * 100, color=colors)
    ax.set_yticks(range(len(model_acc)))
    ax.set_yticklabels(model_acc.index, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("BFCL v3 — Model Accuracy (93 models, 4,751 items)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 85)

    for i, (model, acc) in enumerate(model_acc.items()):
        ax.text(acc * 100 + 0.3, i, f"{acc*100:.1f}%", va="center", fontsize=5)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_category_breakdown(matrix):
    """Box plot of per-item pass rates grouped by category."""
    categories = [extract_category(c) for c in matrix.columns]
    item_rates = matrix.mean(axis=0)

    df = pd.DataFrame({
        "task_id": matrix.columns,
        "category": categories,
        "pass_rate": item_rates.values,
    })

    # Order by median
    cat_order = (df.groupby("category")["pass_rate"]
                 .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x="category", y="pass_rate", order=cat_order,
                ax=ax, palette="Set2", showfliers=False)
    sns.stripplot(data=df, x="category", y="pass_rate", order=cat_order,
                  ax=ax, color="black", size=1.5, alpha=0.3, jitter=True)
    ax.set_ylabel("Item Pass Rate")
    ax.set_xlabel("")
    ax.set_title("BFCL v3 — Item Pass Rate by Category",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    # Add item counts
    for i, cat in enumerate(cat_order):
        n = len(df[df["category"] == cat])
        ax.text(i, -0.03, f"n={n}", ha="center", va="top", fontsize=7,
                color="gray")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_breakdown.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved category_breakdown.pdf/png")


def plot_category_correlation(cat_matrix):
    """Category-category correlation heatmap."""
    corr = cat_matrix.corr()

    # Cluster
    link = linkage(corr.values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.3, vmax=1,
        mask=mask, square=True,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("BFCL v3 — Category-Category Correlation (clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved category_correlation.pdf/png")


def main():
    matrix = load_data()
    print(f"Loaded: {matrix.shape[0]} models × {matrix.shape[1]} items")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_full_heatmap(matrix)
    cat_matrix = plot_category_heatmap(matrix)
    plot_task_difficulty(matrix)
    plot_model_accuracy(matrix)
    plot_category_breakdown(matrix)
    plot_category_correlation(cat_matrix)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
