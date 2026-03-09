"""
Visualize the MMLU-Pro response matrices.

Produces:
1. Full heatmap (12,257 questions x 48 models, sorted — per-question matrix)
2. Category-level heatmap (247 models x 14 categories)
3. Task difficulty distribution by category
4. Model accuracy bar chart
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
    """Load all MMLU-Pro processed data."""
    response_matrix = pd.read_csv(
        f"{DATA_DIR}/response_matrix.csv", index_col=0
    )
    category_matrix = pd.read_csv(
        f"{DATA_DIR}/response_matrix_category.csv", index_col=0
    )
    question_metadata = pd.read_csv(f"{DATA_DIR}/question_metadata.csv")
    return response_matrix, category_matrix, question_metadata


def plot_full_heatmap(matrix):
    """Full heatmap of questions x models (48 models with per-question data)."""
    df = matrix.copy()

    # Drop columns/rows that are entirely NaN
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    n_q, n_m = df.shape

    # Sort columns by overall accuracy (best left)
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    # Sort rows by difficulty (easiest top)
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]

    fig, ax = plt.subplots(figsize=(16, 20))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Correct (1) / Incorrect (0)", "shrink": 0.4},
        xticklabels=True, yticklabels=False,
    )
    ax.set_ylabel(f"Questions ({n_q} items, sorted by difficulty)")
    ax.set_xlabel(f"Models ({n_m} models, sorted by accuracy)")
    ax.set_title(
        f"MMLU-Pro Response Matrix ({n_q} questions x {n_m} models)",
        fontsize=16, fontweight="bold",
    )
    plt.xticks(rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_category_heatmap(cat_matrix):
    """Category-level heatmap: models x 14 categories."""
    df = cat_matrix.copy().T  # transpose to models x categories

    # Drop rows with all NaN
    df = df.dropna(how="all")

    # Sort rows by mean accuracy, columns by mean difficulty
    row_acc = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_acc.index]
    col_diff = df.mean(axis=0).sort_values()
    df = df[col_diff.index]

    # Limit to top 60 models for readability
    df_top = df.head(60)

    fig, ax = plt.subplots(figsize=(14, 20))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df_top, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.3, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 5},
        cbar_kws={"label": "Accuracy", "shrink": 0.4},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Category (sorted by difficulty, hardest left)")
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_title(
        f"MMLU-Pro — Per-Category Accuracy (top 60 of {len(df)} models x "
        f"{len(df.columns)} categories)",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_category.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_category.pdf/png")

    return df


def plot_task_difficulty(matrix, q_meta):
    """Histogram of per-question pass rates, colored by category."""
    # Merge question metadata
    df = matrix.copy()
    df = df.dropna(axis=1, how="all")
    item_rates = df.mean(axis=1)

    # Map question_id to category
    cat_map = dict(zip(
        q_meta["question_id"].astype(str), q_meta["category"]
    ))
    categories = [cat_map.get(str(qid), "Unknown") for qid in item_rates.index]

    cat_rates = {}
    for qid, cat in zip(item_rates.index, categories):
        cat_rates.setdefault(cat, []).append(item_rates[qid])

    cat_order = sorted(cat_rates.keys(),
                       key=lambda c: np.median(cat_rates[c]))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab20", len(cat_order))
    cat_colors = dict(zip(cat_order, colors))

    bins = np.linspace(0, 1, 41)
    for cat in reversed(cat_order):
        ax.hist(cat_rates[cat], bins=bins, alpha=0.7, label=cat,
                color=cat_colors[cat], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Question Pass Rate (across 48 models)")
    ax.set_ylabel("Count")
    ax.set_title("MMLU-Pro — Question Difficulty Distribution by Category",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(matrix):
    """Horizontal bar chart of model accuracy (per-question models)."""
    df = matrix.copy().dropna(axis=1, how="all")
    model_acc = df.mean(axis=0).sort_values(ascending=True)
    model_acc = model_acc.dropna()
    n = len(model_acc)

    fig, ax = plt.subplots(figsize=(10, max(10, n * 0.3)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), model_acc.values * 100, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_acc.index, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(
        f"MMLU-Pro — Model Accuracy ({n} models, {df.shape[0]} questions)",
        fontsize=14, fontweight="bold",
    )

    for i, (model, acc) in enumerate(model_acc.items()):
        ax.text(acc * 100 + 0.3, i, f"{acc*100:.1f}%", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_category_breakdown(matrix, q_meta):
    """Box plot of per-question pass rates grouped by category."""
    df = matrix.copy().dropna(axis=1, how="all")
    item_rates = df.mean(axis=1)

    cat_map = dict(zip(
        q_meta["question_id"].astype(str), q_meta["category"]
    ))
    categories = [cat_map.get(str(qid), "Unknown") for qid in item_rates.index]

    plot_df = pd.DataFrame({
        "question_id": item_rates.index,
        "category": categories,
        "pass_rate": item_rates.values,
    })

    cat_order = (plot_df.groupby("category")["pass_rate"]
                 .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=plot_df, x="category", y="pass_rate", order=cat_order,
                ax=ax, palette="Set2", showfliers=False)
    sns.stripplot(data=plot_df, x="category", y="pass_rate", order=cat_order,
                  ax=ax, color="black", size=0.8, alpha=0.15, jitter=True)
    ax.set_ylabel("Question Pass Rate")
    ax.set_xlabel("")
    ax.set_title("MMLU-Pro — Question Pass Rate by Category",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    for i, cat in enumerate(cat_order):
        n_items = len(plot_df[plot_df["category"] == cat])
        ax.text(i, -0.03, f"n={n_items}", ha="center", va="top",
                fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_breakdown.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved category_breakdown.pdf/png")


def plot_category_correlation(cat_df):
    """Category-category correlation heatmap."""
    # cat_df is models x categories (from plot_category_heatmap)
    corr = cat_df.dropna(how="all").corr()

    # Cluster
    # Fill NaN for clustering
    corr_filled = corr.fillna(0)
    link = linkage(corr_filled.values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.3, vmax=1,
        mask=mask, square=True,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("MMLU-Pro — Category-Category Correlation (clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved category_correlation.pdf/png")


def main():
    response_matrix, category_matrix, q_meta = load_data()
    n_q = response_matrix.shape[0]
    n_m = response_matrix.dropna(axis=1, how="all").shape[1]
    print(f"Loaded per-question matrix: {n_q} questions x {n_m} models")
    print(f"Loaded category matrix: {category_matrix.shape}")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_full_heatmap(response_matrix)
    cat_df = plot_category_heatmap(category_matrix)
    plot_task_difficulty(response_matrix, q_meta)
    plot_model_accuracy(response_matrix)
    plot_category_breakdown(response_matrix, q_meta)
    plot_category_correlation(cat_df)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
