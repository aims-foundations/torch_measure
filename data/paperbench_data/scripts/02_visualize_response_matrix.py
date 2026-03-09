"""
Visualize the PaperBench response matrices.

Produces:
1. Full heatmap (20 papers x 9 models)
2. Model accuracy bar chart (overall scores)
3. Category breakdown (code_dev, execution, result_analysis)
4. Paper difficulty distribution
5. BasicAgent vs IterativeAgent comparison
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(_BENCHMARK_DIR / "processed")
FIG_DIR = str(_BENCHMARK_DIR / "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="white", font_scale=0.9)


def load_data():
    matrix = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    overall = pd.read_csv(f"{DATA_DIR}/overall_scores.csv")
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return matrix, overall, meta


def plot_full_heatmap(matrix):
    """Heatmap of paper x model scores."""
    df = matrix.copy()
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df, ax=ax, cmap=cmap, vmin=0, vmax=0.5,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                cbar_kws={"label": "Score", "shrink": 0.6})
    ax.set_ylabel(f"Papers ({n_t})")
    ax.set_xlabel(f"Models ({n_m})")
    ax.set_title(f"PaperBench Response Matrix ({n_t} papers x {n_m} models)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_model_accuracy(overall):
    """Bar chart of overall model performance."""
    df = overall.sort_values("overall_pct", ascending=True).copy()
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), df["overall_pct"].values, color=colors,
            xerr=df["overall_se"].values, capsize=3, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["model"].values, fontsize=8)
    ax.set_xlabel("Overall Score (%)")
    ax.set_title(f"PaperBench — Model Performance ({n} models)",
                 fontsize=14, fontweight="bold")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["overall_pct"] + r["overall_se"] + 0.3, i,
                f"{r['overall_pct']:.1f}%", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_category_breakdown(overall):
    """Grouped bar: code_dev vs execution vs result_analysis per model."""
    categories = ["code_dev", "execution", "result_analysis"]
    cat_labels = ["Code Development", "Execution", "Result Analysis"]
    palette = ["#3498db", "#2ecc71", "#e74c3c"]

    df = overall.sort_values("overall_pct", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.5)))
    x = np.arange(n)
    width = 0.25

    for j, (cat, label, color) in enumerate(zip(categories, cat_labels,
                                                 palette)):
        col = f"{cat}_pct"
        vals = df[col].values
        ax.barh(x + (j - 1) * width, vals, width, label=label,
                color=color, alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(df["model"].values, fontsize=8)
    ax.set_xlabel("Score (%)")
    ax.set_title("PaperBench — Performance by Category",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/category_breakdown.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/category_breakdown.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved category_breakdown.pdf/png")


def plot_paper_difficulty(matrix):
    """Distribution of paper difficulty (mean score across models)."""
    paper_means = matrix.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, paper_means.max() + 0.02, 15)
    ax.hist(paper_means, bins=bins, alpha=0.8, edgecolor="white",
            color=sns.color_palette("Set2")[0])
    ax.axvline(paper_means.median(), color="red", ls="--", alpha=0.7,
               label=f"Median = {paper_means.median():.3f}")
    ax.set_xlabel("Mean Score (across models)")
    ax.set_ylabel("Count")
    ax.set_title("PaperBench — Paper Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_basic_vs_iterative(overall):
    """Scatter: BasicAgent vs IterativeAgent for models with both."""
    basic = overall[overall["model"].str.contains("BasicAgent")]
    iterative = overall[overall["model"].str.contains("IterativeAgent")]

    basic_map = {}
    for _, r in basic.iterrows():
        key = r["model"].replace(" (BasicAgent)", "").strip()
        basic_map[key] = r["overall_pct"]

    iter_map = {}
    for _, r in iterative.iterrows():
        key = r["model"].replace(" (IterativeAgent)", "").strip()
        iter_map[key] = r["overall_pct"]

    common = set(basic_map) & set(iter_map)
    if not common:
        print("No common models for Basic vs Iterative comparison, skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for m in sorted(common):
        ax.scatter(basic_map[m], iter_map[m], s=80,
                   color=sns.color_palette("Set2")[0], edgecolor="gray",
                   alpha=0.8, zorder=5)
        ax.annotate(m, (basic_map[m], iter_map[m]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4))
    lim = [0, max(max(basic_map.values()), max(iter_map.values())) + 3]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("BasicAgent — Overall (%)")
    ax.set_ylabel("IterativeAgent — Overall (%)")
    ax.set_title("PaperBench — BasicAgent vs IterativeAgent",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/basic_vs_iterative.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/basic_vs_iterative.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved basic_vs_iterative.pdf/png")


def main():
    matrix, overall, meta = load_data()
    print(f"Loaded: {matrix.shape[0]} papers x {matrix.shape[1]} models")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(matrix)
    plot_model_accuracy(overall)
    plot_category_breakdown(overall)
    plot_paper_difficulty(matrix)
    plot_basic_vs_iterative(overall)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
