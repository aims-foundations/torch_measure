"""
Visualize the WebArena response matrix.

Produces:
1. Full heatmap (812 tasks x 14 models, sorted)
2. Task difficulty distribution
3. Model accuracy bar chart
4. Model-model correlation (clustered)
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
    matrix = pd.read_csv(f"{DATA_DIR}/webarena_response_matrix.csv", index_col=0)
    summary = pd.read_csv(f"{DATA_DIR}/webarena_response_matrix_summary.csv")
    return matrix, summary


def plot_full_heatmap(matrix):
    df = matrix.select_dtypes(include=[np.number]).copy()
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    fig, ax = plt.subplots(figsize=(14, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Tasks ({n_t} items)")
    ax.set_xlabel(f"Models ({n_m})")
    ax.set_title(f"WebArena Response Matrix ({n_t} tasks x {n_m} models)",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_task_difficulty(matrix):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    item_rates = matrix[num_cols].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(item_rates, bins=bins, alpha=0.8,
            color=sns.color_palette("Set2")[0], edgecolor="white")
    median = np.median(item_rates)
    ax.axvline(median, color="red", ls="--", alpha=0.7,
               label=f"median={median:.2f}")
    unsolved = (item_rates == 0).sum()
    trivial = (item_rates == 1).sum()
    ax.set_xlabel("Task Pass Rate (across models)")
    ax.set_ylabel("Count")
    ax.set_title(f"WebArena — Task Difficulty ({len(item_rates)} tasks)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    txt = []
    if unsolved > 0:
        txt.append(f"{unsolved} unsolved")
    if trivial > 0:
        txt.append(f"{trivial} trivial")
    if txt:
        ax.text(0.98, 0.95, "\n".join(txt), transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(summary):
    df = summary.sort_values("success_rate_pct", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), df["success_rate_pct"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["model"].values, fontsize=8)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title(f"WebArena — Model Success Rates ({n} models)",
                 fontsize=14, fontweight="bold")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["success_rate_pct"] + 0.3, i,
                f"{r['success_rate_pct']:.1f}%", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_model_correlation(matrix):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    corr = matrix[num_cols].corr()
    if len(corr) > 2:
        link = linkage(corr.fillna(0).values, method="ward")
        order = leaves_list(link)
        corr = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.1, vmax=1,
                mask=mask, square=True, linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.6})
    ax.set_title("WebArena — Model-Model Correlation (clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_correlation.pdf/png")


def main():
    matrix, summary = load_data()
    n_m = len(matrix.select_dtypes(include=[np.number]).columns)
    print(f"Loaded: {matrix.shape[0]} tasks x {n_m} models")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(matrix)
    plot_task_difficulty(matrix)
    plot_model_accuracy(summary)
    plot_model_correlation(matrix)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
