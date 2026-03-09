"""
Visualize the AgentDojo response matrices.

Produces:
1. Full heatmap (main: 133 tasks x 30 models, sorted)
2. Security heatmap (950 task-injection pairs x 29 models)
3. Utility under attack heatmap (98 tasks x 29 models)
4. Task difficulty distribution
5. Model accuracy bar chart (main + security + utility)
6. Model-model correlation (main matrix, clustered)
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

MATRICES = {
    "Utility (no attack)": "response_matrix.csv",
    "Security (under attack)": "response_matrix_security.csv",
    "Utility (under attack)": "response_matrix_utility_under_attack.csv",
}


def load_matrix(fname):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, index_col=0)
    # Keep only numeric columns
    return df.select_dtypes(include=[np.number])


def plot_full_heatmap():
    """Heatmaps for all three matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(28, 10),
                             gridspec_kw={"width_ratios": [1.2, 3, 1]})
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    for ax, (label, fname) in zip(axes, MATRICES.items()):
        mat = load_matrix(fname)
        n_t, n_m = mat.shape
        row_diff = mat.mean(axis=1).sort_values(ascending=False)
        mat = mat.loc[row_diff.index]
        col_acc = mat.mean(axis=0).sort_values(ascending=False)
        mat = mat[col_acc.index]

        sns.heatmap(mat.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                    cbar_kws={"label": "Score", "shrink": 0.4},
                    xticklabels=True, yticklabels=False)
        ax.set_title(f"{label}\n({n_t} tasks x {n_m} models)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Models")
        ax.set_ylabel("Tasks")
        plt.sca(ax)
        plt.xticks(rotation=90, fontsize=5)

    fig.suptitle("AgentDojo Response Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_task_difficulty():
    """Per-task solve rate distributions for all three matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = sns.color_palette("Set2")
    for ax, (label, fname), color in zip(axes, MATRICES.items(), colors):
        mat = load_matrix(fname)
        rates = mat.mean(axis=1)
        bins = np.linspace(0, 1, 26)
        ax.hist(rates, bins=bins, alpha=0.8, color=color, edgecolor="white")
        median = np.median(rates)
        ax.axvline(median, color="red", ls="--", alpha=0.7,
                   label=f"median={median:.2f}")
        ax.set_xlabel("Task Pass Rate")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({len(rates)} tasks)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("AgentDojo — Task Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy():
    """Bar chart comparing models across all three evaluation modes."""
    accs = {}
    for label, fname in MATRICES.items():
        mat = load_matrix(fname)
        accs[label] = mat.mean(axis=0) * 100
    combined = pd.DataFrame(accs).dropna()
    combined = combined.sort_values("Utility (no attack)", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(combined) * 0.35)))
    x = np.arange(len(combined))
    w = 0.25
    for i, col in enumerate(combined.columns):
        ax.barh(x + i * w, combined[col], w, label=col, alpha=0.85)
    ax.set_yticks(x + w)
    ax.set_yticklabels(combined.index, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("AgentDojo — Model Accuracy Across Evaluation Modes",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_model_correlation():
    """Model-model correlation on main matrix (clustered)."""
    mat = load_matrix(MATRICES["Utility (no attack)"])
    corr = mat.corr()
    if len(corr) > 2:
        link = linkage(corr.fillna(0).values, method="ward")
        order = leaves_list(link)
        corr = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(12, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.2, vmax=1,
                mask=mask, square=True, linewidths=0.3, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 5},
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.6})
    ax.set_title("AgentDojo — Model-Model Correlation (Utility, clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_correlation.pdf/png")


def main():
    print("Visualizing AgentDojo response matrices...")
    print(f"Output directory: {FIG_DIR}\n")
    plot_full_heatmap()
    plot_task_difficulty()
    plot_model_accuracy()
    plot_model_correlation()
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
