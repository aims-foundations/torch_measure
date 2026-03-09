"""
Visualize the EditBench response matrices.

Produces:
1. Full heatmap (540 tasks x 44 models, sorted)
2. Difficulty-level heatmap (models x 2 difficulties)
3. Task difficulty distribution
4. Model accuracy bar chart
5. Model-model correlation (top 30, clustered)
6. Score vs binary comparison scatter
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
    score = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    binary = pd.read_csv(f"{DATA_DIR}/response_matrix_binary.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return score, binary, meta


def plot_full_heatmap(binary):
    df = binary.select_dtypes(include=[np.number]).copy()
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    fig, ax = plt.subplots(figsize=(20, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.5},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Tasks ({n_t} items)")
    ax.set_xlabel(f"Models ({n_m})")
    ax.set_title(f"EditBench Response Matrix ({n_t} tasks x {n_m} models)",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_difficulty_heatmap(binary, meta):
    """Models x difficulty level heatmap."""
    num_cols = binary.select_dtypes(include=[np.number]).columns
    meta_map = dict(zip(meta["task_id"].astype(str), meta["difficulty"]))
    difficulties = [meta_map.get(str(tid), "unknown") for tid in binary.index]

    diff_matrix = pd.DataFrame(index=num_cols, dtype=float)
    for d in sorted(set(difficulties)):
        mask = [dd == d for dd in difficulties]
        diff_matrix[d] = binary.loc[mask, num_cols].mean(axis=0)

    diff_matrix = diff_matrix.sort_values(
        diff_matrix.columns[0], ascending=False, na_position="last"
    )

    fig, ax = plt.subplots(figsize=(8, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(diff_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                cbar_kws={"label": "Pass Rate", "shrink": 0.5})
    ax.set_title("EditBench — Per-Difficulty Pass Rate",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_difficulty.pdf/png")


def plot_task_difficulty(binary, meta):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    item_rates = binary[num_cols].mean(axis=1)
    meta_map = dict(zip(meta["task_id"].astype(str), meta["difficulty"]))
    difficulties = [meta_map.get(str(tid), "unknown") for tid in item_rates.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"easy": "#2ecc71", "hard": "#e74c3c", "unknown": "#95a5a6"}
    for d in sorted(set(difficulties)):
        vals = [item_rates.iloc[i] for i, dd in enumerate(difficulties) if dd == d]
        ax.hist(vals, bins=np.linspace(0, 1, 31), alpha=0.7,
                label=f"{d} (n={len(vals)})", color=colors.get(d, "#95a5a6"),
                edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Task Pass Rate")
    ax.set_ylabel("Count")
    ax.set_title("EditBench — Task Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(binary):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    model_acc = binary[num_cols].mean(axis=0).sort_values(ascending=True)
    n = len(model_acc)
    fig, ax = plt.subplots(figsize=(10, max(10, n * 0.3)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), model_acc.values * 100, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_acc.index, fontsize=7)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title(f"EditBench — Model Pass Rate ({n} models)",
                 fontsize=14, fontweight="bold")
    for i, (m, a) in enumerate(model_acc.items()):
        ax.text(a * 100 + 0.3, i, f"{a*100:.1f}%", va="center", fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_model_correlation(binary):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    mat = binary[num_cols]
    model_acc = mat.mean(axis=0).sort_values(ascending=False)
    top = model_acc.head(30).index
    corr = mat[top].corr()
    link = linkage(corr.fillna(0).values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(12, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.1, vmax=1,
                mask=mask, square=True, linewidths=0.3, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 5},
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.6})
    ax.set_title("EditBench — Model-Model Correlation (Top 30, clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_correlation.pdf/png")


def plot_score_vs_binary(score, binary):
    """Scatter: mean score vs binary pass rate per model."""
    num_cols = score.select_dtypes(include=[np.number]).columns
    common = num_cols.intersection(
        binary.select_dtypes(include=[np.number]).columns
    )
    s_mean = score[common].mean(axis=0) * 100
    b_mean = binary[common].mean(axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(s_mean, b_mean, alpha=0.7, s=50,
               color=sns.color_palette("Set2")[0], edgecolor="gray")
    lim = [0, max(s_mean.max(), b_mean.max()) + 5]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Mean Score (%)")
    ax.set_ylabel("Binary Pass Rate (%)")
    ax.set_title("EditBench — Score vs Binary Pass Rate",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    for m in common:
        ax.annotate(m, (s_mean[m], b_mean[m]), fontsize=4, alpha=0.6,
                    textcoords="offset points", xytext=(3, 3))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/score_vs_binary.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/score_vs_binary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved score_vs_binary.pdf/png")


def main():
    print("Visualizing EditBench response matrices...")
    score, binary, meta = load_data()
    n_cols = len(binary.select_dtypes(include=[np.number]).columns)
    print(f"Loaded: {binary.shape[0]} tasks x {n_cols} models")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(binary)
    plot_difficulty_heatmap(binary, meta)
    plot_task_difficulty(binary, meta)
    plot_model_accuracy(binary)
    plot_model_correlation(binary)
    plot_score_vs_binary(score, binary)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
