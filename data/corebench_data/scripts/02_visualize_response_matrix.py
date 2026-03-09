"""
Visualize the CoreBench response matrices.

Produces:
1. Full heatmap (270 tasks x 16 agents, binary pass@k)
2. Difficulty-level heatmap (agents x 3 difficulty levels)
3. Task difficulty distribution by field
4. Model accuracy bar chart
5. Binary vs score comparison
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
    binary = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    scores = pd.read_csv(f"{DATA_DIR}/response_matrix_scores.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return binary, scores, meta


def plot_full_heatmap(binary):
    df = binary.select_dtypes(include=[np.number]).copy()
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Tasks ({n_t})")
    ax.set_xlabel(f"Agents ({n_m})")
    ax.set_title(f"CoreBench Response Matrix ({n_t} tasks x {n_m} agents)",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_difficulty_heatmap(binary, meta):
    """Agents x difficulty level heatmap."""
    num_cols = binary.select_dtypes(include=[np.number]).columns
    diff_map = dict(zip(meta["task_id"].astype(str), meta["difficulty"]))
    difficulties = sorted({diff_map.get(str(t), "unknown")
                          for t in binary.index})

    diff_matrix = pd.DataFrame(index=num_cols, columns=difficulties, dtype=float)
    for d in difficulties:
        mask = [diff_map.get(str(t)) == d for t in binary.index]
        if any(mask):
            diff_matrix[d] = binary.loc[mask, num_cols].mean(axis=0)

    row_acc = diff_matrix.mean(axis=1).sort_values(ascending=False)
    diff_matrix = diff_matrix.loc[row_acc.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(diff_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Pass Rate", "shrink": 0.6})
    ax.set_title("CoreBench — Pass Rate by Difficulty",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_difficulty.pdf/png")


def plot_task_difficulty(scores, meta):
    num_cols = scores.select_dtypes(include=[np.number]).columns
    item_rates = scores[num_cols].mean(axis=1)
    field_map = dict(zip(meta["task_id"].astype(str), meta["field"]))
    fields = [field_map.get(str(t), "unknown") for t in item_rates.index]

    cat_rates = {}
    for idx, f in zip(item_rates.index, fields):
        cat_rates.setdefault(f, []).append(item_rates[idx])

    cat_order = sorted(cat_rates, key=lambda c: np.median(cat_rates[c]))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab20", len(cat_order))
    bins = np.linspace(0, 1, 26)
    for cat, color in zip(reversed(cat_order), reversed(colors)):
        ax.hist(cat_rates[cat], bins=bins, alpha=0.7, label=cat,
                color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Task Score (across agents)")
    ax.set_ylabel("Count")
    ax.set_title("CoreBench — Task Difficulty by Field",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(binary):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    model_acc = binary[num_cols].mean(axis=0).sort_values(ascending=True) * 100
    n = len(model_acc)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), model_acc.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_acc.index, fontsize=6)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title(f"CoreBench — Agent Pass Rate ({n} agents)",
                 fontsize=14, fontweight="bold")
    for i, (m, v) in enumerate(model_acc.items()):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_binary_vs_score(binary, scores):
    """Scatter: binary pass rate vs mean accuracy score per agent."""
    num_cols = binary.select_dtypes(include=[np.number]).columns
    common = num_cols.intersection(
        scores.select_dtypes(include=[np.number]).columns
    )
    b_mean = binary[common].mean(axis=0) * 100
    s_mean = scores[common].mean(axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(b_mean, s_mean, alpha=0.7, s=80,
               color=sns.color_palette("Set2")[0], edgecolor="gray")
    lim = [0, max(b_mean.max(), s_mean.max()) + 5]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    for m in common:
        ax.annotate(m, (b_mean[m], s_mean[m]), fontsize=5, alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Binary Pass Rate (%)")
    ax.set_ylabel("Mean Accuracy Score (%)")
    ax.set_title("CoreBench — Binary Pass vs Accuracy Score",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/binary_vs_score.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/binary_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved binary_vs_score.pdf/png")


def main():
    binary, scores, meta = load_data()
    n_m = len(binary.select_dtypes(include=[np.number]).columns)
    print(f"Loaded: {binary.shape[0]} tasks x {n_m} agents")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(binary)
    plot_difficulty_heatmap(binary, meta)
    plot_task_difficulty(scores, meta)
    plot_model_accuracy(binary)
    plot_binary_vs_score(binary, scores)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
