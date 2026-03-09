"""
Visualize the CyBench response matrices.

Produces:
1. Full heatmap (40 tasks x 8 models, unguided)
2. Guided vs unguided comparison heatmap
3. Task difficulty by CTF category
4. Model accuracy bar chart (unguided + guided)
5. Category breakdown box plot
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
    unguided = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    guided = pd.read_csv(f"{DATA_DIR}/response_matrix_subtask_guided.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return unguided, guided, meta


def plot_full_heatmap(unguided, guided):
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    for ax, (mat, title) in zip(axes, [
        (unguided, "Unguided"), (guided, "Subtask-Guided")
    ]):
        df = mat.select_dtypes(include=[np.number]).copy()
        n_t, n_m = df.shape
        row_diff = df.mean(axis=1).sort_values(ascending=False)
        df = df.loc[row_diff.index]
        col_acc = df.mean(axis=0).sort_values(ascending=False)
        df = df[col_acc.index]
        sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                    cbar_kws={"label": "Solved", "shrink": 0.4},
                    xticklabels=True, yticklabels=True)
        ax.set_title(f"{title} ({n_t} tasks x {n_m} models)",
                     fontsize=12, fontweight="bold")
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=6)
    fig.suptitle("CyBench Response Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_task_difficulty(unguided, meta):
    num_cols = unguided.select_dtypes(include=[np.number]).columns
    item_rates = unguided[num_cols].mean(axis=1)
    cat_map = dict(zip(meta["task_name"], meta["category"]))
    categories = [cat_map.get(t, "unknown") for t in item_rates.index]

    df = pd.DataFrame({"task": item_rates.index, "category": categories,
                        "solve_rate": item_rates.values})
    cat_order = df.groupby("category")["solve_rate"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="category", y="solve_rate", order=cat_order,
                ax=ax, palette="Set2", showfliers=False)
    sns.stripplot(data=df, x="category", y="solve_rate", order=cat_order,
                  ax=ax, color="black", size=3, alpha=0.4, jitter=True)
    ax.set_ylabel("Solve Rate (across models)")
    ax.set_xlabel("CTF Category")
    ax.set_title("CyBench — Task Difficulty by Category",
                 fontsize=14, fontweight="bold")
    for i, cat in enumerate(cat_order):
        n = len(df[df["category"] == cat])
        ax.text(i, -0.05, f"n={n}", ha="center", va="top", fontsize=8, color="gray")
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(unguided, guided):
    u_num = unguided.select_dtypes(include=[np.number]).columns
    g_num = guided.select_dtypes(include=[np.number]).columns
    u_acc = unguided[u_num].mean(axis=0) * 100
    g_acc = guided[g_num].mean(axis=0) * 100
    common = u_acc.index.intersection(g_acc.index)
    df = pd.DataFrame({"Unguided": u_acc[common], "Guided": g_acc[common]})
    df = df.sort_values("Unguided", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))
    x = np.arange(len(df))
    ax.barh(x - 0.15, df["Unguided"], 0.3, label="Unguided", color="#3498db")
    ax.barh(x + 0.15, df["Guided"], 0.3, label="Guided", color="#e74c3c")
    ax.set_yticks(x)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("Solve Rate (%)")
    ax.set_title("CyBench — Model Accuracy (Unguided vs Guided)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def main():
    unguided, guided, meta = load_data()
    print(f"Loaded unguided: {unguided.shape}, guided: {guided.shape}")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(unguided, guided)
    plot_task_difficulty(unguided, meta)
    plot_model_accuracy(unguided, guided)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
