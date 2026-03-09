"""
Visualize the VisualWebArena response matrix.

Produces:
1. Full heatmap (tasks x models, sorted)
2. Domain-level heatmap (models x domains)
3. Task difficulty distribution by domain
4. Model accuracy bar chart
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
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return matrix, meta


def plot_full_heatmap(matrix):
    df = matrix.select_dtypes(include=[np.number]).copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    fig, ax = plt.subplots(figsize=(10, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Tasks ({n_t})")
    ax.set_xlabel(f"Models ({n_m})")
    ax.set_title(f"VisualWebArena Response Matrix ({n_t} tasks x {n_m} models)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_domain_heatmap(matrix, meta):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    dom_map = dict(zip(meta["task_id"].astype(str), meta["domain"]))
    domains = sorted({dom_map.get(str(t), "unknown") for t in matrix.index})

    dom_matrix = pd.DataFrame(index=num_cols, columns=domains, dtype=float)
    for dom in domains:
        mask = [dom_map.get(str(t)) == dom for t in matrix.index]
        if any(mask):
            dom_matrix[dom] = matrix.loc[mask, num_cols].mean(axis=0)

    dom_matrix = dom_matrix.dropna(how="all")
    dom_matrix = dom_matrix.sort_values(dom_matrix.columns[0],
                                        ascending=False, na_position="last")

    fig, ax = plt.subplots(figsize=(10, max(4, len(dom_matrix) * 0.6)))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(dom_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Success Rate", "shrink": 0.6})
    ax.set_xlabel("Domain")
    ax.set_ylabel("Model")
    ax.set_title("VisualWebArena — Per-Domain Success Rate",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_domain.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_domain.pdf/png")


def plot_task_difficulty(matrix, meta):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    item_rates = matrix[num_cols].mean(axis=1).dropna()
    dom_map = dict(zip(meta["task_id"].astype(str), meta["domain"]))
    domains = [dom_map.get(str(t), "unknown") for t in item_rates.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    cat_rates = {}
    for idx, dom in zip(item_rates.index, domains):
        cat_rates.setdefault(dom, []).append(item_rates[idx])
    colors = sns.color_palette("tab10", len(cat_rates))
    bins = np.linspace(0, 1, 21)
    for (dom, vals), color in zip(sorted(cat_rates.items()), colors):
        ax.hist(vals, bins=bins, alpha=0.7, label=f"{dom} (n={len(vals)})",
                color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Task Success Rate (across models)")
    ax.set_ylabel("Count")
    ax.set_title("VisualWebArena — Task Difficulty by Domain",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(matrix):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    model_acc = matrix[num_cols].mean(axis=0).dropna().sort_values(ascending=True) * 100
    n = len(model_acc)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.6)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), model_acc.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_acc.index, fontsize=8)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title(f"VisualWebArena — Model Success Rate ({n} models)",
                 fontsize=14, fontweight="bold")
    for i, (m, v) in enumerate(model_acc.items()):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def main():
    matrix, meta = load_data()
    n_m = len(matrix.select_dtypes(include=[np.number]).columns)
    print(f"Loaded: {matrix.shape[0]} tasks x {n_m} models")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(matrix)
    plot_domain_heatmap(matrix, meta)
    plot_task_difficulty(matrix, meta)
    plot_model_accuracy(matrix)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
