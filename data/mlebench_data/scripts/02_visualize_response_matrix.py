"""
Visualize the MLE-bench response matrices.

Produces:
1. Full heatmap (75 competitions x 30 agents, medal-level scores)
2. Category-level heatmap (agents x task categories)
3. Task difficulty distribution by category
4. Model accuracy bar chart (medal counts)
5. Medal distribution stacked bar
6. Category breakdown box plot
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from matplotlib.colors import ListedColormap
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(_BENCHMARK_DIR / "processed")
FIG_DIR = str(_BENCHMARK_DIR / "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="white", font_scale=0.9)


def load_data():
    medal = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    binary = pd.read_csv(f"{DATA_DIR}/response_matrix_binary.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return medal, binary, meta


def plot_full_heatmap(medal):
    df = medal.select_dtypes(include=[np.number]).copy()
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    medal_cmap = ListedColormap(["#2c3e50", "#cd7f32", "#c0c0c0", "#ffd700"])
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(df.values, ax=ax, cmap=medal_cmap, vmin=0, vmax=3,
                cbar_kws={"label": "Medal (0=none, 1=bronze, 2=silver, 3=gold)",
                          "shrink": 0.4, "ticks": [0, 1, 2, 3]},
                xticklabels=True, yticklabels=True)
    ax.set_ylabel(f"Competitions ({n_t})")
    ax.set_xlabel(f"Agents ({n_m})")
    ax.set_title(f"MLE-bench Medal Matrix ({n_t} competitions x {n_m} agents)",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_category_heatmap(binary, meta):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    cat_map = dict(zip(meta["competition_id"].astype(str), meta["category"]))
    categories = sorted({cat_map.get(str(t), "unknown") for t in binary.index})

    cat_matrix = pd.DataFrame(index=num_cols, columns=categories, dtype=float)
    for cat in categories:
        mask = [cat_map.get(str(t)) == cat for t in binary.index]
        if any(mask):
            cat_matrix[cat] = binary.loc[mask, num_cols].mean(axis=0)

    row_acc = cat_matrix.mean(axis=1).sort_values(ascending=False)
    cat_matrix = cat_matrix.loc[row_acc.index]
    col_diff = cat_matrix.mean(axis=0).sort_values()
    cat_matrix = cat_matrix[col_diff.index]

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(cat_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                cbar_kws={"label": "Medal Rate", "shrink": 0.5})
    ax.set_xlabel("Task Category")
    ax.set_ylabel("Agent")
    ax.set_title("MLE-bench — Per-Category Medal Rate",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_category.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_category.pdf/png")


def plot_task_difficulty(binary, meta):
    num_cols = binary.select_dtypes(include=[np.number]).columns
    item_rates = binary[num_cols].mean(axis=1)
    cat_map = dict(zip(meta["competition_id"].astype(str), meta["category"]))
    categories = [cat_map.get(str(t), "unknown") for t in item_rates.index]

    cat_rates = {}
    for idx, cat in zip(item_rates.index, categories):
        cat_rates.setdefault(cat, []).append(item_rates[idx])

    cat_order = sorted(cat_rates, key=lambda c: np.median(cat_rates[c]))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab20", len(cat_order))
    bins = np.linspace(0, 1, 21)
    for cat, color in zip(reversed(cat_order), reversed(colors)):
        ax.hist(cat_rates[cat], bins=bins, alpha=0.7, label=cat,
                color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Competition Medal Rate (across agents)")
    ax.set_ylabel("Count")
    ax.set_title("MLE-bench — Competition Difficulty by Category",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(medal):
    """Stacked bar chart showing medal distribution per agent."""
    num_cols = medal.select_dtypes(include=[np.number]).columns
    counts = {}
    for col in num_cols:
        vals = medal[col].dropna()
        counts[col] = {
            "Gold": (vals == 3).sum(),
            "Silver": (vals == 2).sum(),
            "Bronze": (vals == 1).sum(),
            "None": (vals == 0).sum() + (vals == 0.5).sum(),
        }
    df = pd.DataFrame(counts).T
    df["Total Medals"] = df["Gold"] + df["Silver"] + df["Bronze"]
    df = df.sort_values("Total Medals", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.35)))
    colors = {"Gold": "#ffd700", "Silver": "#c0c0c0", "Bronze": "#cd7f32",
              "None": "#2c3e50"}
    left = np.zeros(len(df))
    for medal_type in ["Gold", "Silver", "Bronze", "None"]:
        ax.barh(range(len(df)), df[medal_type], left=left,
                label=medal_type, color=colors[medal_type], alpha=0.85)
        left += df[medal_type].values

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=6)
    ax.set_xlabel("Count")
    ax.set_title("MLE-bench — Medal Distribution per Agent",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def main():
    medal, binary, meta = load_data()
    n_m = len(medal.select_dtypes(include=[np.number]).columns)
    print(f"Loaded: {medal.shape[0]} competitions x {n_m} agents")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(medal)
    plot_category_heatmap(binary, meta)
    plot_task_difficulty(binary, meta)
    plot_model_accuracy(medal)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
