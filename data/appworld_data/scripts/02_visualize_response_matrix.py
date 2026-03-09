"""
Visualize the AppWorld benchmark results.

Note: AppWorld provides aggregate per-agent metrics (TGC/SGC), not per-task
binary data. The response_matrix.csv is a leaderboard table of 18 agent entries
with Task Goal Completion (TGC) and Scenario Goal Completion (SGC) by split and
difficulty level.

Produces:
1. Agent leaderboard bar chart (TGC on test_normal)
2. Difficulty-level grouped bar chart
3. Test Normal vs Test Challenge comparison scatter
4. Interactions vs performance scatter
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
    df = pd.read_csv(f"{DATA_DIR}/response_matrix.csv")
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return df, meta


def plot_leaderboard(df):
    """Bar chart of TGC on test_normal for all agents."""
    col = "test_normal__all__task_goal_completion"
    best = df.sort_values(col, ascending=True).copy()
    n = len(best)

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.4)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), best[col].values, color=colors, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(best["model_agent"].values, fontsize=7)
    ax.set_xlabel("Task Goal Completion (%)")
    ax.set_title(f"AppWorld — Test Normal TGC ({n} agents)",
                 fontsize=14, fontweight="bold")
    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(r[col] + 0.3, i, f"{r[col]:.1f}%", va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_difficulty_breakdown(df):
    """Grouped bar chart: TGC by difficulty level for each agent."""
    levels = ["level_1", "level_2", "level_3"]
    cols = [f"test_normal__{lv}__task_goal_completion" for lv in levels]
    labels = ["Level 1 (Easy)", "Level 2 (Medium)", "Level 3 (Hard)"]

    agents = df.sort_values(cols[0], ascending=True)["model_agent"].values
    n = len(agents)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.45)))
    x = np.arange(n)
    width = 0.25
    palette = ["#2ecc71", "#f39c12", "#e74c3c"]

    for j, (col, label, color) in enumerate(zip(cols, labels, palette)):
        vals = df.set_index("model_agent").loc[agents, col].values
        ax.barh(x + (j - 1) * width, vals, width, label=label, color=color,
                alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(agents, fontsize=7)
    ax.set_xlabel("Task Goal Completion (%)")
    ax.set_title("AppWorld — TGC by Difficulty Level (Test Normal)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/difficulty_breakdown.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/difficulty_breakdown.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved difficulty_breakdown.pdf/png")


def plot_normal_vs_challenge(df):
    """Scatter: test_normal TGC vs test_challenge TGC."""
    col_n = "test_normal__all__task_goal_completion"
    col_c = "test_challenge__all__task_goal_completion"
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df[col_n], df[col_c], alpha=0.7, s=80,
               color=sns.color_palette("Set2")[0], edgecolor="gray")
    lim = [0, max(df[col_n].max(), df[col_c].max()) + 5]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    for _, r in df.iterrows():
        ax.annotate(r["model_agent"], (r[col_n], r[col_c]),
                    fontsize=5, alpha=0.7, textcoords="offset points",
                    xytext=(3, 3))
    ax.set_xlabel("Test Normal — TGC (%)")
    ax.set_ylabel("Test Challenge — TGC (%)")
    ax.set_title("AppWorld — Normal vs Challenge Performance",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/normal_vs_challenge.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/normal_vs_challenge.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved normal_vs_challenge.pdf/png")


def plot_interactions_vs_performance(df):
    """Scatter: average interactions vs TGC (test_normal)."""
    col_tgc = "test_normal__all__task_goal_completion"
    col_int = "test_normal__all__interactions"
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df[col_int], df[col_tgc], alpha=0.7, s=80,
               color=sns.color_palette("Set2")[1], edgecolor="gray")
    for _, r in df.iterrows():
        ax.annotate(r["model_agent"], (r[col_int], r[col_tgc]),
                    fontsize=5, alpha=0.7, textcoords="offset points",
                    xytext=(3, 3))
    ax.set_xlabel("Average Interactions")
    ax.set_ylabel("Task Goal Completion (%)")
    ax.set_title("AppWorld — Interactions vs Performance",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/interactions_vs_performance.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/interactions_vs_performance.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved interactions_vs_performance.pdf/png")


def main():
    df, meta = load_data()
    print(f"Loaded: {len(df)} agent entries, {len(meta)} tasks in metadata")
    print(f"Output: {FIG_DIR}\n")
    plot_leaderboard(df)
    plot_difficulty_breakdown(df)
    plot_normal_vs_challenge(df)
    plot_interactions_vs_performance(df)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
