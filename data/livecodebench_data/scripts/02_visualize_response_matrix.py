"""
Visualize the LiveCodeBench response matrix.

Produces:
1. Full heatmap (72 models × 1,055 problems, sorted by accuracy/difficulty)
2. Difficulty-level heatmap (72 models × 3 difficulty levels)
3. Platform-level heatmap (72 models × 3 platforms)
4. Task difficulty distribution by difficulty level and platform
5. Model accuracy bar chart
6. Difficulty breakdown box plot
7. Problem-problem correlation at difficulty level
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
    matrix = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/problem_metadata.csv")
    return matrix, meta


def plot_full_heatmap(matrix, meta):
    """Full heatmap of 72 models × 1,055 problems."""
    df = matrix.copy()

    # Sort rows by overall accuracy (best on top)
    row_acc = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_acc.index]

    # Build metadata lookups
    qid_to_diff = dict(zip(
        meta["question_id"].astype(str), meta["difficulty"]))
    qid_to_plat = dict(zip(
        meta["question_id"].astype(str), meta["platform"]))

    # Sort columns: first by difficulty, then by platform, then by pass rate
    col_pass = df.mean(axis=0)
    diff_order = {"easy": 0, "medium": 1, "hard": 2}
    col_order = sorted(
        range(len(df.columns)),
        key=lambda i: (
            diff_order.get(qid_to_diff.get(str(df.columns[i]), ""), 3),
            qid_to_plat.get(str(df.columns[i]), ""),
            col_pass.iloc[i],
        )
    )
    df = df.iloc[:, col_order]
    sorted_diffs = [qid_to_diff.get(str(df.columns[i]), "")
                    for i in range(len(df.columns))]

    fig, ax = plt.subplots(figsize=(24, 12))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    # Handle NaN with gray
    cmap.set_bad(color="lightgray")
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "pass@1", "shrink": 0.5},
        xticklabels=False, yticklabels=True,
    )
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_xlabel(f"Problems ({len(df.columns)} items, grouped by difficulty)")
    ax.set_title("LiveCodeBench Response Matrix (72 models × 1,055 problems)",
                 fontsize=16, fontweight="bold")
    plt.yticks(fontsize=5)

    # Add difficulty separators
    prev_diff = None
    diff_positions = []
    unique_diffs = []
    for i, d in enumerate(sorted_diffs):
        if d != prev_diff:
            if prev_diff is not None:
                ax.axvline(i, color="black", linewidth=1, alpha=0.7)
            diff_positions.append(i)
            unique_diffs.append(d)
            prev_diff = d

    # Add difficulty labels at bottom
    for i, (diff, pos) in enumerate(zip(unique_diffs, diff_positions)):
        next_pos = (diff_positions[i + 1]
                    if i + 1 < len(diff_positions)
                    else len(sorted_diffs))
        mid = (pos + next_pos) / 2
        ax.text(mid, len(df) + 1, diff.upper(), ha="center", va="top",
                fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_difficulty_heatmap(matrix, meta):
    """Difficulty-level heatmap: 72 models × 3 difficulty levels."""
    qid_to_diff = dict(zip(
        meta["question_id"].astype(str), meta["difficulty"]))

    diff_levels = ["easy", "medium", "hard"]
    diff_matrix = pd.DataFrame(
        index=matrix.index, columns=diff_levels, dtype=float)

    for diff in diff_levels:
        cols = [c for c in matrix.columns
                if qid_to_diff.get(str(c), "") == diff]
        diff_matrix[diff] = matrix[cols].mean(axis=1)

    # Sort rows by overall accuracy
    row_acc = diff_matrix.mean(axis=1).sort_values(ascending=False)
    diff_matrix = diff_matrix.loc[row_acc.index]

    fig, ax = plt.subplots(figsize=(8, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        diff_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        cbar_kws={"label": "Mean pass@1", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_title("LiveCodeBench — Per-Difficulty Accuracy",
                 fontsize=14, fontweight="bold")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_difficulty.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_difficulty.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_difficulty.pdf/png")

    return diff_matrix


def plot_platform_heatmap(matrix, meta):
    """Platform-level heatmap: 72 models × 3 platforms."""
    qid_to_plat = dict(zip(
        meta["question_id"].astype(str), meta["platform"]))

    platforms = sorted(set(qid_to_plat.values()))
    plat_matrix = pd.DataFrame(
        index=matrix.index, columns=platforms, dtype=float)

    for plat in platforms:
        cols = [c for c in matrix.columns
                if qid_to_plat.get(str(c), "") == plat]
        plat_matrix[plat] = matrix[cols].mean(axis=1)

    # Sort rows by overall accuracy
    row_acc = plat_matrix.mean(axis=1).sort_values(ascending=False)
    plat_matrix = plat_matrix.loc[row_acc.index]

    fig, ax = plt.subplots(figsize=(8, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        plat_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        cbar_kws={"label": "Mean pass@1", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Platform")
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_title("LiveCodeBench — Per-Platform Accuracy",
                 fontsize=14, fontweight="bold")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_platform.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_platform.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_platform.pdf/png")


def plot_task_difficulty(matrix, meta):
    """Histogram of per-problem pass rates, colored by difficulty."""
    qid_to_diff = dict(zip(
        meta["question_id"].astype(str), meta["difficulty"]))
    item_rates = matrix.mean(axis=0)

    diff_rates = {}
    for col in matrix.columns:
        diff = qid_to_diff.get(str(col), "unknown")
        diff_rates.setdefault(diff, []).append(item_rates[col])

    diff_order = ["easy", "medium", "hard"]
    colors = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 1, 41)
    for diff in reversed(diff_order):
        if diff in diff_rates:
            ax.hist(diff_rates[diff], bins=bins, alpha=0.7, label=diff,
                    color=colors[diff], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Problem Pass Rate (across 72 models)")
    ax.set_ylabel("Count")
    ax.set_title("LiveCodeBench — Problem Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, title="Difficulty")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(matrix):
    """Horizontal bar chart of model accuracy."""
    model_acc = matrix.mean(axis=1).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 16))
    colors = sns.color_palette("viridis", len(model_acc))
    ax.barh(range(len(model_acc)), model_acc.values * 100, color=colors)
    ax.set_yticks(range(len(model_acc)))
    ax.set_yticklabels(model_acc.index, fontsize=7)
    ax.set_xlabel("Mean pass@1 (%)")
    ax.set_title("LiveCodeBench — Model Accuracy (72 models)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 95)

    for i, (model, acc) in enumerate(model_acc.items()):
        ax.text(acc * 100 + 0.3, i, f"{acc*100:.1f}%",
                va="center", fontsize=5)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_difficulty_breakdown(matrix, meta):
    """Box plot of per-problem pass rates grouped by difficulty."""
    qid_to_diff = dict(zip(
        meta["question_id"].astype(str), meta["difficulty"]))
    item_rates = matrix.mean(axis=0)

    df = pd.DataFrame({
        "question_id": matrix.columns,
        "difficulty": [qid_to_diff.get(str(c), "unknown")
                       for c in matrix.columns],
        "pass_rate": item_rates.values,
    })
    df = df[df["difficulty"].isin(["easy", "medium", "hard"])]

    diff_order = ["easy", "medium", "hard"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="difficulty", y="pass_rate", order=diff_order,
                ax=ax, palette={"easy": "#2ca02c", "medium": "#ff7f0e",
                                "hard": "#d62728"},
                hue="difficulty", legend=False, showfliers=False)
    sns.stripplot(data=df, x="difficulty", y="pass_rate", order=diff_order,
                  ax=ax, color="black", size=1.5, alpha=0.3, jitter=True)
    ax.set_ylabel("Problem Pass Rate")
    ax.set_xlabel("")
    ax.set_title("LiveCodeBench — Problem Pass Rate by Difficulty",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    # Add item counts
    for i, diff in enumerate(diff_order):
        n = len(df[df["difficulty"] == diff])
        ax.text(i, -0.03, f"n={n}", ha="center", va="top", fontsize=9,
                color="gray")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/difficulty_breakdown.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/difficulty_breakdown.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved difficulty_breakdown.pdf/png")


def plot_platform_breakdown(matrix, meta):
    """Box plot of per-problem pass rates grouped by platform."""
    qid_to_plat = dict(zip(
        meta["question_id"].astype(str), meta["platform"]))
    item_rates = matrix.mean(axis=0)

    df = pd.DataFrame({
        "question_id": matrix.columns,
        "platform": [qid_to_plat.get(str(c), "unknown")
                     for c in matrix.columns],
        "pass_rate": item_rates.values,
    })

    plat_order = (df.groupby("platform")["pass_rate"]
                  .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="platform", y="pass_rate", order=plat_order,
                ax=ax, palette="Set2", hue="platform",
                legend=False, showfliers=False)
    sns.stripplot(data=df, x="platform", y="pass_rate", order=plat_order,
                  ax=ax, color="black", size=1.5, alpha=0.3, jitter=True)
    ax.set_ylabel("Problem Pass Rate")
    ax.set_xlabel("")
    ax.set_title("LiveCodeBench — Problem Pass Rate by Platform",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    for i, plat in enumerate(plat_order):
        n = len(df[df["platform"] == plat])
        ax.text(i, -0.03, f"n={n}", ha="center", va="top", fontsize=9,
                color="gray")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/platform_breakdown.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/platform_breakdown.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved platform_breakdown.pdf/png")


def plot_difficulty_correlation(diff_matrix):
    """Difficulty-level correlation heatmap."""
    corr = diff_matrix.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=0, vmax=1,
        square=True,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".3f", annot_kws={"fontsize": 12},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("LiveCodeBench — Difficulty-Level Correlation",
                 fontsize=14, fontweight="bold")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/difficulty_correlation.pdf",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/difficulty_correlation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved difficulty_correlation.pdf/png")


def main():
    matrix, meta = load_data()
    print(f"Loaded: {matrix.shape[0]} models × {matrix.shape[1]} problems")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_full_heatmap(matrix, meta)
    diff_matrix = plot_difficulty_heatmap(matrix, meta)
    plot_platform_heatmap(matrix, meta)
    plot_task_difficulty(matrix, meta)
    plot_model_accuracy(matrix)
    plot_difficulty_breakdown(matrix, meta)
    plot_platform_breakdown(matrix, meta)
    plot_difficulty_correlation(diff_matrix)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
