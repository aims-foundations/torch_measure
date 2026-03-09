"""
Visualize the BigCodeBench response matrices.

Produces:
1. Full heatmap (153 models x 1,140 tasks, sorted by accuracy/difficulty) — primary
2. Variant-level heatmap (models x 4 variants, analogous to category-level)
3. Task difficulty distribution
4. Model accuracy bar chart
5. Model-model correlation (top 30, clustered)
6. Complete vs Instruct comparison (scatter + bar)
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

VARIANTS = {
    "Full-Complete": "response_matrix.csv",
    "Full-Instruct": "response_matrix_instruct.csv",
    "Hard-Complete": "response_matrix_hard_complete.csv",
    "Hard-Instruct": "response_matrix_hard_instruct.csv",
}


def load_matrix(filename):
    """Load a response matrix CSV."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path, index_col=0)


def plot_full_heatmap(matrix, variant_label, filename_prefix):
    """Full heatmap of models x tasks (no tick labels on x — too dense)."""
    df = matrix.copy()
    n_models, n_tasks = df.shape

    # Sort rows by overall accuracy (best on top)
    row_acc = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_acc.index]

    # Sort columns by difficulty (easiest left, hardest right)
    col_diff = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_diff.index]

    fig, ax = plt.subplots(figsize=(24, 12))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.5},
        xticklabels=False, yticklabels=True,
    )
    ax.set_ylabel("Model (sorted by accuracy, best top)")
    ax.set_xlabel(f"Tasks ({n_tasks} items, sorted by difficulty)")
    ax.set_title(
        f"BigCodeBench {variant_label} Response Matrix "
        f"({n_models} models x {n_tasks} tasks)",
        fontsize=16, fontweight="bold"
    )
    plt.yticks(fontsize=5)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename_prefix}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/{filename_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename_prefix}.pdf/png")


def plot_variant_heatmap():
    """Variant-level heatmap: models x 4 variants (mean accuracy per variant).

    Analogous to the BFCL category-level heatmap.
    """
    # Collect per-model per-variant accuracy
    all_models = set()
    variant_data = {}
    for label, fname in VARIANTS.items():
        mat = load_matrix(fname)
        variant_data[label] = mat.mean(axis=1)
        all_models.update(mat.index.tolist())

    # Build combined DataFrame
    combined = pd.DataFrame(index=sorted(all_models))
    for label in VARIANTS:
        combined[label] = variant_data[label]

    # Sort rows by Full-Complete accuracy (descending), NaN at bottom
    combined = combined.sort_values("Full-Complete", ascending=False, na_position="last")

    fig, ax = plt.subplots(figsize=(10, 20))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        combined, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 5},
        cbar_kws={"label": "Pass@1 Rate", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Benchmark Variant")
    ax.set_ylabel("Model (sorted by Full-Complete accuracy, best top)")
    ax.set_title(
        f"BigCodeBench — Per-Variant Accuracy "
        f"({len(combined)} models x {len(VARIANTS)} variants)",
        fontsize=14, fontweight="bold"
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_variant.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_variant.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_variant.pdf/png")


def plot_task_difficulty():
    """Histogram of per-task pass rates, one subplot per variant."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, (label, fname) in zip(axes.flat, VARIANTS.items()):
        matrix = load_matrix(fname)
        item_rates = matrix.mean(axis=0)

        bins = np.linspace(0, 1, 41)
        ax.hist(item_rates, bins=bins, alpha=0.8,
                color=sns.color_palette("Set2")[0], edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        median_rate = np.median(item_rates)
        ax.axvline(median_rate, color="red", linestyle="--", alpha=0.7,
                   label=f"median = {median_rate:.2f}")

        n_tasks = len(item_rates)
        unsolved = (item_rates == 0).sum()
        trivial = (item_rates == 1).sum()

        ax.set_xlabel("Task Pass Rate (across all models)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({n_tasks} tasks)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        # Annotate unsolved / trivial
        txt_parts = []
        if unsolved > 0:
            txt_parts.append(f"{unsolved} unsolved")
        if trivial > 0:
            txt_parts.append(f"{trivial} trivial")
        if txt_parts:
            ax.text(0.98, 0.95, "\n".join(txt_parts), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    fig.suptitle("BigCodeBench — Task Difficulty Distribution",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy():
    """Horizontal bar chart of model accuracy (Full-Complete variant)."""
    matrix = load_matrix(VARIANTS["Full-Complete"])
    model_acc = matrix.mean(axis=1).sort_values(ascending=True)
    n_models = len(model_acc)

    fig, ax = plt.subplots(figsize=(10, max(12, n_models * 0.18)))
    colors = sns.color_palette("viridis", n_models)
    ax.barh(range(n_models), model_acc.values * 100, color=colors)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_acc.index, fontsize=6)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(
        f"BigCodeBench Full-Complete — Model Accuracy "
        f"({n_models} models, {matrix.shape[1]} tasks)",
        fontsize=14, fontweight="bold"
    )

    for i, (model, acc) in enumerate(model_acc.items()):
        ax.text(acc * 100 + 0.3, i, f"{acc*100:.1f}%", va="center", fontsize=4.5)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_model_correlation():
    """Model-model correlation heatmap (top 30 by accuracy, clustered)."""
    matrix = load_matrix(VARIANTS["Full-Complete"])

    # Pick top 30 models
    model_acc = matrix.mean(axis=1).sort_values(ascending=False)
    top_models = model_acc.head(30).index.tolist()
    sub = matrix.loc[top_models]

    corr = sub.T.corr()

    # Cluster
    link = linkage(corr.values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(12, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.1, vmax=1,
        mask=mask, square=True,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 6},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("BigCodeBench — Model-Model Correlation (Top 30, clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_correlation.pdf/png")


def plot_complete_vs_instruct():
    """Scatter plot: Complete accuracy vs Instruct accuracy per model."""
    summary_path = os.path.join(DATA_DIR, "model_summary.csv")
    df = pd.read_csv(summary_path)

    # Full benchmark
    both_full = df.dropna(subset=["complete_pass_rate", "instruct_pass_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Full benchmark scatter
    ax = axes[0]
    ax.scatter(both_full["complete_pass_rate"] * 100,
               both_full["instruct_pass_rate"] * 100,
               alpha=0.6, s=30, color=sns.color_palette("Set2")[0], edgecolor="gray")
    lim = [0, max(both_full["complete_pass_rate"].max(),
                  both_full["instruct_pass_rate"].max()) * 105]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Complete Pass Rate (%)")
    ax.set_ylabel("Instruct Pass Rate (%)")
    ax.set_title("Full Benchmark: Complete vs Instruct", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Annotate top-5 by complete
    top5 = both_full.nlargest(5, "complete_pass_rate")
    for _, row in top5.iterrows():
        ax.annotate(row["model"],
                    (row["complete_pass_rate"] * 100, row["instruct_pass_rate"] * 100),
                    fontsize=6, alpha=0.8, textcoords="offset points", xytext=(5, 5))

    # Panel 2: Hard benchmark scatter
    both_hard = df.dropna(subset=["hard_complete_pass_rate", "hard_instruct_pass_rate"])
    ax = axes[1]
    ax.scatter(both_hard["hard_complete_pass_rate"] * 100,
               both_hard["hard_instruct_pass_rate"] * 100,
               alpha=0.6, s=30, color=sns.color_palette("Set2")[1], edgecolor="gray")
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Hard-Complete Pass Rate (%)")
    ax.set_ylabel("Hard-Instruct Pass Rate (%)")
    ax.set_title("Hard Subset: Complete vs Instruct", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    top5h = both_hard.nlargest(5, "hard_complete_pass_rate")
    for _, row in top5h.iterrows():
        ax.annotate(row["model"],
                    (row["hard_complete_pass_rate"] * 100,
                     row["hard_instruct_pass_rate"] * 100),
                    fontsize=6, alpha=0.8, textcoords="offset points", xytext=(5, 5))

    fig.suptitle("BigCodeBench — Complete vs Instruct Pass Rates",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/complete_vs_instruct.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/complete_vs_instruct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved complete_vs_instruct.pdf/png")


def main():
    print("Visualizing BigCodeBench response matrices...")
    print(f"Output directory: {FIG_DIR}\n")

    # 1. Full heatmaps (primary + hard)
    primary = load_matrix(VARIANTS["Full-Complete"])
    print(f"Loaded Full-Complete: {primary.shape[0]} models x {primary.shape[1]} tasks")
    plot_full_heatmap(primary, "Full-Complete", "heatmap_full")

    hard = load_matrix(VARIANTS["Hard-Complete"])
    print(f"Loaded Hard-Complete: {hard.shape[0]} models x {hard.shape[1]} tasks")
    plot_full_heatmap(hard, "Hard-Complete", "heatmap_hard")

    # 2. Variant-level heatmap
    plot_variant_heatmap()

    # 3. Task difficulty
    plot_task_difficulty()

    # 4. Model accuracy bar chart
    plot_model_accuracy()

    # 5. Model correlation
    plot_model_correlation()

    # 6. Complete vs Instruct
    plot_complete_vs_instruct()

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
