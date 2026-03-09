"""
Visualize the EvalPlus response matrices.

Produces:
1. Full heatmap (HumanEval+, 31 models x 164 tasks, sorted)
2. Variant-level heatmap (models x 4 variants: HE-base, HE+, MBPP-base, MBPP+)
3. Task difficulty distribution
4. Model accuracy bar chart
5. Model-model correlation (clustered)
6. Base vs Plus comparison (scatter)
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
    "HumanEval-Base": "response_matrix_humaneval_base.csv",
    "HumanEval+": "response_matrix_humaneval_plus.csv",
    "MBPP-Base": "response_matrix_mbpp_base.csv",
    "MBPP+": "response_matrix_mbpp_plus.csv",
}


def load_matrix(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path, index_col=0)


def plot_full_heatmap():
    """Full heatmaps for HumanEval+ and MBPP+ (primary benchmarks)."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for ax, (label, fname) in zip(axes, [
        ("HumanEval+", "response_matrix_humaneval_plus.csv"),
        ("MBPP+", "response_matrix_mbpp_plus.csv"),
    ]):
        matrix = load_matrix(fname)
        n_m, n_t = matrix.shape

        # Sort rows by accuracy (best top), columns by difficulty
        row_acc = matrix.mean(axis=1).sort_values(ascending=False)
        matrix = matrix.loc[row_acc.index]
        col_diff = matrix.mean(axis=0).sort_values(ascending=False)
        matrix = matrix[col_diff.index]

        cmap = sns.color_palette("RdYlGn", as_cmap=True)
        sns.heatmap(
            matrix.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
            cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.5},
            xticklabels=False, yticklabels=True,
        )
        ax.set_ylabel("Model (sorted by accuracy, best top)")
        ax.set_xlabel(f"Tasks ({n_t} items, sorted by difficulty)")
        ax.set_title(f"{label} ({n_m} models x {n_t} tasks)",
                     fontsize=13, fontweight="bold")
        plt.sca(ax)
        plt.yticks(fontsize=7)

    fig.suptitle("EvalPlus Response Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_variant_heatmap():
    """Variant-level heatmap: models x 4 variants (mean accuracy per variant)."""
    all_models = set()
    variant_data = {}
    for label, fname in VARIANTS.items():
        mat = load_matrix(fname)
        variant_data[label] = mat.mean(axis=1)
        all_models.update(mat.index.tolist())

    combined = pd.DataFrame(index=sorted(all_models))
    for label in VARIANTS:
        combined[label] = variant_data[label]

    # Sort by HumanEval+ accuracy
    combined = combined.sort_values(
        "HumanEval+", ascending=False, na_position="last"
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        combined, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        cbar_kws={"label": "Pass@1 Rate", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Benchmark Variant")
    ax.set_ylabel("Model (sorted by HumanEval+ accuracy, best top)")
    ax.set_title(
        f"EvalPlus — Per-Variant Accuracy "
        f"({len(combined)} models x {len(VARIANTS)} variants)",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_variant.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_variant.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_variant.pdf/png")


def plot_task_difficulty():
    """Histogram of per-task pass rates for each variant."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, (label, fname) in zip(axes.flat, VARIANTS.items()):
        matrix = load_matrix(fname)
        item_rates = matrix.mean(axis=0)

        bins = np.linspace(0, 1, 26)
        color = sns.color_palette("Set2")[list(VARIANTS.keys()).index(label)]
        ax.hist(item_rates, bins=bins, alpha=0.8, color=color,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        median_rate = np.median(item_rates)
        ax.axvline(median_rate, color="red", linestyle="--", alpha=0.7,
                   label=f"median = {median_rate:.2f}")

        unsolved = (item_rates == 0).sum()
        trivial = (item_rates == 1).sum()
        ax.set_xlabel("Task Pass Rate (across models)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({len(item_rates)} tasks)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        txt_parts = []
        if unsolved > 0:
            txt_parts.append(f"{unsolved} unsolved")
        if trivial > 0:
            txt_parts.append(f"{trivial} trivial")
        if txt_parts:
            ax.text(0.98, 0.95, "\n".join(txt_parts), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    fig.suptitle("EvalPlus — Task Difficulty Distribution",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy():
    """Horizontal bar chart of model accuracy for HumanEval+ and MBPP+."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, (label, fname), color in zip(
        axes,
        [("HumanEval+", "response_matrix_humaneval_plus.csv"),
         ("MBPP+", "response_matrix_mbpp_plus.csv")],
        ["#3498db", "#e74c3c"],
    ):
        matrix = load_matrix(fname)
        model_acc = matrix.mean(axis=1).sort_values(ascending=True)
        n = len(model_acc)

        colors = sns.color_palette("viridis", n)
        ax.barh(range(n), model_acc.values * 100, color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels(model_acc.index, fontsize=7)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"{label} ({n} models)", fontsize=12, fontweight="bold")

        for i, (model, acc) in enumerate(model_acc.items()):
            ax.text(acc * 100 + 0.5, i, f"{acc*100:.1f}%",
                    va="center", fontsize=6)

    fig.suptitle("EvalPlus — Model Accuracy", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_model_correlation():
    """Model-model correlation heatmap (HumanEval+, clustered)."""
    matrix = load_matrix(VARIANTS["HumanEval+"])
    corr = matrix.T.corr()

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
    ax.set_title("EvalPlus — Model-Model Correlation (HumanEval+, clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_correlation.pdf/png")


def plot_base_vs_plus():
    """Scatter: Base accuracy vs Plus accuracy for HumanEval and MBPP."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # HumanEval
    ax = axes[0]
    he_base = load_matrix(VARIANTS["HumanEval-Base"]).mean(axis=1) * 100
    he_plus = load_matrix(VARIANTS["HumanEval+"]).mean(axis=1) * 100
    common = he_base.index.intersection(he_plus.index)
    ax.scatter(he_base[common], he_plus[common], alpha=0.7, s=60,
               color=sns.color_palette("Set2")[0], edgecolor="gray")
    lim = [0, max(he_base[common].max(), he_plus[common].max()) + 5]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("HumanEval-Base (%)")
    ax.set_ylabel("HumanEval+ (%)")
    ax.set_title("HumanEval: Base vs Plus", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    for model in common:
        ax.annotate(model, (he_base[model], he_plus[model]),
                    fontsize=5, alpha=0.7, textcoords="offset points",
                    xytext=(4, 4))

    # MBPP
    ax = axes[1]
    mbpp_base = load_matrix(VARIANTS["MBPP-Base"]).mean(axis=1) * 100
    mbpp_plus = load_matrix(VARIANTS["MBPP+"]).mean(axis=1) * 100
    common_m = mbpp_base.index.intersection(mbpp_plus.index)
    ax.scatter(mbpp_base[common_m], mbpp_plus[common_m], alpha=0.7, s=60,
               color=sns.color_palette("Set2")[1], edgecolor="gray")
    lim_m = [0, max(mbpp_base[common_m].max(), mbpp_plus[common_m].max()) + 5]
    ax.plot(lim_m, lim_m, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("MBPP-Base (%)")
    ax.set_ylabel("MBPP+ (%)")
    ax.set_title("MBPP: Base vs Plus", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    for model in common_m:
        ax.annotate(model, (mbpp_base[model], mbpp_plus[model]),
                    fontsize=5, alpha=0.7, textcoords="offset points",
                    xytext=(4, 4))

    fig.suptitle("EvalPlus — Base vs Plus (stricter tests reduce scores)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/base_vs_plus.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/base_vs_plus.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved base_vs_plus.pdf/png")


def main():
    print("Visualizing EvalPlus response matrices...")
    print(f"Output directory: {FIG_DIR}\n")

    plot_full_heatmap()
    plot_variant_heatmap()
    plot_task_difficulty()
    plot_model_accuracy()
    plot_model_correlation()
    plot_base_vs_plus()

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
