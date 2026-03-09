"""
Visualize the TAU-bench response matrices.

Produces:
1. Full heatmap per domain (airline_v1, retail, telecom)
2. Domain-level heatmap (models x domains)
3. Task difficulty distribution per domain
4. Model accuracy bar chart
5. Model-model correlation (combined, clustered)
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

DOMAIN_FILES = {
    "Airline (v1)": "response_matrix_v1_airline.csv",
    "Airline (v2)": "response_matrix_v2_airline.csv",
    "Retail": "response_matrix_retail.csv",
    "Telecom": "response_matrix_telecom.csv",
}


def load_domain_matrix(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df.select_dtypes(include=[np.number])


def plot_domain_heatmaps():
    """Individual heatmaps per domain."""
    valid = {k: load_domain_matrix(v) for k, v in DOMAIN_FILES.items()
             if load_domain_matrix(v) is not None and load_domain_matrix(v).shape[1] > 0}
    n = len(valid)
    if n == 0:
        print("No domain matrices found, skipping heatmaps")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 10))
    if n == 1:
        axes = [axes]
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    for ax, (label, mat) in zip(axes, valid.items()):
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
        plt.xticks(rotation=90, fontsize=6)

    fig.suptitle("TAU-bench Response Matrices by Domain",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{FIG_DIR}/heatmap_domains.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_domains.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_domains.pdf/png")


def plot_domain_level_heatmap():
    """Models x domains (mean score per domain)."""
    all_models = set()
    domain_data = {}
    for label, fname in DOMAIN_FILES.items():
        mat = load_domain_matrix(fname)
        if mat is not None and mat.shape[1] > 0:
            domain_data[label] = mat.mean(axis=0)
            all_models.update(mat.columns.tolist())

    if not domain_data:
        return

    combined = pd.DataFrame(index=sorted(all_models))
    for label, series in domain_data.items():
        combined[label] = series

    first_col = list(domain_data.keys())[0]
    combined = combined.sort_values(first_col, ascending=False, na_position="last")

    fig, ax = plt.subplots(figsize=(10, max(8, len(combined) * 0.35)))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(combined, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                cbar_kws={"label": "Mean Score", "shrink": 0.5})
    ax.set_xlabel("Domain")
    ax.set_ylabel("Model")
    ax.set_title(f"TAU-bench — Per-Domain Mean Score ({len(combined)} models)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_domain_level.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_domain_level.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_domain_level.pdf/png")


def plot_task_difficulty():
    """Per-task score distributions for each domain."""
    valid = {k: load_domain_matrix(v) for k, v in DOMAIN_FILES.items()
             if load_domain_matrix(v) is not None and load_domain_matrix(v).shape[1] > 0}
    n = len(valid)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    colors = sns.color_palette("Set2")
    for ax, (label, mat), color in zip(axes, valid.items(), colors):
        rates = mat.mean(axis=1)
        bins = np.linspace(0, 1, 21)
        ax.hist(rates, bins=bins, alpha=0.8, color=color, edgecolor="white")
        median = np.median(rates)
        ax.axvline(median, color="red", ls="--", alpha=0.7,
                   label=f"median={median:.2f}")
        ax.set_xlabel("Task Mean Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({len(rates)} tasks)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("TAU-bench — Task Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy():
    """Bar chart comparing models across domains."""
    accs = {}
    for label, fname in DOMAIN_FILES.items():
        mat = load_domain_matrix(fname)
        if mat is not None and mat.shape[1] > 0:
            accs[label] = mat.mean(axis=0) * 100
    if not accs:
        return
    combined = pd.DataFrame(accs).dropna(how="all")
    sort_col = combined.columns[0]
    combined = combined.sort_values(sort_col, ascending=True, na_position="first")

    fig, ax = plt.subplots(figsize=(12, max(8, len(combined) * 0.4)))
    x = np.arange(len(combined))
    w = 0.8 / len(combined.columns)
    for i, col in enumerate(combined.columns):
        vals = combined[col].fillna(0)
        ax.barh(x + i * w, vals, w, label=col, alpha=0.85)
    ax.set_yticks(x + w * len(combined.columns) / 2)
    ax.set_yticklabels(combined.index, fontsize=7)
    ax.set_xlabel("Mean Score (%)")
    ax.set_title("TAU-bench — Model Scores Across Domains",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def main():
    print("Visualizing TAU-bench response matrices...")
    print(f"Output: {FIG_DIR}\n")
    plot_domain_heatmaps()
    plot_domain_level_heatmap()
    plot_task_difficulty()
    plot_model_accuracy()
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
