"""
Visualize the DPAI response matrices.

Produces:
1. Full heatmap (binary pass@50, 141 tasks x 9 agents)
2. Variant comparison heatmap (agents x 4 scoring variants)
3. Task difficulty distribution
4. Model accuracy bar chart
5. Blind vs informed score scatter
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

VARIANTS = {
    "Binary Pass@50": "response_matrix_binary_pass50.csv",
    "Blind Score": "response_matrix_blind_score.csv",
    "Informed Score": "response_matrix_informed_score.csv",
    "Total Score": "response_matrix_total_score.csv",
}


def load_matrix(fname):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, index_col=0)
    return df.select_dtypes(include=[np.number])


def plot_full_heatmap():
    mat = load_matrix(VARIANTS["Binary Pass@50"])
    n_t, n_m = mat.shape
    row_diff = mat.mean(axis=1).sort_values(ascending=False)
    mat = mat.loc[row_diff.index]
    col_acc = mat.mean(axis=0).sort_values(ascending=False)
    mat = mat[col_acc.index]

    fig, ax = plt.subplots(figsize=(10, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(mat.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Tasks ({n_t})")
    ax.set_xlabel(f"Agents ({n_m})")
    ax.set_title(f"DPAI Binary Pass@50 ({n_t} tasks x {n_m} agents)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_variant_heatmap():
    """Agents x 4 scoring variants."""
    all_models = set()
    variant_data = {}
    for label, fname in VARIANTS.items():
        mat = load_matrix(fname)
        if label == "Binary Pass@50":
            variant_data[label] = mat.mean(axis=0)
        else:
            variant_data[label] = mat.mean(axis=0) / 100.0  # Normalize 0-100 to 0-1
        all_models.update(mat.columns.tolist())

    combined = pd.DataFrame(index=sorted(all_models))
    for label in VARIANTS:
        combined[label] = variant_data[label]

    combined = combined.sort_values("Binary Pass@50", ascending=False,
                                    na_position="last")
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(combined, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Score (normalized)", "shrink": 0.6})
    ax.set_xlabel("Scoring Variant")
    ax.set_ylabel("Agent")
    ax.set_title("DPAI — Per-Variant Mean Score",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_variant.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_variant.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_variant.pdf/png")


def plot_task_difficulty():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (label, fname) in zip(axes.flat, VARIANTS.items()):
        mat = load_matrix(fname)
        rates = mat.mean(axis=1)
        if "Score" in label:
            rates = rates / 100.0
        bins = np.linspace(0, 1, 26)
        color = sns.color_palette("Set2")[list(VARIANTS.keys()).index(label)]
        ax.hist(rates, bins=bins, alpha=0.8, color=color, edgecolor="white")
        median = np.median(rates)
        ax.axvline(median, color="red", ls="--", alpha=0.7,
                   label=f"median={median:.2f}")
        ax.set_xlabel("Task Score (normalized)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({len(rates)} tasks)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("DPAI — Task Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy():
    mat = load_matrix(VARIANTS["Binary Pass@50"])
    model_acc = mat.mean(axis=0).sort_values(ascending=True) * 100
    n = len(model_acc)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.5)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), model_acc.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_acc.index, fontsize=8)
    ax.set_xlabel("Pass@50 Rate (%)")
    ax.set_title(f"DPAI — Agent Pass@50 Rate ({n} agents)",
                 fontsize=14, fontweight="bold")
    for i, (m, v) in enumerate(model_acc.items()):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_blind_vs_informed():
    blind = load_matrix(VARIANTS["Blind Score"])
    informed = load_matrix(VARIANTS["Informed Score"])
    common = blind.columns.intersection(informed.columns)
    b_mean = blind[common].mean(axis=0)
    i_mean = informed[common].mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(b_mean, i_mean, alpha=0.7, s=80,
               color=sns.color_palette("Set2")[0], edgecolor="gray")
    lim = [0, max(b_mean.max(), i_mean.max()) + 2]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    for m in common:
        ax.annotate(m, (b_mean[m], i_mean[m]), fontsize=6, alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Blind Score (mean)")
    ax.set_ylabel("Informed Score (mean)")
    ax.set_title("DPAI — Blind vs Informed Score",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/blind_vs_informed.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/blind_vs_informed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved blind_vs_informed.pdf/png")


def main():
    print("Visualizing DPAI response matrices...")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap()
    plot_variant_heatmap()
    plot_task_difficulty()
    plot_model_accuracy()
    plot_blind_vs_informed()
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
