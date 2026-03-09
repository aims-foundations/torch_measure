"""
Visualize the SWE-PolyBench response matrices.

Produces:
1. Full heatmap (verified: 382 instances x 3 models)
2. Language-level heatmap (models x programming languages)
3. Task difficulty distribution by language
4. Model accuracy bar chart
5. Task category breakdown
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
    verified = pd.read_csv(f"{DATA_DIR}/response_matrix_verified.csv", index_col=0)
    meta_path = f"{DATA_DIR}/instance_metadata_verified.csv"
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    return verified, meta


def plot_full_heatmap(matrix):
    df = matrix.select_dtypes(include=[np.number]).copy()
    n_t, n_m = df.shape
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    fig, ax = plt.subplots(figsize=(8, 16))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Resolved (1) / Failed (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Instances ({n_t})")
    ax.set_xlabel(f"Models ({n_m})")
    ax.set_title(f"SWE-PolyBench Verified ({n_t} instances x {n_m} models)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_language_heatmap(matrix, meta):
    if meta is None:
        return
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    lang_map = dict(zip(meta["instance_id"].astype(str), meta["language"]))
    languages = sorted({lang_map.get(str(t), "unknown") for t in matrix.index})

    lang_matrix = pd.DataFrame(index=num_cols, columns=languages, dtype=float)
    for lang in languages:
        mask = [lang_map.get(str(t)) == lang for t in matrix.index]
        if any(mask):
            lang_matrix[lang] = matrix.loc[mask, num_cols].mean(axis=0)

    lang_matrix = lang_matrix.sort_values(lang_matrix.columns[0],
                                          ascending=False, na_position="last")

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(lang_matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Resolve Rate", "shrink": 0.6})
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    ax.set_title("SWE-PolyBench — Per-Language Resolve Rate",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_language.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_language.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_language.pdf/png")


def plot_task_difficulty(matrix, meta):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    item_rates = matrix[num_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    if meta is not None:
        lang_map = dict(zip(meta["instance_id"].astype(str), meta["language"]))
        langs = [lang_map.get(str(t), "unknown") for t in item_rates.index]
        lang_rates = {}
        for idx, lang in zip(item_rates.index, langs):
            lang_rates.setdefault(lang, []).append(item_rates[idx])
        colors = sns.color_palette("tab10", len(lang_rates))
        bins = np.linspace(0, 1, 11)
        for (lang, vals), color in zip(sorted(lang_rates.items()), colors):
            ax.hist(vals, bins=bins, alpha=0.7, label=f"{lang} (n={len(vals)})",
                    color=color, edgecolor="white", linewidth=0.3)
        ax.legend(fontsize=8, ncol=2)
    else:
        bins = np.linspace(0, 1, 21)
        ax.hist(item_rates, bins=bins, alpha=0.8, edgecolor="white",
                color=sns.color_palette("Set2")[0])
    ax.set_xlabel("Instance Resolve Rate (across models)")
    ax.set_ylabel("Count")
    ax.set_title("SWE-PolyBench — Instance Difficulty Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(matrix):
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    model_acc = matrix[num_cols].mean(axis=0).sort_values(ascending=True) * 100
    n = len(model_acc)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.6)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), model_acc.values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_acc.index, fontsize=9)
    ax.set_xlabel("Resolve Rate (%)")
    ax.set_title(f"SWE-PolyBench — Model Resolve Rates ({n} models)",
                 fontsize=14, fontweight="bold")
    for i, (m, v) in enumerate(model_acc.items()):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def main():
    matrix, meta = load_data()
    n_m = len(matrix.select_dtypes(include=[np.number]).columns)
    print(f"Loaded: {matrix.shape[0]} instances x {n_m} models")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(matrix)
    plot_language_heatmap(matrix, meta)
    plot_task_difficulty(matrix, meta)
    plot_model_accuracy(matrix)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
