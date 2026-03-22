#!/usr/bin/env python3
"""
Visualize the Bridging-the-Gap African Languages response matrix.

Produces:
1. Response matrix heatmap (items x model-language, sorted)
2. Model accuracy by language bar chart
3. Language accuracy comparison (English vs African languages)
4. Item difficulty distribution
5. Language family accuracy comparison
6. Cross-language correlation heatmap
7. Per-model language gap analysis
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BENCHMARK_DIR / "processed"
FIG_DIR = _BENCHMARK_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=0.9)

# Language metadata
LANG_NAMES = {
    "en": "English", "xh": "Xhosa", "zu": "Zulu", "af": "Afrikaans",
    "ig": "Igbo", "sn": "Shona", "ts": "Tsonga", "st": "Sesotho",
    "nso": "Sepedi", "tn": "Setswana", "bm": "Bambara", "am": "Amharic",
}

LANG_FAMILIES = {
    "en": "Germanic", "af": "Germanic",
    "xh": "Nguni", "zu": "Nguni",
    "sn": "Shona", "ts": "Tsonga",
    "st": "Sotho-Tswana", "nso": "Sotho-Tswana", "tn": "Sotho-Tswana",
    "ig": "Igboid", "bm": "Manding", "am": "Semitic",
}


def load_data():
    """Load all processed data."""
    response_matrix = pd.read_csv(DATA_DIR / "response_matrix.csv", index_col=0)
    averaged_matrix = pd.read_csv(DATA_DIR / "response_matrix_averaged.csv", index_col=0)
    model_summary = pd.read_csv(DATA_DIR / "model_summary.csv")
    task_metadata = pd.read_csv(DATA_DIR / "task_metadata.csv")
    return response_matrix, averaged_matrix, model_summary, task_metadata


def parse_model_language(col_name):
    """Parse 'gpt-4o_en' into (model, language)."""
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return col_name, "unknown"


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Response matrix heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_response_heatmap(matrix):
    """Full heatmap of items x model-language combinations."""
    df = matrix.copy()

    # Sort columns by accuracy (best first)
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    # Sort rows by difficulty (easiest first)
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]

    n_items, n_cols = df.shape

    fig, ax = plt.subplots(figsize=(18, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Correct (1) / Incorrect (0)", "shrink": 0.4},
        xticklabels=True, yticklabels=False,
    )
    ax.set_ylabel(f"Items ({n_items}, sorted by difficulty)")
    ax.set_xlabel(f"Model-Language ({n_cols} combinations, sorted by accuracy)")
    ax.set_title(
        f"Bridging-the-Gap — Winogrande Response Matrix\n"
        f"({n_items} items x {n_cols} model-language combos, majority vote)",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"heatmap_full.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved heatmap_full.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Model accuracy by language (grouped bar chart)
# ──────────────────────────────────────────────────────────────────────
def plot_model_accuracy_by_language(model_summary):
    """Grouped bar chart: accuracy by language for each model."""
    df = model_summary.copy()

    # Pivot for grouped bar chart
    pivot = df.pivot(index="language", columns="model", values="accuracy_mean_pct")

    # Sort languages: English first, then by mean accuracy descending
    lang_order = ["en"] + [
        l for l in pivot.index if l != "en"
    ]
    # Sort non-English by mean across models
    non_en = [l for l in lang_order if l != "en"]
    non_en_sorted = sorted(non_en, key=lambda l: pivot.loc[l].mean(), reverse=True)
    lang_order = ["en"] + non_en_sorted
    pivot = pivot.loc[lang_order]

    # Add language names
    lang_labels = [f"{l} ({LANG_NAMES.get(l, l)})" for l in lang_order]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(lang_order))
    width = 0.25
    models = sorted(pivot.columns)
    colors = {"gpt-4o": "#1f77b4", "gpt-4": "#ff7f0e", "gpt-3.5": "#2ca02c"}

    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        vals = pivot[model].values
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors.get(model, f"C{i}"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Bridging-the-Gap — Model Accuracy by Language (Winogrande)",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Model", fontsize=10)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.4, label="Random baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"accuracy_by_language.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved accuracy_by_language.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 3: English vs African language gap
# ──────────────────────────────────────────────────────────────────────
def plot_language_gap(model_summary):
    """Horizontal bar chart showing gap between English and each African language."""
    df = model_summary.copy()

    # Get English accuracy per model
    en_acc = df[df["language"] == "en"].set_index("model")["accuracy_mean_pct"]

    # Compute gaps
    afr_rows = df[df["language"] != "en"].copy()
    afr_rows["en_accuracy"] = afr_rows["model"].map(en_acc)
    afr_rows["gap"] = afr_rows["en_accuracy"] - afr_rows["accuracy_mean_pct"]

    # Average gap by language
    lang_gap = afr_rows.groupby("language").agg(
        mean_accuracy=("accuracy_mean_pct", "mean"),
        mean_gap=("gap", "mean"),
    ).sort_values("mean_gap")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: accuracy
    lang_labels = [f"{l} ({LANG_NAMES.get(l, l)})" for l in lang_gap.index]
    colors_acc = sns.color_palette("YlOrRd_r", len(lang_gap))
    axes[0].barh(lang_labels, lang_gap["mean_accuracy"].values, color=colors_acc)
    axes[0].set_xlabel("Mean Accuracy (%)")
    axes[0].set_title("Mean Accuracy per African Language\n(averaged across models and runs)")
    axes[0].axvline(50, color="gray", linestyle="--", alpha=0.4)
    for i, v in enumerate(lang_gap["mean_accuracy"].values):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: gap from English
    colors_gap = sns.color_palette("Reds", len(lang_gap))
    axes[1].barh(lang_labels, lang_gap["mean_gap"].values, color=colors_gap)
    axes[1].set_xlabel("Accuracy Gap from English (%)")
    axes[1].set_title("Performance Gap: English minus African Language\n(averaged across models and runs)")
    for i, v in enumerate(lang_gap["mean_gap"].values):
        axes[1].text(v + 0.3, i, f"{v:.1f}pp", va="center", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"language_gap.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved language_gap.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 4: Item difficulty distribution
# ──────────────────────────────────────────────────────────────────────
def plot_item_difficulty(matrix):
    """Histogram of per-item pass rates."""
    item_pass_rate = matrix.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall histogram
    axes[0].hist(item_pass_rate, bins=40, color="#2ca02c", edgecolor="white",
                 alpha=0.8)
    axes[0].axvline(item_pass_rate.mean(), color="red", linestyle="--",
                    label=f"Mean: {item_pass_rate.mean():.2f}")
    axes[0].axvline(0.5, color="gray", linestyle=":", alpha=0.5,
                    label="Random baseline")
    axes[0].set_xlabel("Pass Rate (across all model-language combos)")
    axes[0].set_ylabel("Number of Items")
    axes[0].set_title("Item Difficulty Distribution")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Per-language pass rate comparison
    model_langs = matrix.columns
    lang_pass_rates = {}
    for col in model_langs:
        _, lang = parse_model_language(col)
        if lang not in lang_pass_rates:
            lang_pass_rates[lang] = []
        lang_pass_rates[lang].append(matrix[col])

    lang_means = {}
    for lang, cols_data in lang_pass_rates.items():
        combined = pd.concat(cols_data, axis=1).mean(axis=1)
        lang_means[lang] = combined

    # Box plot of item difficulty by language
    boxplot_data = pd.DataFrame(lang_means)
    # Reorder: English first, then by median
    cols_order = ["en"] + sorted(
        [c for c in boxplot_data.columns if c != "en"],
        key=lambda c: boxplot_data[c].median(), reverse=True
    )
    boxplot_data = boxplot_data[cols_order]
    col_labels = [f"{c}\n({LANG_NAMES.get(c, c)})" for c in cols_order]

    bp = axes[1].boxplot(
        [boxplot_data[c].dropna().values for c in cols_order],
        labels=col_labels, patch_artist=True, showfliers=False
    )
    colors_box = ["#1f77b4"] + sns.color_palette("Set2", len(cols_order) - 1).as_hex()
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel("Item Pass Rate")
    axes[1].set_title("Item Difficulty by Language")
    axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"item_difficulty.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved item_difficulty.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 5: Language family accuracy comparison
# ──────────────────────────────────────────────────────────────────────
def plot_language_family(model_summary):
    """Bar chart grouping languages by language family."""
    df = model_summary.copy()
    df["family"] = df["language"].map(LANG_FAMILIES)

    family_acc = df.groupby("family")["accuracy_mean_pct"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(family_acc))
    bars = ax.barh(family_acc.index, family_acc.values, color=colors)

    for bar, val in zip(bars, family_acc.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Mean Accuracy (%)")
    ax.set_title("Bridging-the-Gap — Accuracy by Language Family",
                 fontsize=14, fontweight="bold")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"language_family.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved language_family.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 6: Cross-language correlation
# ──────────────────────────────────────────────────────────────────────
def plot_language_correlation(matrix):
    """Heatmap of per-language item-level correlations."""
    # Average across models for each language
    model_langs = matrix.columns
    lang_data = {}
    for col in model_langs:
        _, lang = parse_model_language(col)
        if lang not in lang_data:
            lang_data[lang] = []
        lang_data[lang].append(matrix[col])

    lang_avg = {}
    for lang, cols_data in lang_data.items():
        lang_avg[lang] = pd.concat(cols_data, axis=1).mean(axis=1)

    lang_df = pd.DataFrame(lang_avg)
    lang_labels = {l: f"{l} ({LANG_NAMES.get(l, l)})" for l in lang_df.columns}
    lang_df = lang_df.rename(columns=lang_labels)

    corr = lang_df.corr()

    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
        mask=mask, square=True,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
    )
    ax.set_title("Bridging-the-Gap — Cross-Language Item Correlation\n"
                 "(averaged across models, majority vote)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"language_correlation.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved language_correlation.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Plot 7: Per-model language gap heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_model_language_heatmap(model_summary):
    """Heatmap of accuracy: models x languages."""
    df = model_summary.copy()
    pivot = df.pivot(index="model", columns="language", values="accuracy_mean_pct")

    # Order languages: English first, then by mean accuracy
    lang_order = ["en"] + sorted(
        [c for c in pivot.columns if c != "en"],
        key=lambda c: pivot[c].mean(), reverse=True
    )
    pivot = pivot[lang_order]
    col_labels = [f"{l}\n({LANG_NAMES.get(l, l)})" for l in lang_order]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot.values, ax=ax,
        cmap="RdYlGn", vmin=40, vmax=90,
        annot=True, fmt=".1f", annot_kws={"fontsize": 9},
        linewidths=0.5, linecolor="white",
        xticklabels=col_labels,
        yticklabels=pivot.index,
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
    )
    ax.set_title("Bridging-the-Gap — Model x Language Accuracy (%)\n"
                 "(Winogrande, averaged across 3 runs)",
                 fontsize=14, fontweight="bold")
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"model_language_heatmap.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved model_language_heatmap.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Bridging-the-Gap African Languages — Visualization")
    print("=" * 70)

    response_matrix, averaged_matrix, model_summary, task_metadata = load_data()
    n_items, n_cols = response_matrix.shape
    print(f"\nLoaded response matrix: {n_items} items x {n_cols} model-language combos")
    print(f"Loaded model summary: {len(model_summary)} rows")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_response_heatmap(response_matrix)
    plot_model_accuracy_by_language(model_summary)
    plot_language_gap(model_summary)
    plot_item_difficulty(response_matrix)
    plot_language_family(model_summary)
    plot_language_correlation(response_matrix)
    plot_model_language_heatmap(model_summary)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
