#!/usr/bin/env python3
"""
Visualize the SIB-200 response matrix.

Produces:
  1. Full heatmap (items x models, sorted by language and difficulty)
  2. Per-language accuracy comparison (GPT-4 vs GPT-3.5)
  3. Per-language accuracy heatmap grouped by script
  4. Category-level accuracy comparison
  5. Language difficulty distribution
  6. Parse rate by language (model ability to process each language)
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BENCHMARK_DIR / "processed"
FIG_DIR = _BENCHMARK_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=0.9)


def load_data():
    """Load all SIB-200 processed data."""
    response_matrix = pd.read_csv(DATA_DIR / "response_matrix.csv", index_col=0)
    task_metadata = pd.read_csv(DATA_DIR / "task_metadata.csv")
    model_summary = pd.read_csv(DATA_DIR / "model_summary.csv")
    return response_matrix, task_metadata, model_summary


def compute_per_language_stats(response_matrix, task_metadata):
    """Compute per-language accuracy and parse rate for each model."""
    item_lang = dict(zip(task_metadata["item_id"], task_metadata["language"]))
    item_script = dict(zip(task_metadata["item_id"], task_metadata["language_script"]))

    records = []
    for model_name in response_matrix.columns:
        col = response_matrix[model_name]
        lang_scores = {}
        lang_parse = {}

        for item_id, score in zip(col.index, col.values):
            lang = item_lang.get(item_id, "unknown")
            lang_scores.setdefault(lang, []).append(score)

        for lang, scores in lang_scores.items():
            valid = [s for s in scores if not np.isnan(s)]
            total = len(scores)
            n_valid = len(valid)
            acc = np.mean(valid) if valid else np.nan
            parse_rate = n_valid / total if total > 0 else 0

            # Get script for this language
            script = ""
            for iid, l in item_lang.items():
                if l == lang:
                    script = item_script.get(iid, "")
                    break

            records.append({
                "model": model_name,
                "language": lang,
                "script": script,
                "accuracy": acc,
                "parse_rate": parse_rate,
                "n_valid": n_valid,
                "n_total": total,
            })

    return pd.DataFrame(records)


def plot_full_heatmap(response_matrix, task_metadata):
    """Full heatmap of items x models, sorted by language then difficulty."""
    print("  Generating full heatmap...")
    df = response_matrix.copy()
    df = df.dropna(axis=1, how="all")

    # Add language info for sorting
    item_lang = dict(zip(task_metadata["item_id"], task_metadata["language"]))
    langs = [item_lang.get(iid, "zzz") for iid in df.index]

    # Sort by language, then by difficulty within language
    df["_lang"] = langs
    df["_diff"] = df[response_matrix.columns].mean(axis=1)
    df = df.sort_values(["_lang", "_diff"], ascending=[True, False])
    df = df.drop(columns=["_lang", "_diff"])

    # Sample for visualization (full 41k rows is too large)
    n_items = len(df)
    if n_items > 5000:
        # Sample 25 items per language
        item_lang_series = pd.Series(langs, index=response_matrix.index)
        sampled_ids = []
        for lang in sorted(item_lang_series.unique()):
            lang_ids = item_lang_series[item_lang_series == lang].index.tolist()
            if len(lang_ids) > 25:
                # Sample evenly across difficulty
                lang_df = df.loc[lang_ids]
                step = max(1, len(lang_ids) // 25)
                sampled_ids.extend(lang_ids[::step][:25])
            else:
                sampled_ids.extend(lang_ids)
        df = df.loc[df.index.isin(sampled_ids)]

    n_q, n_m = df.shape
    fig, ax = plt.subplots(figsize=(6, max(12, n_q * 0.01)))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Correct (1) / Incorrect (0)", "shrink": 0.3},
        xticklabels=list(df.columns), yticklabels=False,
    )
    ax.set_ylabel(f"Items ({n_q} sampled, sorted by language)")
    ax.set_xlabel(f"Models ({n_m})")
    ax.set_title(
        f"SIB-200 Response Matrix (sampled {n_q} of {n_items} items x {n_m} models)",
        fontsize=12, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved heatmap_full.pdf/png")


def plot_language_accuracy_comparison(lang_stats):
    """Bar chart comparing GPT-4 vs GPT-3.5 accuracy per language."""
    print("  Generating per-language accuracy comparison...")

    # Pivot to get languages x models
    pivot = lang_stats.pivot(index="language", columns="model", values="accuracy")
    pivot = pivot.dropna(how="all")

    # Sort by GPT-4 accuracy
    if "gpt-4" in pivot.columns:
        pivot = pivot.sort_values("gpt-4", ascending=True)
    else:
        pivot = pivot.sort_values(pivot.columns[0], ascending=True)

    n_langs = len(pivot)
    fig, ax = plt.subplots(figsize=(12, max(8, n_langs * 0.12)))

    y_pos = np.arange(n_langs)
    width = 0.35
    colors = {"gpt-4": "#2196F3", "gpt-3.5": "#FF9800"}

    for i, model in enumerate(pivot.columns):
        offset = (i - 0.5) * width
        vals = pivot[model].values
        ax.barh(y_pos + offset, vals * 100, width, label=model,
                color=colors.get(model, f"C{i}"), alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=4)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("SIB-200 — Per-Language Accuracy (GPT-4 vs GPT-3.5)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4, label="Chance")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "language_accuracy_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "language_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved language_accuracy_comparison.pdf/png")


def plot_script_accuracy_heatmap(lang_stats):
    """Heatmap of accuracy grouped by writing script."""
    print("  Generating script-grouped accuracy heatmap...")

    # Focus on GPT-4
    gpt4 = lang_stats[lang_stats["model"] == "gpt-4"].copy()
    if gpt4.empty:
        gpt4 = lang_stats[lang_stats["model"] == lang_stats["model"].iloc[0]].copy()

    gpt4 = gpt4.sort_values(["script", "accuracy"], ascending=[True, False])

    # Create a pivot for the heatmap
    scripts = sorted(gpt4["script"].unique())
    script_data = {}
    for script in scripts:
        subset = gpt4[gpt4["script"] == script].sort_values("accuracy", ascending=False)
        for _, row in subset.iterrows():
            script_data[row["language"]] = {
                "script": script,
                "accuracy": row["accuracy"],
                "parse_rate": row["parse_rate"],
            }

    # Build matrix grouped by script
    ordered_langs = []
    for script in scripts:
        script_langs = gpt4[gpt4["script"] == script].sort_values("accuracy", ascending=False)
        ordered_langs.extend(script_langs["language"].tolist())

    acc_values = [script_data[l]["accuracy"] for l in ordered_langs]
    parse_values = [script_data[l]["parse_rate"] for l in ordered_langs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(8, len(ordered_langs) * 0.08)),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Accuracy heatmap
    acc_arr = np.array(acc_values).reshape(-1, 1)
    im1 = ax1.imshow(acc_arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax1.set_yticks(range(len(ordered_langs)))
    ax1.set_yticklabels(ordered_langs, fontsize=3)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Accuracy"])
    ax1.set_title("Accuracy (GPT-4)", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=ax1, shrink=0.3)

    # Parse rate heatmap
    parse_arr = np.array(parse_values).reshape(-1, 1)
    im2 = ax2.imshow(parse_arr, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax2.set_yticks(range(len(ordered_langs)))
    ax2.set_yticklabels(ordered_langs, fontsize=3)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Parse Rate"])
    ax2.set_title("Parse Rate (GPT-4)", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.3)

    # Add script group separators
    prev_script = None
    for i, lang in enumerate(ordered_langs):
        script = script_data[lang]["script"]
        if prev_script is not None and script != prev_script:
            ax1.axhline(i - 0.5, color="black", linewidth=0.5)
            ax2.axhline(i - 0.5, color="black", linewidth=0.5)
        prev_script = script

    fig.suptitle("SIB-200 — Per-Language Accuracy & Parse Rate by Script",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "script_accuracy_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "script_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved script_accuracy_heatmap.pdf/png")


def plot_category_accuracy(response_matrix, task_metadata):
    """Bar chart of per-category accuracy for each model."""
    print("  Generating per-category accuracy chart...")

    item_cat = dict(zip(task_metadata["item_id"], task_metadata["category"]))
    categories = sorted(task_metadata["category"].unique())

    records = []
    for model_name in response_matrix.columns:
        col = response_matrix[model_name]
        for cat in categories:
            cat_items = [iid for iid in col.index if item_cat.get(iid) == cat]
            cat_scores = col.loc[cat_items].dropna()
            acc = cat_scores.mean() if len(cat_scores) > 0 else np.nan
            records.append({
                "model": model_name,
                "category": cat,
                "accuracy": acc,
                "n_items": len(cat_scores),
            })

    cat_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35
    colors = {"gpt-4": "#2196F3", "gpt-3.5": "#FF9800"}

    for i, model in enumerate(response_matrix.columns):
        model_data = cat_df[cat_df["model"] == model]
        vals = [model_data[model_data["category"] == c]["accuracy"].values[0] * 100
                for c in categories]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors.get(model, f"C{i}"), alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SIB-200 — Per-Category Accuracy", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(100 / 7, color="gray", linestyle="--", alpha=0.4, label="Random (14.3%)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "category_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "category_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved category_accuracy.pdf/png")


def plot_difficulty_distribution(lang_stats):
    """Distribution of per-language accuracy (how hard are different languages?)."""
    print("  Generating language difficulty distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, model in enumerate(sorted(lang_stats["model"].unique())):
        ax = axes[i]
        model_data = lang_stats[lang_stats["model"] == model]
        acc_values = model_data["accuracy"].dropna()

        ax.hist(acc_values * 100, bins=30, color="#2196F3" if "4" in model else "#FF9800",
                alpha=0.75, edgecolor="white")
        ax.axvline(acc_values.mean() * 100, color="red", linestyle="--",
                   label=f"Mean: {acc_values.mean()*100:.1f}%")
        ax.axvline(acc_values.median() * 100, color="green", linestyle="--",
                   label=f"Median: {acc_values.median()*100:.1f}%")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Number of Languages")
        ax.set_title(f"{model} — Language Difficulty Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 100)

    fig.suptitle("SIB-200 — Distribution of Per-Language Accuracy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "language_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "language_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved language_difficulty.pdf/png")


def plot_parse_rate_by_language(lang_stats):
    """Parse rate (ability to produce valid prediction) by language for each model."""
    print("  Generating parse rate by language...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, model in enumerate(sorted(lang_stats["model"].unique())):
        ax = axes[i]
        model_data = lang_stats[lang_stats["model"] == model].sort_values("parse_rate")
        parse_values = model_data["parse_rate"]

        ax.hist(parse_values * 100, bins=30,
                color="#2196F3" if "4" in model else "#FF9800",
                alpha=0.75, edgecolor="white")
        ax.axvline(parse_values.mean() * 100, color="red", linestyle="--",
                   label=f"Mean: {parse_values.mean()*100:.1f}%")
        ax.set_xlabel("Parse Rate (%)")
        ax.set_ylabel("Number of Languages")
        ax.set_title(f"{model} — Parse Rate Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 105)

        # Count how many languages have 0% parse rate
        n_zero = (parse_values == 0).sum()
        if n_zero > 0:
            ax.text(0.05, 0.95, f"{n_zero} langs at 0%", transform=ax.transAxes,
                    fontsize=9, va="top", ha="left", color="red")

    fig.suptitle("SIB-200 — Parse Rate Distribution (model ability to process language)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "parse_rate_distribution.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "parse_rate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved parse_rate_distribution.pdf/png")


def main():
    print("SIB-200 Response Matrix Visualizer")
    print("=" * 40)

    response_matrix, task_metadata, model_summary = load_data()
    print(f"Loaded response matrix: {response_matrix.shape}")
    print(f"Loaded task metadata: {task_metadata.shape}")
    print(f"Generating figures in {FIG_DIR}/\n")

    # Compute per-language stats
    lang_stats = compute_per_language_stats(response_matrix, task_metadata)

    # Generate all plots
    plot_full_heatmap(response_matrix, task_metadata)
    plot_language_accuracy_comparison(lang_stats)
    plot_script_accuracy_heatmap(lang_stats)
    plot_category_accuracy(response_matrix, task_metadata)
    plot_difficulty_distribution(lang_stats)
    plot_parse_rate_by_language(lang_stats)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
