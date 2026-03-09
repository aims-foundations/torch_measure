"""
Visualize the Aider benchmark results.

Note: Aider provides aggregate per-model metrics, not per-exercise binary data.
The response_matrix.csv is a long-format table of model runs with pass rates.

Produces:
1. Model leaderboard bar chart (pass@1 and pass@2)
2. Edit format comparison (diff vs whole)
3. Cost vs performance scatter
4. Polyglot leaderboard
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
    main = pd.read_csv(f"{DATA_DIR}/response_matrix.csv")
    poly_path = f"{DATA_DIR}/polyglot_response_matrix.csv"
    poly = pd.read_csv(poly_path) if os.path.exists(poly_path) else None
    return main, poly


def plot_leaderboard(main):
    """Bar chart of pass@1 rates for main (edit) benchmark."""
    edit = main[main["benchmark"].str.contains("edit", case=False, na=False)].copy()
    if edit.empty:
        edit = main.copy()
    # Keep best run per model
    best = edit.sort_values("pass_rate_1", ascending=False).drop_duplicates(
        subset=["model"], keep="first"
    )
    best = best.sort_values("pass_rate_1", ascending=True)
    n = len(best)

    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.25)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), best["pass_rate_1"].values, color=colors, alpha=0.85)
    if "pass_rate_2" in best.columns:
        ax.barh(range(n), best["pass_rate_2"].values, color="lightcoral",
                alpha=0.3, label="pass@2")
    ax.set_yticks(range(n))
    ax.set_yticklabels(best["model"].values, fontsize=6)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title(f"Aider Code Editing Benchmark ({n} models)",
                 fontsize=14, fontweight="bold")
    if "pass_rate_2" in best.columns:
        ax.legend(fontsize=9)
    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(r["pass_rate_1"] + 0.3, i, f"{r['pass_rate_1']:.1f}%",
                va="center", fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_cost_vs_performance(main):
    """Scatter: total cost vs pass@1."""
    df = main.dropna(subset=["total_cost", "pass_rate_1"])
    df = df[df["total_cost"] > 0]
    if df.empty:
        print("No cost data, skipping cost_vs_performance")
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["total_cost"], df["pass_rate_1"], alpha=0.6, s=40,
               color=sns.color_palette("Set2")[0], edgecolor="gray")
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["total_cost"], r["pass_rate_1"]),
                    fontsize=5, alpha=0.7, textcoords="offset points",
                    xytext=(3, 3))
    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("Pass@1 Rate (%)")
    ax.set_title("Aider — Cost vs Performance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cost_vs_performance.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/cost_vs_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cost_vs_performance.pdf/png")


def plot_polyglot(poly):
    """Bar chart for polyglot benchmark."""
    if poly is None or poly.empty:
        print("No polyglot data, skipping")
        return
    best = poly.sort_values("pass_rate_1", ascending=False).drop_duplicates(
        subset=["model"], keep="first"
    )
    best = best.sort_values("pass_rate_1", ascending=True)
    n = len(best)

    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.35)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), best["pass_rate_1"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(best["model"].values, fontsize=7)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title(f"Aider Polyglot Benchmark ({n} models)",
                 fontsize=14, fontweight="bold")
    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(r["pass_rate_1"] + 0.3, i, f"{r['pass_rate_1']:.1f}%",
                va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/polyglot_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/polyglot_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved polyglot_accuracy.pdf/png")


def main():
    main_df, poly = load_data()
    print(f"Loaded main: {main_df.shape[0]} rows")
    print(f"Output: {FIG_DIR}\n")
    plot_leaderboard(main_df)
    plot_cost_vs_performance(main_df)
    plot_polyglot(poly)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
