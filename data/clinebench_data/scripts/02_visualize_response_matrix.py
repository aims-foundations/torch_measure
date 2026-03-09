"""
Visualize the ClineBench results matrix.

ClineBench has 12 tasks evaluated by 2 agents (Terminus, Cline) plus an oracle.
Scores are continuous (0.0-1.0) with some missing values.

Produces:
1. Task scores grouped bar chart (Terminus vs Cline)
2. Difficulty breakdown box plot
3. Agent summary bar chart
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
    df = pd.read_csv(f"{DATA_DIR}/results_matrix.csv")
    return df


def _parse_score(val):
    """Parse score values that may be ranges like '0.75-0.88'."""
    if pd.isna(val) or val == "":
        return np.nan
    s = str(val).strip()
    if "-" in s and not s.startswith("-"):
        parts = s.split("-")
        try:
            return np.mean([float(p) for p in parts])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def plot_task_scores(df):
    """Grouped bar chart: Terminus vs Cline scores per task."""
    df = df.copy()
    df["terminus"] = df["terminus_score"].apply(_parse_score)
    df["cline"] = df["cline_score"].apply(_parse_score)
    df = df.sort_values("short_name")
    n = len(df)

    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.45)))
    x = np.arange(n)
    width = 0.35

    t_vals = df["terminus"].values
    c_vals = df["cline"].values

    ax.barh(x - width / 2, t_vals, width, label="Terminus",
            color="#3498db", alpha=0.85)
    ax.barh(x + width / 2, c_vals, width, label="Cline",
            color="#e74c3c", alpha=0.85)

    ax.set_yticks(x)
    labels = [f"{n} [{d}]" for n, d in
              zip(df["short_name"], df["difficulty"])]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Score (0-1)")
    ax.set_title(f"ClineBench — Task Scores ({n} tasks)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_scores.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_scores.pdf/png")


def plot_difficulty_breakdown(df):
    """Box plot of scores by difficulty level."""
    df = df.copy()
    df["terminus"] = df["terminus_score"].apply(_parse_score)
    df["cline"] = df["cline_score"].apply(_parse_score)

    rows = []
    for _, r in df.iterrows():
        if not np.isnan(r["terminus"]):
            rows.append({"difficulty": r["difficulty"], "agent": "Terminus",
                         "score": r["terminus"]})
        if not np.isnan(r["cline"]):
            rows.append({"difficulty": r["difficulty"], "agent": "Cline",
                         "score": r["cline"]})

    if not rows:
        print("No valid scores for difficulty breakdown, skipping")
        return

    plot_df = pd.DataFrame(rows)
    diff_order = ["easy", "medium", "hard"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="difficulty", y="score", hue="agent",
                order=diff_order, ax=ax, palette=["#3498db", "#e74c3c"],
                showfliers=False)
    sns.stripplot(data=plot_df, x="difficulty", y="score", hue="agent",
                  order=diff_order, ax=ax, palette=["#3498db", "#e74c3c"],
                  dodge=True, size=5, alpha=0.6, jitter=True, legend=False)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Score")
    ax.set_title("ClineBench — Score by Difficulty Level",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/difficulty_breakdown.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/difficulty_breakdown.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved difficulty_breakdown.pdf/png")


def plot_agent_summary(df):
    """Bar chart summarizing overall agent performance."""
    df = df.copy()
    df["terminus"] = df["terminus_score"].apply(_parse_score)
    df["cline"] = df["cline_score"].apply(_parse_score)

    agents = {
        "Terminus": df["terminus"].dropna(),
        "Cline": df["cline"].dropna(),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(agents.keys())
    means = [agents[a].mean() * 100 for a in names]
    counts = [len(agents[a]) for a in names]
    colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(names, means, color=colors, alpha=0.85, width=0.5)
    for bar, m, c in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{m:.1f}%\n(n={c})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Score (%)")
    ax.set_title("ClineBench — Agent Summary",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3 if means else 100)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def main():
    df = load_data()
    print(f"Loaded: {len(df)} tasks")
    print(f"Output: {FIG_DIR}\n")
    plot_task_scores(df)
    plot_difficulty_breakdown(df)
    plot_agent_summary(df)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
