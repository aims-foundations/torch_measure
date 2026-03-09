"""
Visualize the ScienceAgentBench results.

The per-task response_matrix has many NAs, so we primarily use
aggregate_results.csv which has success rates, execution rates,
CodeBERT scores, and costs per model-framework-knowledge combination.

Produces:
1. Full heatmap (102 tasks x model configs, from response_matrix)
2. Aggregate leaderboard bar chart (success rate)
3. Framework comparison (Direct vs OpenHands vs SelfDebug)
4. Cost vs performance scatter
5. Domain breakdown (from task metadata)
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
    agg = pd.read_csv(f"{DATA_DIR}/aggregate_results.csv")
    matrix = pd.read_csv(f"{DATA_DIR}/response_matrix.csv")
    meta = pd.read_csv(f"{DATA_DIR}/task_metadata.csv")
    return agg, matrix, meta


def plot_full_heatmap(matrix):
    """Heatmap of per-task results (102 tasks x model configs)."""
    df = matrix.set_index("instance_id")
    meta_cols = ["domain", "subtask_categories"]
    num_cols = [c for c in df.columns if c not in meta_cols]
    df_num = df[num_cols].replace("NA", np.nan).astype(float)
    df_num = df_num.dropna(axis=1, how="all")

    if df_num.empty or df_num.shape[1] < 2:
        print("Insufficient per-task data for heatmap, skipping")
        return

    n_t, n_m = df_num.shape
    row_diff = df_num.mean(axis=1).sort_values(ascending=False)
    df_num = df_num.loc[row_diff.index]
    col_acc = df_num.mean(axis=0).sort_values(ascending=False)
    df_num = df_num[col_acc.index]

    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(df_num.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={"label": "Success (1) / Fail (0)", "shrink": 0.4},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel(f"Tasks ({n_t})")
    ax.set_xlabel(f"Model Configs ({n_m})")
    ax.set_title(f"ScienceAgentBench ({n_t} tasks x {n_m} configs)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=4)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_leaderboard(agg):
    """Bar chart of success rates for all model configs."""
    df = agg.sort_values("success_rate_pct", ascending=True).copy()
    n = len(df)
    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.3)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), df["success_rate_pct"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["model_config"].values, fontsize=5)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title(f"ScienceAgentBench — Success Rate ({n} configs)",
                 fontsize=14, fontweight="bold")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["success_rate_pct"] + 0.3, i,
                f"{r['success_rate_pct']:.1f}%", va="center", fontsize=5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_framework_comparison(agg):
    """Grouped bar: success rate by framework for paper-sourced results."""
    paper = agg[agg["source"] == "paper"].copy()
    if paper.empty:
        print("No paper-sourced results, skipping framework comparison")
        return

    frameworks = paper["framework"].unique()
    models = paper["model"].unique()
    knowledge_opts = paper["knowledge"].unique()

    fig, axes = plt.subplots(1, len(knowledge_opts),
                             figsize=(7 * len(knowledge_opts), 6),
                             sharey=True)
    if len(knowledge_opts) == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", len(frameworks))
    fw_colors = dict(zip(frameworks, palette))

    for ax, know in zip(axes, sorted(knowledge_opts)):
        subset = paper[paper["knowledge"] == know]
        for fw in sorted(frameworks):
            fw_data = subset[subset["framework"] == fw].set_index("model")
            fw_models = [m for m in sorted(models) if m in fw_data.index]
            vals = [fw_data.loc[m, "success_rate_pct"] for m in fw_models]
            x = np.arange(len(fw_models))
            ax.bar(x, vals, width=0.25, label=fw, color=fw_colors[fw],
                   alpha=0.85)
        ax.set_xticks(np.arange(len(sorted(models))))
        ax.set_xticklabels(sorted(models), rotation=45, ha="right",
                           fontsize=7)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Knowledge: {know}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("ScienceAgentBench — Framework Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{FIG_DIR}/framework_comparison.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/framework_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved framework_comparison.pdf/png")


def plot_cost_vs_performance(agg):
    """Scatter: cost vs success rate."""
    df = agg.dropna(subset=["cost_usd", "success_rate_pct"])
    df = df[df["cost_usd"] > 0]
    if df.empty:
        print("No cost data, skipping cost_vs_performance")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    sources = df["source"].unique()
    palette = sns.color_palette("Set1", len(sources))
    for src, color in zip(sorted(sources), palette):
        sub = df[df["source"] == src]
        ax.scatter(sub["cost_usd"], sub["success_rate_pct"], alpha=0.6, s=50,
                   color=color, edgecolor="gray", label=src)
        for _, r in sub.iterrows():
            label = r["model"] if len(r["model"]) < 25 else r["model"][:22] + "..."
            ax.annotate(label, (r["cost_usd"], r["success_rate_pct"]),
                        fontsize=4, alpha=0.7, textcoords="offset points",
                        xytext=(3, 3))
    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("ScienceAgentBench — Cost vs Performance",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cost_vs_performance.pdf", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/cost_vs_performance.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved cost_vs_performance.pdf/png")


def plot_domain_breakdown(matrix, meta):
    """Per-domain success rates."""
    df = matrix.set_index("instance_id")
    meta_cols = ["domain", "subtask_categories"]
    num_cols = [c for c in df.columns if c not in meta_cols]
    df_num = df[num_cols].replace("NA", np.nan).astype(float)

    dom_map = dict(zip(meta["instance_id"].astype(str),
                       meta["domain"]))
    domains = sorted(set(dom_map.values()))

    dom_rates = {}
    for dom in domains:
        mask = [dom_map.get(str(t)) == dom for t in df_num.index]
        if any(mask):
            dom_rates[dom] = df_num.loc[mask].mean(axis=1).dropna()

    if not dom_rates:
        print("No domain data, skipping domain breakdown")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(dom_rates))
    bins = np.linspace(0, 1, 11)
    for (dom, vals), color in zip(sorted(dom_rates.items()), colors):
        ax.hist(vals.values, bins=bins, alpha=0.7,
                label=f"{dom} (n={len(vals)})",
                color=color, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Task Success Rate (across model configs)")
    ax.set_ylabel("Count")
    ax.set_title("ScienceAgentBench — Task Difficulty by Domain",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def main():
    agg, matrix, meta = load_data()
    print(f"Loaded: {len(agg)} aggregate entries, "
          f"{matrix.shape[0]} tasks in response matrix")
    print(f"Output: {FIG_DIR}\n")
    plot_full_heatmap(matrix)
    plot_leaderboard(agg)
    plot_framework_comparison(agg)
    plot_cost_vs_performance(agg)
    plot_domain_breakdown(matrix, meta)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
