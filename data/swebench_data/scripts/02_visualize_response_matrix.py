"""
Visualize the SWE-bench Verified response matrix.

Produces:
1. Full heatmap (134 models x 500 instances, sorted by resolve rate/difficulty)
2. Repo-level heatmap (models x repos, analogous to category-level)
3. Task difficulty distribution by repo
4. Model accuracy bar chart
5. Repo breakdown box plot
6. Repo-repo correlation (clustered)
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


def load_data():
    matrix = pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0)
    summary = pd.read_csv(f"{DATA_DIR}/model_summary.csv")
    return matrix, summary


def extract_repo(instance_id):
    """Extract repo name from instance ID like 'django__django-12345'."""
    parts = instance_id.split("__")
    return parts[0] if len(parts) >= 2 else instance_id


def plot_full_heatmap(matrix):
    """Full heatmap of 134 models x 500 instances."""
    df = matrix.copy()
    n_m, n_t = df.shape

    # Sort rows by resolve rate (best top)
    row_acc = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_acc.index]

    # Sort columns: first by repo, then by difficulty within repo
    repos = [extract_repo(c) for c in df.columns]
    col_diff = df.mean(axis=0)
    col_order = sorted(
        range(len(df.columns)),
        key=lambda i: (repos[i], col_diff.iloc[i])
    )
    df = df.iloc[:, col_order]
    sorted_repos = [repos[i] for i in col_order]

    fig, ax = plt.subplots(figsize=(24, 18))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values, ax=ax, cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": "Resolved (1) / Not Resolved (0)", "shrink": 0.4},
        xticklabels=False, yticklabels=True,
    )
    ax.set_ylabel("Model (sorted by resolve rate, best top)")
    ax.set_xlabel(f"Instances ({n_t} items, grouped by repository)")
    ax.set_title(
        f"SWE-bench Verified Response Matrix ({n_m} models x {n_t} instances)",
        fontsize=16, fontweight="bold",
    )
    plt.yticks(fontsize=4)

    # Add repo separators
    unique_repos = []
    repo_positions = []
    prev_repo = None
    for i, repo in enumerate(sorted_repos):
        if repo != prev_repo:
            if prev_repo is not None:
                ax.axvline(i, color="black", linewidth=0.5, alpha=0.7)
            repo_positions.append(i)
            unique_repos.append(repo)
            prev_repo = repo

    # Add repo labels at bottom
    for i, (repo, pos) in enumerate(zip(unique_repos, repo_positions)):
        next_pos = (repo_positions[i + 1]
                    if i + 1 < len(repo_positions) else len(sorted_repos))
        mid = (pos + next_pos) / 2
        if (next_pos - pos) > 5:
            ax.text(mid, n_m + 1, repo, ha="center", va="top",
                    fontsize=6, rotation=90)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_repo_heatmap(matrix):
    """Repo-level heatmap: models x repos (mean resolve rate per repo).

    Analogous to BFCL category-level heatmap.
    """
    repos = [extract_repo(c) for c in matrix.columns]
    unique_repos = sorted(set(repos))

    repo_matrix = pd.DataFrame(
        index=matrix.index, columns=unique_repos, dtype=float
    )
    for repo in unique_repos:
        cols = [c for c, r in zip(matrix.columns, repos) if r == repo]
        repo_matrix[repo] = matrix[cols].mean(axis=1)

    # Sort rows by overall resolve rate, columns by difficulty
    row_acc = repo_matrix.mean(axis=1).sort_values(ascending=False)
    repo_matrix = repo_matrix.loc[row_acc.index]
    col_diff = repo_matrix.mean(axis=0).sort_values()
    repo_matrix = repo_matrix[col_diff.index]

    # Take top 40 models for readability
    repo_top = repo_matrix.head(40)

    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        repo_top, ax=ax, cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 5},
        cbar_kws={"label": "Resolve Rate", "shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("Repository (sorted by difficulty, hardest left)")
    ax.set_ylabel("Model (sorted by resolve rate, best top)")
    ax.set_title(
        f"SWE-bench Verified — Per-Repo Resolve Rate "
        f"(top 40 of {len(repo_matrix)} models x {len(unique_repos)} repos)",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_repo.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_repo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_repo.pdf/png")

    return repo_matrix


def plot_task_difficulty(matrix):
    """Histogram of per-instance resolve rates, colored by repo."""
    repos = [extract_repo(c) for c in matrix.columns]
    item_rates = matrix.mean(axis=0)

    cat_rates = {}
    for col, repo in zip(matrix.columns, repos):
        cat_rates.setdefault(repo, []).append(item_rates[col])

    cat_order = sorted(cat_rates.keys(),
                       key=lambda c: np.median(cat_rates[c]))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab20", len(cat_order))
    cat_colors = dict(zip(cat_order, colors))

    bins = np.linspace(0, 1, 31)
    for repo in reversed(cat_order):
        ax.hist(cat_rates[repo], bins=bins, alpha=0.7, label=repo,
                color=cat_colors[repo], edgecolor="white", linewidth=0.3)

    ax.set_xlabel(f"Instance Resolve Rate (across {matrix.shape[0]} models)")
    ax.set_ylabel("Count")
    ax.set_title("SWE-bench Verified — Instance Difficulty Distribution by Repo",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(summary):
    """Horizontal bar chart of model resolve rates."""
    df = summary.sort_values("resolve_rate", ascending=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(10, max(16, n * 0.16)))
    colors = sns.color_palette("viridis", n)
    ax.barh(range(n), df["resolve_rate"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["model"].values, fontsize=4)
    ax.set_xlabel("Resolve Rate (%)")
    ax.set_title(
        f"SWE-bench Verified — Model Resolve Rates "
        f"({n} models, 500 instances)",
        fontsize=14, fontweight="bold",
    )

    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["resolve_rate"] + 0.3, i, f"{r['resolve_rate']:.1f}%",
                va="center", fontsize=3.5)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_repo_breakdown(matrix):
    """Box plot of per-instance resolve rates grouped by repository."""
    repos = [extract_repo(c) for c in matrix.columns]
    item_rates = matrix.mean(axis=0)

    df = pd.DataFrame({
        "instance_id": matrix.columns,
        "repo": repos,
        "resolve_rate": item_rates.values,
    })

    cat_order = (df.groupby("repo")["resolve_rate"]
                 .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x="repo", y="resolve_rate", order=cat_order,
                ax=ax, palette="Set2", showfliers=False)
    sns.stripplot(data=df, x="repo", y="resolve_rate", order=cat_order,
                  ax=ax, color="black", size=1.5, alpha=0.3, jitter=True)
    ax.set_ylabel("Instance Resolve Rate")
    ax.set_xlabel("")
    ax.set_title("SWE-bench Verified — Instance Resolve Rate by Repository",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    for i, repo in enumerate(cat_order):
        n_items = len(df[df["repo"] == repo])
        ax.text(i, -0.03, f"n={n_items}", ha="center", va="top",
                fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/repo_breakdown.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/repo_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved repo_breakdown.pdf/png")


def plot_repo_correlation(repo_matrix):
    """Repo-repo correlation heatmap (clustered)."""
    corr = repo_matrix.corr()

    link = linkage(corr.values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-0.3, vmax=1,
        mask=mask, square=True,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title("SWE-bench Verified — Repo-Repo Correlation (clustered)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/repo_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/repo_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved repo_correlation.pdf/png")


def main():
    matrix, summary = load_data()
    print(f"Loaded: {matrix.shape[0]} models x {matrix.shape[1]} instances")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_full_heatmap(matrix)
    repo_matrix = plot_repo_heatmap(matrix)
    plot_task_difficulty(matrix)
    plot_model_accuracy(summary)
    plot_repo_breakdown(matrix)
    plot_repo_correlation(repo_matrix)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
