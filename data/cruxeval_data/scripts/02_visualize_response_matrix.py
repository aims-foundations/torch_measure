"""
Visualize the CRUXEval response matrices.

Produces:
1. Full heatmap (800 samples x 38 configs, binary, sorted by accuracy/difficulty)
2. Task-level heatmap (models x 2 tasks: Input vs Output accuracy)
3. Task difficulty distribution (per-sample solve rate)
4. Model accuracy bar chart
5. Model-model correlation (clustered)
6. Input vs Output comparison (scatter + pass@1 vs pass@5)
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
    """Load all CRUXEval matrices and summary."""
    matrices = {
        "combined": pd.read_csv(f"{DATA_DIR}/response_matrix.csv", index_col=0),
        "combined_binary": pd.read_csv(
            f"{DATA_DIR}/response_matrix_binary.csv", index_col=0
        ),
        "input": pd.read_csv(f"{DATA_DIR}/response_matrix_input.csv", index_col=0),
        "input_binary": pd.read_csv(
            f"{DATA_DIR}/response_matrix_input_binary.csv", index_col=0
        ),
        "output": pd.read_csv(f"{DATA_DIR}/response_matrix_output.csv", index_col=0),
        "output_binary": pd.read_csv(
            f"{DATA_DIR}/response_matrix_output_binary.csv", index_col=0
        ),
    }
    summary = pd.read_csv(f"{DATA_DIR}/model_summary.csv")
    return matrices, summary


def parse_config(config):
    """Parse config like 'codellama-13b+cot_temp0.2_input' into components."""
    parts = config.rsplit("_", 1)
    task_type = parts[1]  # 'input' or 'output'
    model_temp = parts[0]
    temp_parts = model_temp.rsplit("_", 1)
    model_name = temp_parts[0]
    temperature = temp_parts[1]
    return model_name, temperature, task_type


def short_name(config):
    """Shorten config name for display."""
    model, temp, task = parse_config(config)
    t = temp.replace("temp", "T")
    return f"{model} ({t}, {task[0].upper()})"


def plot_full_heatmap(matrix_binary):
    """Full heatmap of 800 samples x 38 configs (binary pass/fail)."""
    df = matrix_binary.copy()
    n_samples, n_configs = df.shape

    # Sort columns by overall accuracy (best left)
    col_acc = df.mean(axis=0).sort_values(ascending=False)
    df = df[col_acc.index]

    # Sort rows by difficulty (easiest top, hardest bottom)
    row_diff = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_diff.index]

    # Group columns by task type for visual separation
    input_cols = [c for c in df.columns if c.endswith("_input")]
    output_cols = [c for c in df.columns if c.endswith("_output")]
    # Sort within each group by accuracy
    input_cols = sorted(input_cols, key=lambda c: col_acc[c], reverse=True)
    output_cols = sorted(output_cols, key=lambda c: col_acc[c], reverse=True)
    ordered_cols = input_cols + output_cols
    df = df[ordered_cols]

    fig, ax = plt.subplots(figsize=(16, 14))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df.values,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Pass (1) / Fail (0)", "shrink": 0.5},
        xticklabels=[short_name(c) for c in ordered_cols],
        yticklabels=False,
    )
    ax.set_ylabel(f"Samples ({n_samples} items, sorted by difficulty)")
    ax.set_xlabel(f"Model Configurations ({n_configs} configs)")
    ax.set_title(
        f"CRUXEval Response Matrix ({n_samples} samples x {n_configs} configs)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xticks(rotation=90, fontsize=7)

    # Add separator between input and output
    sep_pos = len(input_cols)
    ax.axvline(sep_pos, color="blue", linewidth=2, alpha=0.7)
    ax.text(
        sep_pos / 2,
        -15,
        "CRUXEval-I (Input)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="navy",
    )
    ax.text(
        sep_pos + len(output_cols) / 2,
        -15,
        "CRUXEval-O (Output)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="darkred",
    )

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_full.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_full.pdf/png")


def plot_task_heatmap(summary):
    """Task-level heatmap: models x 2 tasks (Input vs Output), using pass@1.

    Analogous to BFCL category-level heatmap.
    """
    # Pivot: one row per model, columns = task (input, output)
    models = summary["model"].unique()
    task_data = []
    for model in sorted(models):
        row = {"model": model}
        sub = summary[summary["model"] == model]
        for _, r in sub.iterrows():
            key = f"{r['task']} ({r['temperature']})"
            row[key] = r["pass_at_1"] / 100.0
        task_data.append(row)

    df = pd.DataFrame(task_data).set_index("model")
    # Sort columns and rows
    df = df.reindex(sorted(df.columns), axis=1)
    row_mean = df.mean(axis=1).sort_values(ascending=False)
    df = df.loc[row_mean.index]

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=0.8,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Pass@1 Rate", "shrink": 0.6},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xlabel("Task (temperature)")
    ax.set_ylabel("Model (sorted by mean accuracy, best top)")
    ax.set_title(
        f"CRUXEval — Per-Task Pass@1 ({len(df)} models x {len(df.columns)} configs)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/heatmap_task.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/heatmap_task.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap_task.pdf/png")


def plot_task_difficulty(matrices):
    """Histogram of per-sample solve rates for Input, Output, and Combined."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels_keys = [
        ("CRUXEval-I (Input)", "input_binary"),
        ("CRUXEval-O (Output)", "output_binary"),
        ("Combined", "combined_binary"),
    ]
    colors = [sns.color_palette("Set2")[i] for i in range(3)]

    for ax, (label, key), color in zip(axes, labels_keys, colors):
        mat = matrices[key]
        item_rates = mat.mean(axis=1)

        bins = np.linspace(0, 1, 31)
        ax.hist(
            item_rates,
            bins=bins,
            alpha=0.8,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        median_rate = np.median(item_rates)
        ax.axvline(
            median_rate,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"median = {median_rate:.2f}",
        )

        unsolved = (item_rates == 0).sum()
        trivial = (item_rates == 1).sum()
        n_samples = len(item_rates)

        ax.set_xlabel("Sample Solve Rate (across models)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({n_samples} samples)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        txt_parts = []
        if unsolved > 0:
            txt_parts.append(f"{unsolved} unsolved")
        if trivial > 0:
            txt_parts.append(f"{trivial} trivial")
        if txt_parts:
            ax.text(
                0.98,
                0.95,
                "\n".join(txt_parts),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
            )

    fig.suptitle(
        "CRUXEval — Sample Difficulty Distribution (binary: any generation correct)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{FIG_DIR}/task_difficulty.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/task_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task_difficulty.pdf/png")


def plot_model_accuracy(summary):
    """Horizontal bar chart of model accuracy (pass@1), grouped by task."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, task, color in zip(
        axes, ["input", "output"], ["#3498db", "#e74c3c"]
    ):
        sub = summary[summary["task"] == task].sort_values(
            "pass_at_1", ascending=True
        )
        n = len(sub)
        ax.barh(range(n), sub["pass_at_1"].values, color=color, alpha=0.85)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f"{r['model']} ({r['temperature']})" for _, r in sub.iterrows()],
            fontsize=8,
        )
        ax.set_xlabel("Pass@1 (%)")
        title = "CRUXEval-I (Input)" if task == "input" else "CRUXEval-O (Output)"
        ax.set_title(f"{title}", fontsize=12, fontweight="bold")

        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(
                r["pass_at_1"] + 0.5, i, f"{r['pass_at_1']:.1f}%",
                va="center", fontsize=7,
            )

    fig.suptitle(
        "CRUXEval — Model Pass@1 by Task", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/model_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_accuracy.pdf/png")


def plot_model_correlation(matrix_binary):
    """Model-model correlation heatmap (all configs, clustered)."""
    # Transpose so rows=models, cols=samples, then correlate
    corr = matrix_binary.corr()
    n = len(corr)

    # Cluster
    link = linkage(corr.values, method="ward")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 5},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.6},
        xticklabels=[short_name(c) for c in corr.columns],
        yticklabels=[short_name(c) for c in corr.index],
    )
    ax.set_title(
        f"CRUXEval — Config-Config Correlation ({n} configs, clustered)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_correlation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_correlation.pdf/png")


def plot_input_vs_output(summary, matrices):
    """Scatter: Input pass@1 vs Output pass@1 per model, plus pass@1 vs pass@5."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Input vs Output pass@1 (per unique model)
    ax = axes[0]
    models_with_both = []
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model]
        inp = sub[sub["task"] == "input"]
        out = sub[sub["task"] == "output"]
        if len(inp) > 0 and len(out) > 0:
            models_with_both.append({
                "model": model,
                "input_p1": inp["pass_at_1"].mean(),
                "output_p1": out["pass_at_1"].mean(),
            })

    if models_with_both:
        df_both = pd.DataFrame(models_with_both)
        ax.scatter(
            df_both["input_p1"],
            df_both["output_p1"],
            alpha=0.7,
            s=60,
            color=sns.color_palette("Set2")[0],
            edgecolor="gray",
        )
        lim = [0, max(df_both["input_p1"].max(), df_both["output_p1"].max()) + 5]
        ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
        ax.set_xlabel("CRUXEval-I Pass@1 (%)")
        ax.set_ylabel("CRUXEval-O Pass@1 (%)")
        ax.set_title("Input vs Output Prediction", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        for _, row in df_both.iterrows():
            ax.annotate(
                row["model"],
                (row["input_p1"], row["output_p1"]),
                fontsize=6,
                alpha=0.8,
                textcoords="offset points",
                xytext=(5, 5),
            )

    # Panel 2: pass@1 vs pass@5
    ax = axes[1]
    ax.scatter(
        summary["pass_at_1"],
        summary["pass_at_5"],
        alpha=0.7,
        s=50,
        c=summary["task"].map({"input": "#3498db", "output": "#e74c3c"}),
        edgecolor="gray",
    )
    lim_p = [0, max(summary["pass_at_5"].max(), summary["pass_at_1"].max()) + 5]
    ax.plot(lim_p, lim_p, "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Pass@1 (%)")
    ax.set_ylabel("Pass@5 (%)")
    ax.set_title("Pass@1 vs Pass@5", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Manual legend for task colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=8, label="Input"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="Output"),
        Line2D([0], [0], color="k", linestyle="--", alpha=0.3, label="y = x"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle(
        "CRUXEval — Input vs Output & Pass@1 vs Pass@5",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/input_vs_output.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{FIG_DIR}/input_vs_output.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved input_vs_output.pdf/png")


def main():
    matrices, summary = load_data()
    print(
        f"Loaded: {matrices['combined'].shape[0]} samples x "
        f"{matrices['combined'].shape[1]} configs"
    )
    print(f"Summary: {len(summary)} model configurations")
    print(f"Generating figures in {FIG_DIR}/\n")

    plot_full_heatmap(matrices["combined_binary"])
    plot_task_heatmap(summary)
    plot_task_difficulty(matrices)
    plot_model_accuracy(summary)
    plot_model_correlation(matrices["combined_binary"])
    plot_input_vs_output(summary, matrices)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
