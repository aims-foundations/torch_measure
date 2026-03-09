"""
Visualize Aider Leaderboard aggregate data.

Since Aider only publishes aggregate pass rates (no per-exercise results),
we visualize model-level metrics:
1. Model accuracy bar chart (edit benchmark, sorted by pass_rate_2)
2. Model accuracy bar chart (polyglot benchmark)
3. Pass rate improvement: first attempt vs second attempt
4. Edit format comparison (grouped bar chart)
5. Cost vs performance scatter
6. Error analysis: syntax errors, malformed responses, context exhaustion
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = str(_BENCHMARK_DIR / "processed")
FIGURE_DIR = str(_BENCHMARK_DIR / "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Load data
full_df = pd.read_csv(os.path.join(PROCESSED_DIR, "response_matrix.csv"))

# Split by benchmark
edit_df = full_df[full_df["benchmark"] == "edit_133_python"].copy()
edit_df = edit_df.sort_values("pass_rate_2", ascending=True).reset_index(drop=True)

poly_df = full_df[full_df["benchmark"] == "polyglot_225"].copy()
poly_df = poly_df.sort_values("pass_rate_2", ascending=True).reset_index(drop=True)


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(
            os.path.join(FIGURE_DIR, f"{name}.{ext}"),
            dpi=150, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  Saved {name}")


# --- Figure 1: Edit benchmark model accuracy ---
print("Figure 1: Edit benchmark model accuracy")
fig, ax = plt.subplots(figsize=(10, max(12, len(edit_df) * 0.22)))
y = np.arange(len(edit_df))
bars = ax.barh(y, edit_df["pass_rate_2"], color="#4C72B0", alpha=0.85, height=0.7)
# Mark pass_rate_1 with dots
ax.scatter(edit_df["pass_rate_1"], y, color="#C44E52", s=15, zorder=5,
           label="1st attempt")
ax.set_yticks(y)
ax.set_yticklabels(edit_df["model"], fontsize=6)
ax.set_xlabel("Pass Rate (%)")
ax.set_title("Aider Edit Benchmark (133 Python exercises)\n"
             "Bar = best of 2 attempts, dot = 1st attempt only")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.3)
save_fig(fig, "edit_model_accuracy")


# --- Figure 2: Polyglot benchmark model accuracy ---
print("Figure 2: Polyglot benchmark model accuracy")
fig, ax = plt.subplots(figsize=(10, max(12, len(poly_df) * 0.25)))
y = np.arange(len(poly_df))
bars = ax.barh(y, poly_df["pass_rate_2"], color="#55A868", alpha=0.85, height=0.7)
ax.scatter(poly_df["pass_rate_1"], y, color="#C44E52", s=15, zorder=5,
           label="1st attempt")
ax.set_yticks(y)
ax.set_yticklabels(poly_df["model"], fontsize=6.5)
ax.set_xlabel("Pass Rate (%)")
ax.set_title("Aider Polyglot Benchmark (225 exercises, 6 languages)\n"
             "Bar = best of 2 attempts, dot = 1st attempt only")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.3)
save_fig(fig, "polyglot_model_accuracy")


# --- Figure 3: Pass rate improvement (attempt 1 → attempt 2) ---
print("Figure 3: Pass rate improvement")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df, title in [
    (axes[0], edit_df, "Edit (133 Python)"),
    (axes[1], poly_df, "Polyglot (225 exercises)")
]:
    improvement = df["pass_rate_2"] - df["pass_rate_1"]
    ax.scatter(df["pass_rate_1"], df["pass_rate_2"], alpha=0.6, s=30,
               c=improvement, cmap="RdYlGn", edgecolors="gray", linewidths=0.5)
    # Diagonal line (no improvement)
    lim = [0, max(df["pass_rate_2"].max(), df["pass_rate_1"].max()) + 5]
    ax.plot(lim, lim, "k--", alpha=0.3, label="No improvement")
    ax.set_xlabel("Pass Rate: 1st Attempt (%)")
    ax.set_ylabel("Pass Rate: Best of 2 Attempts (%)")
    ax.set_title(title)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    mean_imp = improvement.mean()
    ax.text(0.05, 0.92, f"Mean improvement: +{mean_imp:.1f}pp",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.suptitle("Second Attempt Improvement in Aider Benchmarks", fontsize=13)
fig.tight_layout()
save_fig(fig, "pass_rate_improvement")


# --- Figure 4: Edit format comparison ---
print("Figure 4: Edit format comparison")
# For edit benchmark, compare edit formats
format_stats = edit_df.groupby("edit_format").agg(
    n_models=("model", "count"),
    mean_pass1=("pass_rate_1", "mean"),
    mean_pass2=("pass_rate_2", "mean"),
    best_pass2=("pass_rate_2", "max"),
    mean_wellformed=("percent_cases_well_formed", "mean"),
).sort_values("mean_pass2", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: pass rates by format
x = np.arange(len(format_stats))
w = 0.35
axes[0].bar(x - w/2, format_stats["mean_pass1"], w, label="1st attempt",
            color="#C44E52", alpha=0.8)
axes[0].bar(x + w/2, format_stats["mean_pass2"], w, label="Best of 2",
            color="#4C72B0", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(format_stats.index, fontsize=9)
axes[0].set_ylabel("Mean Pass Rate (%)")
axes[0].set_title("Pass Rates by Edit Format")
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.3)
# Add count annotations
for i, n in enumerate(format_stats["n_models"]):
    axes[0].text(i, 2, f"n={n}", ha="center", fontsize=8, color="gray")

# Right: well-formedness by format
axes[1].bar(x, format_stats["mean_wellformed"], color="#55A868", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(format_stats.index, fontsize=9)
axes[1].set_ylabel("Mean % Well-Formed Responses")
axes[1].set_title("Response Well-Formedness by Edit Format")
axes[1].set_ylim(0, 105)
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Edit Format Comparison (Edit Benchmark)", fontsize=13)
fig.tight_layout()
save_fig(fig, "edit_format_comparison")


# --- Figure 5: Cost vs Performance ---
print("Figure 5: Cost vs Performance")
# Filter to models with cost data
cost_df = full_df[
    (full_df["total_cost"] > 0) &
    (full_df["benchmark"].isin(["edit_133_python", "polyglot_225"]))
].copy()

if len(cost_df) > 5:
    fig, ax = plt.subplots(figsize=(10, 7))
    for bench, marker, color in [
        ("edit_133_python", "o", "#4C72B0"),
        ("polyglot_225", "s", "#55A868")
    ]:
        sub = cost_df[cost_df["benchmark"] == bench]
        if len(sub) == 0:
            continue
        ax.scatter(sub["total_cost"], sub["pass_rate_2"],
                   marker=marker, color=color, alpha=0.6, s=40,
                   label=bench.replace("_", " "), edgecolors="gray",
                   linewidths=0.5)
        # Label top-5 by pass_rate_2
        top5 = sub.nlargest(5, "pass_rate_2")
        for _, row in top5.iterrows():
            ax.annotate(row["model"], (row["total_cost"], row["pass_rate_2"]),
                        fontsize=6, alpha=0.7,
                        xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Cost vs Performance in Aider Benchmarks")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if cost_df["total_cost"].max() > 50:
        ax.set_xscale("log")
        ax.set_xlabel("Total Cost ($, log scale)")
    save_fig(fig, "cost_vs_performance")
else:
    print("  Skipped (insufficient cost data)")


# --- Figure 6: Error analysis ---
print("Figure 6: Error analysis")
error_cols = ["syntax_errors", "indentation_errors",
              "exhausted_context_windows", "test_timeouts"]
# Use edit benchmark for cleaner comparison
err_df = edit_df[["model"] + error_cols + ["pass_rate_2"]].copy()
err_df = err_df.sort_values("pass_rate_2", ascending=True).reset_index(drop=True)

# Only keep models with at least 1 error for readability
err_df["total_errors"] = err_df[error_cols].sum(axis=1)
err_nonzero = err_df[err_df["total_errors"] > 0].reset_index(drop=True)

if len(err_nonzero) > 3:
    fig, ax = plt.subplots(figsize=(10, max(8, len(err_nonzero) * 0.22)))
    y = np.arange(len(err_nonzero))
    colors = ["#C44E52", "#DD8452", "#4C72B0", "#937860"]
    left = np.zeros(len(err_nonzero))
    for col, color in zip(error_cols, colors):
        vals = err_nonzero[col].fillna(0).values
        ax.barh(y, vals, left=left, color=color, alpha=0.85, height=0.7,
                label=col.replace("_", " ").title())
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(err_nonzero["model"], fontsize=6)
    ax.set_xlabel("Number of Errors")
    ax.set_title("Error Types per Model (Edit Benchmark, models with errors)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "error_analysis")
else:
    print("  Skipped (few models with errors)")


# --- Figure 7: Timeline of benchmark performance ---
print("Figure 7: Performance over time")
# Convert date to datetime
time_df = full_df[full_df["benchmark"].isin(
    ["edit_133_python", "polyglot_225"])].copy()
time_df["date"] = pd.to_datetime(time_df["date"], errors="coerce")
time_df = time_df.dropna(subset=["date"])

if len(time_df) > 5:
    fig, ax = plt.subplots(figsize=(12, 6))
    for bench, marker, color in [
        ("edit_133_python", "o", "#4C72B0"),
        ("polyglot_225", "s", "#55A868")
    ]:
        sub = time_df[time_df["benchmark"] == bench].sort_values("date")
        ax.scatter(sub["date"], sub["pass_rate_2"], marker=marker,
                   color=color, alpha=0.5, s=30,
                   label=bench.replace("_", " "))
        # Rolling best (frontier)
        running_best = sub["pass_rate_2"].cummax()
        ax.step(sub["date"], running_best, color=color, alpha=0.7,
                linewidth=1.5, where="post")

    ax.set_xlabel("Model Release / Evaluation Date")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Aider Benchmark Performance Over Time\n"
                 "(line = running best)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    save_fig(fig, "performance_timeline")
else:
    print("  Skipped (insufficient date data)")


# --- Figure 8: Wellformedness vs Pass Rate ---
print("Figure 8: Wellformedness vs Pass Rate")
wf_df = full_df[full_df["benchmark"].isin(
    ["edit_133_python", "polyglot_225"])].copy()

fig, ax = plt.subplots(figsize=(9, 6))
for bench, marker, color in [
    ("edit_133_python", "o", "#4C72B0"),
    ("polyglot_225", "s", "#55A868")
]:
    sub = wf_df[wf_df["benchmark"] == bench]
    ax.scatter(sub["percent_cases_well_formed"], sub["pass_rate_2"],
               marker=marker, color=color, alpha=0.5, s=30,
               label=bench.replace("_", " "), edgecolors="gray",
               linewidths=0.3)

ax.set_xlabel("% Well-Formed Responses")
ax.set_ylabel("Pass Rate (%)")
ax.set_title("Response Well-Formedness vs Pass Rate")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
save_fig(fig, "wellformedness_vs_passrate")


print(f"\nDone. All figures saved to {FIGURE_DIR}/")
