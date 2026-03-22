"""
Visualize the AfriMed-QA response matrix.

Generates:
  1. Model accuracy bar chart
  2. Response matrix heatmap (items x models)
  3. Item difficulty histogram
  4. Specialty accuracy breakdown
  5. Coverage heatmap showing which models evaluated which items
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load processed data files."""
    response_matrix = pd.read_csv(
        os.path.join(PROCESSED_DIR, "response_matrix.csv"), index_col=0
    )
    task_metadata = pd.read_csv(
        os.path.join(PROCESSED_DIR, "task_metadata.csv")
    )
    model_summary = pd.read_csv(
        os.path.join(PROCESSED_DIR, "model_summary.csv")
    )
    return response_matrix, task_metadata, model_summary


def plot_model_accuracy(model_summary):
    """Bar chart of per-model accuracy, sorted descending."""
    fig, ax = plt.subplots(figsize=(12, 8))

    df = model_summary.sort_values("accuracy", ascending=True)
    colors = plt.cm.RdYlGn(df["accuracy"].values)

    bars = ax.barh(range(len(df)), df["accuracy"] * 100, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"], fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("AfriMed-QA: Model Accuracy (MCQ, base-prompt, 0-shot)", fontsize=13)
    ax.set_xlim(0, 100)

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        n_items = int(row["n_items_evaluated"])
        ax.text(row["accuracy"] * 100 + 0.5, i,
                f'{row["accuracy"] * 100:.1f}% ({n_items})',
                va="center", fontsize=8, color="black")

    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50% (chance for 2-opt)")
    ax.axvline(x=25, color="red", linestyle="--", alpha=0.3, label="25% (chance for 4-opt)")

    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_response_heatmap(response_matrix, model_summary):
    """Heatmap of the response matrix (sampled items for readability)."""
    # Sort models by accuracy
    model_order = model_summary.sort_values("accuracy", ascending=False)["model"].tolist()
    model_order = [m for m in model_order if m in response_matrix.columns]

    rm = response_matrix[model_order]

    # Sort items by difficulty (mean accuracy across models)
    item_acc = rm.mean(axis=1)
    rm = rm.loc[item_acc.sort_values(ascending=False).index]

    # Sample items for visualization (max 500 for readability)
    n_items = len(rm)
    if n_items > 500:
        step = n_items // 500
        rm_sampled = rm.iloc[::step]
    else:
        rm_sampled = rm

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create a colormap: NaN=white, 0=red, 1=green
    cmap = mcolors.ListedColormap(["#d32f2f", "#4caf50"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    data = rm_sampled.values.astype(float)
    masked = np.ma.masked_invalid(data)

    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Items (n={len(rm_sampled)}, sorted by difficulty)", fontsize=10)
    ax.set_title("AfriMed-QA Response Matrix (green=correct, red=incorrect, white=missing)",
                 fontsize=11)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.5)
    cbar.ax.set_yticklabels(["Incorrect", "Correct"])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "response_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_item_difficulty(response_matrix):
    """Histogram of item difficulty (fraction of models getting each item correct)."""
    item_acc = response_matrix.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(item_acc * 100, bins=50, color="#1976d2", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Item Accuracy (% of models correct)", fontsize=12)
    ax.set_ylabel("Number of Items", fontsize=12)
    ax.set_title("AfriMed-QA: Item Difficulty Distribution", fontsize=13)

    # Add summary statistics
    median_acc = item_acc.median() * 100
    mean_acc = item_acc.mean() * 100
    ax.axvline(median_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {median_acc:.1f}%")
    ax.axvline(mean_acc, color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_acc:.1f}%")

    n_unsolved = (item_acc == 0).sum()
    n_all_correct = (item_acc == 1).sum()
    ax.text(0.02, 0.95,
            f"Items: {len(item_acc)}\n"
            f"No model correct: {n_unsolved}\n"
            f"All models correct: {n_all_correct}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "item_difficulty.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_specialty_accuracy(response_matrix, task_metadata):
    """Bar chart of accuracy by medical specialty."""
    meta = task_metadata.copy()
    meta = meta[meta["specialty"].notna() & (meta["specialty"] != "") & (meta["specialty"] != "nan")]
    if len(meta) == 0:
        print("  Skipping specialty plot (no specialty data)")
        return

    specialties = meta["specialty"].value_counts()
    # Only plot specialties with >= 20 items
    specialties = specialties[specialties >= 20]

    spec_data = []
    for spec in specialties.index:
        mask = meta["specialty"] == spec
        item_ids = meta.loc[mask, "item_id"].values
        valid_ids = [sid for sid in item_ids if sid in response_matrix.index]
        if len(valid_ids) == 0:
            continue
        acc = response_matrix.loc[valid_ids].mean().mean()
        spec_data.append({
            "specialty": spec.replace("_", " "),
            "accuracy": acc,
            "n_items": len(valid_ids),
        })

    if not spec_data:
        return

    df = pd.DataFrame(spec_data).sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(df["accuracy"].values)

    ax.barh(range(len(df)), df["accuracy"] * 100, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f'{row["specialty"]} (n={row["n_items"]})'
                        for _, row in df.iterrows()], fontsize=9)
    ax.set_xlabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title("AfriMed-QA: Accuracy by Medical Specialty", fontsize=13)
    ax.set_xlim(0, 100)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["accuracy"] * 100 + 0.5, i,
                f'{row["accuracy"] * 100:.1f}%',
                va="center", fontsize=9)

    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "specialty_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_coverage_heatmap(response_matrix, model_summary):
    """Show which models have evaluated which items (coverage pattern)."""
    model_order = model_summary.sort_values("accuracy", ascending=False)["model"].tolist()
    model_order = [m for m in model_order if m in response_matrix.columns]

    rm = response_matrix[model_order]

    # Binary coverage: 1 if evaluated, 0 if missing
    coverage = rm.notna().astype(int)

    # Sort items by number of models that evaluated them
    item_coverage = coverage.sum(axis=1)
    coverage = coverage.loc[item_coverage.sort_values(ascending=False).index]

    # Sample for visualization
    n_items = len(coverage)
    if n_items > 500:
        step = n_items // 500
        coverage_sampled = coverage.iloc[::step]
    else:
        coverage_sampled = coverage

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = mcolors.ListedColormap(["#eeeeee", "#1976d2"])
    im = ax.imshow(coverage_sampled.values, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Items (n={len(coverage_sampled)}, sorted by coverage)", fontsize=10)
    ax.set_title("AfriMed-QA: Evaluation Coverage (blue=evaluated, gray=missing)", fontsize=11)

    # Add per-model coverage as text at bottom
    for i, model in enumerate(model_order):
        n_eval = int(coverage[model].sum())
        pct = n_eval / len(coverage) * 100
        ax.text(i, len(coverage_sampled) + 2, f"{pct:.0f}%",
                ha="center", va="top", fontsize=6, rotation=45)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "coverage_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("AfriMed-QA Response Matrix Visualization")
    print("=" * 60)

    response_matrix, task_metadata, model_summary = load_data()
    print(f"  Response matrix: {response_matrix.shape[0]} items x {response_matrix.shape[1]} models")
    print(f"  Task metadata:   {len(task_metadata)} items")
    print(f"  Model summary:   {len(model_summary)} models")
    print()

    print("Generating figures:")
    plot_model_accuracy(model_summary)
    plot_response_heatmap(response_matrix, model_summary)
    plot_item_difficulty(response_matrix)
    plot_specialty_accuracy(response_matrix, task_metadata)
    plot_coverage_heatmap(response_matrix, model_summary)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    for f in sorted(os.listdir(FIGURES_DIR)):
        fpath = os.path.join(FIGURES_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:40s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
