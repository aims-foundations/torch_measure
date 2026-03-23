#!/usr/bin/env python3
"""Visualize all intervention / treatment-response datasets.

Generates per-dataset figures in each <benchmark>_data/figures/ directory:
  - heatmap_paired.pdf: side-by-side heatmaps for control vs. treatment conditions
  - subject_effect.pdf: per-subject treatment effect (sorted bar chart)
  - summary_bar.pdf: aggregate accuracy/time by condition

Usage:
    python data/scripts/visualize_intervention.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", font_scale=0.9)

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_matrix(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Per-dataset visualizers
# ---------------------------------------------------------------------------


def viz_paired_heatmaps(
    matrices: dict[str, pd.DataFrame],
    fig_dir: Path,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Side-by-side heatmaps for each condition."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, max(4, min(12, matrices[list(matrices)[0]].shape[0] * 0.15))))

    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, matrices.items()):
        # Sort by subject mean
        subject_order = df.mean(axis=1).sort_values(ascending=False).index
        df_sorted = df.loc[subject_order]

        sns.heatmap(
            df_sorted,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=df_sorted.shape[0] <= 60,
            cbar_kws={"shrink": 0.6},
        )
        n_subj, n_items = df.shape
        mean_val = df.mean().mean()
        ax.set_title(f"{label}\n({n_subj} subj × {n_items} items, mean={mean_val:.3f})", fontsize=10)
        ax.set_ylabel("Subjects" if ax == axes[0] else "")
        ax.set_xlabel("Items")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, fig_dir / "heatmap_paired.pdf")


def viz_subject_effect(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    fig_dir: Path,
    title: str,
    control_label: str = "Control",
    treatment_label: str = "Treatment",
    metric_name: str = "Accuracy",
) -> None:
    """Per-subject bar chart comparing conditions."""
    # Align subjects
    common = control.index.intersection(treatment.index)
    if len(common) == 0:
        # Between-subjects: plot separate distributions
        fig, ax = plt.subplots(figsize=(8, 5))
        ctrl_means = control.mean(axis=1).sort_values()
        treat_means = treatment.mean(axis=1).sort_values()
        ax.hist(ctrl_means, bins=20, alpha=0.6, label=control_label, color="#4575b4")
        ax.hist(treat_means, bins=20, alpha=0.6, label=treatment_label, color="#d73027")
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"{title}\nSubject-level {metric_name} distribution")
        fig.tight_layout()
        save(fig, fig_dir / "subject_effect.pdf")
        return

    ctrl_means = control.loc[common].mean(axis=1)
    treat_means = treatment.loc[common].mean(axis=1)
    effect = treat_means - ctrl_means
    effect_sorted = effect.sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, len(effect_sorted) * 0.25)))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in effect_sorted]
    ax.barh(range(len(effect_sorted)), effect_sorted.values, color=colors)
    if len(effect_sorted) <= 60:
        ax.set_yticks(range(len(effect_sorted)))
        ax.set_yticklabels(effect_sorted.index, fontsize=7)
    else:
        ax.set_yticks([])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Δ {metric_name} ({treatment_label} − {control_label})")
    ax.set_title(f"{title}\nPer-subject treatment effect (mean Δ = {effect.mean():.3f})")
    fig.tight_layout()
    save(fig, fig_dir / "subject_effect.pdf")


def viz_summary_bar(
    condition_means: dict[str, float],
    fig_dir: Path,
    title: str,
    metric_name: str = "Accuracy",
) -> None:
    """Simple bar chart of aggregate means per condition."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(condition_means.keys())
    values = list(condition_means.values())
    colors = sns.color_palette("Set2", len(labels))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    fig.tight_layout()
    save(fig, fig_dir / "summary_bar.pdf")


# ---------------------------------------------------------------------------
# Dataset-specific pipelines
# ---------------------------------------------------------------------------


def viz_collab_cxr() -> None:
    print("\n=== Collab-CXR ===")
    d = BASE_DIR / "collab_cxr_data"
    fig_dir = d / "figures"

    matrices = {}
    for name, fname in [
        ("Image only", "accuracy_matrix_image_only.csv"),
        ("Image + history", "accuracy_matrix_image_history.csv"),
        ("Image + AI", "accuracy_matrix_image_ai.csv"),
        ("Image + AI + history", "accuracy_matrix_image_ai_history.csv"),
    ]:
        m = load_matrix(d / "processed" / fname)
        if m is not None:
            matrices[name] = m

    if not matrices:
        print("  No data found, skipping")
        return

    viz_paired_heatmaps(matrices, fig_dir, "Collab-CXR: Radiologist Diagnostic Accuracy", vmin=0.5, vmax=1.0)
    viz_summary_bar(
        {k: v.mean().mean() for k, v in matrices.items()},
        fig_dir, "Collab-CXR: Mean Accuracy by Condition", "Accuracy",
    )

    # Treatment effect: image_only vs image_ai
    ctrl = matrices.get("Image only")
    treat = matrices.get("Image + AI")
    if ctrl is not None and treat is not None:
        viz_subject_effect(ctrl, treat, fig_dir, "Collab-CXR", "No AI", "With AI", "Accuracy")


def viz_metr(name: str, label: str, ai_file: str, no_ai_file: str) -> None:
    print(f"\n=== {label} ===")
    d = BASE_DIR / f"{name}_data"
    fig_dir = d / "figures"

    ai = load_matrix(d / "processed" / ai_file)
    no_ai = load_matrix(d / "processed" / no_ai_file)

    if ai is None and no_ai is None:
        print("  No data found, skipping")
        return

    matrices = {}
    if no_ai is not None:
        matrices["AI disallowed"] = no_ai
    if ai is not None:
        matrices["AI allowed"] = ai

    viz_paired_heatmaps(matrices, fig_dir, f"{label}: Completion Time (min)", cmap="YlOrRd_r")
    viz_summary_bar(
        {k: v.mean().mean() for k, v in matrices.items()},
        fig_dir, f"{label}: Mean Completion Time by Condition", "Time (min)",
    )

    if ai is not None and no_ai is not None:
        viz_subject_effect(no_ai, ai, fig_dir, label, "No AI", "AI allowed", "Time (min)")


def viz_haiid() -> None:
    print("\n=== HAIID ===")
    d = BASE_DIR / "haiid_data"
    fig_dir = d / "figures"

    domains = ["art", "census", "cities", "dermatology", "sarcasm"]

    # Per-domain paired heatmaps
    for domain in domains:
        matrices = {}
        for stage, label in [("pre", "Pre-advice"), ("post_ai", "Post AI advice"), ("post_human", "Post human advice")]:
            m = load_matrix(d / "processed" / f"response_matrix_{domain}_{stage}.csv")
            if m is not None:
                matrices[label] = m
        if matrices:
            viz_paired_heatmaps(matrices, fig_dir / domain, f"HAIID — {domain.title()}", vmin=0, vmax=1)

    # Cross-domain summary
    summary = {}
    for domain in domains:
        for stage, label in [("pre", "Pre"), ("post_ai", "Post-AI"), ("post_human", "Post-human")]:
            m = load_matrix(d / "processed" / f"response_matrix_{domain}_{stage}.csv")
            if m is not None:
                summary[f"{domain}\n{label}"] = m.mean().mean()

    if summary:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = list(summary.keys())
        values = list(summary.values())
        # Color by stage
        colors = []
        for l in labels:
            if "Pre" in l and "Post" not in l:
                colors.append("#bdbdbd")
            elif "AI" in l:
                colors.append("#4575b4")
            else:
                colors.append("#d73027")
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_ylabel("Accuracy")
        ax.set_title("HAIID: Accuracy by Domain and Condition")
        ax.set_ylim(0.5, 0.85)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#bdbdbd", label="Pre-advice"),
            Patch(color="#4575b4", label="Post AI-labeled advice"),
            Patch(color="#d73027", label="Post human-labeled advice"),
        ], loc="upper right")
        fig.tight_layout()
        save(fig, fig_dir / "summary_bar.pdf")


def viz_genai_learning() -> None:
    print("\n=== GenAI Learning ===")
    d = BASE_DIR / "genai_learning_data"
    fig_dir = d / "figures"

    # Practice phase heatmaps
    practice_matrices = {}
    for arm, label in [("control", "Control"), ("augmented", "GPT Tutor"), ("vanilla", "GPT Base")]:
        m = load_matrix(d / "processed" / f"response_matrix_practice_{arm}.csv")
        if m is not None:
            practice_matrices[label] = m
    if practice_matrices:
        viz_paired_heatmaps(practice_matrices, fig_dir / "practice", "GenAI Learning — Practice Phase (with AI)", vmin=0, vmax=1)

    # Exam phase heatmaps
    exam_matrices = {}
    for arm, label in [("control", "Control"), ("augmented", "GPT Tutor"), ("vanilla", "GPT Base")]:
        m = load_matrix(d / "processed" / f"response_matrix_exam_{arm}.csv")
        if m is not None:
            exam_matrices[label] = m
    if exam_matrices:
        viz_paired_heatmaps(exam_matrices, fig_dir / "exam", "GenAI Learning — Exam Phase (no AI)", vmin=0, vmax=1)

    # The key chart: practice vs exam by condition
    summary = {}
    for phase in ["practice", "exam"]:
        for arm, label in [("control", "Control"), ("augmented", "GPT Tutor"), ("vanilla", "GPT Base")]:
            m = load_matrix(d / "processed" / f"response_matrix_{phase}_{arm}.csv")
            if m is not None:
                summary[(phase, label)] = m.mean().mean()

    if summary:
        fig, ax = plt.subplots(figsize=(8, 5))
        arms = ["Control", "GPT Tutor", "GPT Base"]
        x = np.arange(len(arms))
        width = 0.35

        practice_vals = [summary.get(("practice", a), 0) for a in arms]
        exam_vals = [summary.get(("exam", a), 0) for a in arms]

        bars1 = ax.bar(x - width / 2, practice_vals, width, label="Practice (with AI)", color="#4575b4")
        bars2 = ax.bar(x + width / 2, exam_vals, width, label="Exam (no AI)", color="#d73027")

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_ylabel("Mean Score")
        ax.set_title("GenAI Learning: AI Inflates Practice Scores but Not Exam Scores")
        ax.legend()
        ax.set_ylim(0, 0.85)
        fig.tight_layout()
        save(fig, fig_dir / "practice_vs_exam.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    viz_collab_cxr()
    viz_metr("metr_early2025", "METR Early-2025", "response_matrix_ai.csv", "response_matrix_no_ai.csv")
    viz_metr("metr_late2025", "METR Late-2025", "response_matrix_ai-allowed.csv", "response_matrix_ai-disallowed.csv")
    viz_haiid()
    viz_genai_learning()
    print("\nDone.")
