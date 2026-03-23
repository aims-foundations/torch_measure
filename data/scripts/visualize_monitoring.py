#!/usr/bin/env python3
"""Visualize all monitoring / post-deployment datasets.

Generates per-dataset heatmap and summary figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", font_scale=0.9)
BASE_DIR = Path(__file__).resolve().parent.parent


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def viz_nhtsa():
    print("\n=== NHTSA SGO ===")
    d = BASE_DIR / "nhtsa_sgo_data"
    fig_dir = d / "figures"

    for name in ["make_x_crash_type_combined.csv", "make_x_injury_severity.csv"]:
        path = d / "processed" / name
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        # Keep top 15 manufacturers by total
        df = df.loc[df.sum(axis=1).nlargest(15).index]
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 0.6), max(4, df.shape[0] * 0.35)))
        sns.heatmap(df, annot=True, fmt="g", cmap="YlOrRd", ax=ax, linewidths=0.5)
        title = name.replace(".csv", "").replace("_", " ").title()
        ax.set_title(f"NHTSA SGO: {title}")
        fig.tight_layout()
        save(fig, fig_dir / name.replace(".csv", ".pdf"))


def viz_ca_dmv():
    print("\n=== CA DMV Disengagements ===")
    d = BASE_DIR / "ca_dmv_disengagement_data"
    fig_dir = d / "figures"

    for name in ["manufacturer_x_initiator.csv", "manufacturer_x_location.csv"]:
        path = d / "processed" / name
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        df = df.loc[df.sum(axis=1).nlargest(15).index]
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 0.8), max(4, df.shape[0] * 0.35)))
        sns.heatmap(df, annot=True, fmt="g", cmap="YlOrRd", ax=ax, linewidths=0.5)
        title = name.replace(".csv", "").replace("_", " ").title()
        ax.set_title(f"CA DMV: {title}")
        fig.tight_layout()
        save(fig, fig_dir / name.replace(".csv", ".pdf"))


def viz_aegis():
    print("\n=== NVIDIA Aegis 2.0 ===")
    d = BASE_DIR / "aegis_data"
    fig_dir = d / "figures"

    # Category co-occurrence
    path = d / "processed" / "category_cooccurrence.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
        sns.heatmap(df, mask=mask, annot=True, fmt="g", cmap="Blues", ax=ax, linewidths=0.5)
        ax.set_title("Aegis 2.0: Hazard Category Co-occurrence")
        fig.tight_layout()
        save(fig, fig_dir / "category_cooccurrence.pdf")

    # Response matrix heatmap (sample)
    path = d / "processed" / "response_matrix.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        # Sample 200 rows for readability
        if len(df) > 200:
            df = df.sample(200, random_state=42)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(df, cmap="RdYlGn", ax=ax, xticklabels=True, yticklabels=False, cbar_kws={"shrink": 0.6})
        ax.set_title(f"Aegis 2.0: Sample × Hazard Category (200/{len(df)} samples)")
        ax.set_xlabel("Hazard Category")
        ax.set_ylabel("Samples")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        save(fig, fig_dir / "response_matrix_sample.pdf")


def viz_chatgpt_drift():
    print("\n=== ChatGPT Drift ===")
    d = BASE_DIR / "chatgpt_drift_data"
    fig_dir = d / "figures"

    path = d / "processed" / "model_x_task_accuracy.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)

    fig, ax = plt.subplots(figsize=(10, max(4, df.shape[0] * 0.4)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("ChatGPT Drift: Model × Task Accuracy")
    fig.tight_layout()
    save(fig, fig_dir / "model_x_task_accuracy.pdf")

    # Drift plot (version comparison)
    drift_path = d / "processed" / "drift_by_version_task.csv"
    if drift_path.exists():
        drift = pd.read_csv(drift_path)
        if "accuracy_change" in drift.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            tasks = drift["task"].unique()
            x = np.arange(len(tasks))
            changes = [drift[drift["task"] == t]["accuracy_change"].values[0] if t in drift["task"].values else 0 for t in tasks]
            colors = ["#d73027" if c < 0 else "#4575b4" for c in changes]
            ax.bar(x, changes, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Accuracy Change")
            ax.set_title("ChatGPT Drift: Accuracy Change Between API Versions")
            fig.tight_layout()
            save(fig, fig_dir / "drift_by_task.pdf")


def viz_toxicchat():
    print("\n=== LMSYS ToxicChat ===")
    d = BASE_DIR / "lmsys_toxicchat_data"
    fig_dir = d / "figures"

    # Moderation category means
    path = d / "processed" / "moderation_category_means.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        means = df.mean().sort_values(ascending=False)
        ax.bar(range(len(means)), means.values, color=sns.color_palette("Reds_r", len(means)))
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Moderation Score")
        ax.set_title("ToxicChat: OpenAI Moderation Category Scores")
        fig.tight_layout()
        save(fig, fig_dir / "moderation_categories.pdf")


def viz_aiid():
    print("\n=== AIID ===")
    d = BASE_DIR / "aiid_data"
    fig_dir = d / "figures"

    path = d / "processed" / "incidents_by_year.csv"
    if not path.exists():
        return
    by_year = pd.read_csv(path, index_col=0).squeeze()
    by_year = by_year[by_year.index != "year"]
    by_year.index = by_year.index.astype(float).astype(int)
    by_year = by_year.sort_index()
    # Filter to 2015+
    by_year = by_year[by_year.index >= 2015]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_year.index, by_year.values, color=sns.color_palette("Blues_d", len(by_year)))
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Incidents")
    ax.set_title("AI Incident Database: Reported Incidents by Year")
    fig.tight_layout()
    save(fig, fig_dir / "incidents_by_year.pdf")


def viz_wildchat():
    print("\n=== WildChat ===")
    d = BASE_DIR / "wildchat_data"
    fig_dir = d / "figures"

    path = d / "processed" / "language_distribution.csv"
    if not path.exists():
        return
    langs = pd.read_csv(path, index_col=0, header=None).squeeze()
    top = langs.head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), top.values, color=sns.color_palette("viridis", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Conversations")
    ax.set_title("WildChat: Top 15 Languages (100K sample)")
    fig.tight_layout()
    save(fig, fig_dir / "language_distribution.pdf")


if __name__ == "__main__":
    viz_nhtsa()
    viz_ca_dmv()
    viz_aegis()
    viz_chatgpt_drift()
    viz_toxicchat()
    viz_aiid()
    viz_wildchat()
    print("\nDone.")
