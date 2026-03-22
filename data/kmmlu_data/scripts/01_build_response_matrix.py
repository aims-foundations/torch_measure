#!/usr/bin/env python3
"""
Build task metadata and item content for KMMLU (Korean MMLU) from HuggingFace.

Data source:
  HAERAE-HUB/KMMLU on HuggingFace
  - 45 subject configs, each with train/dev/test splits
  - Columns: question, answer (int 0-3 mapping to A-D), A, B, C, D,
             Category, Human Accuracy
  - Human Accuracy is a per-item float (0.0 to 1.0)

Since there are no per-model prediction files, we cannot build a full
response matrix.  Instead we produce:
  - task_metadata.csv : item_id, question (first 200 chars), answer_key,
                        category, human_accuracy, split, source_dataset
  - item_content.csv  : item_id, full question + options text
  - model_summary.csv : placeholder with human-baseline row
  - response_matrix.csv : single-column "human_accuracy" pseudo-matrix

All paths are relative to the benchmark directory using
    Path(__file__).resolve().parent.parent
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# KMMLU configs (45 subjects)
# ──────────────────────────────────────────────────────────────────────
KMMLU_CONFIGS = [
    "Accounting",
    "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance",
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Construction",
    "Criminal-Law",
    "Ecology",
    "Economics",
    "Education",
    "Electrical-Engineering",
    "Electronics-Engineering",
    "Energy-Management",
    "Environmental-Science",
    "Fashion",
    "Food-Processing",
    "Gas-Technology-and-Engineering",
    "Geomatics",
    "Health",
    "Industrial-Engineer",
    "Information-Technology",
    "Interior-Architecture-and-Design",
    "Korean-History",
    "Law",
    "Machine-Design-and-Manufacturing",
    "Management",
    "Maritime-Engineering",
    "Marketing",
    "Materials-Engineering",
    "Mechanical-Engineering",
    "Nondestructive-Testing",
    "Patent",
    "Political-Science-and-Sociology",
    "Psychology",
    "Public-Safety",
    "Railway-and-Automotive-Engineering",
    "Real-Estate",
    "Refrigerating-Machinery",
    "Social-Welfare",
    "Taxation",
    "Telecommunications-and-Wireless-Technology",
    "Math",
]

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_kmmlu():
    """Load all KMMLU configs from HuggingFace and return combined rows."""
    from datasets import load_dataset

    all_rows = []
    item_id = 0

    for cfg in KMMLU_CONFIGS:
        print(f"  Loading {cfg}...", end=" ")
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", cfg)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        for split_name in ["test", "dev", "train"]:
            if split_name not in ds:
                continue
            split_ds = ds[split_name]
            for row in split_ds:
                q = row["question"]
                a_key = ANSWER_MAP.get(row["answer"], str(row["answer"]))
                options_text = (
                    f"A: {row['A']}\nB: {row['B']}\n"
                    f"C: {row['C']}\nD: {row['D']}"
                )
                full_content = f"{q}\n\n{options_text}"
                human_acc = row.get("Human Accuracy", np.nan)
                if human_acc is None:
                    human_acc = np.nan

                all_rows.append({
                    "item_id": f"kmmlu_{item_id:06d}",
                    "question_short": q[:200],
                    "answer_key": a_key,
                    "category": row.get("Category", cfg),
                    "config": cfg,
                    "human_accuracy": human_acc,
                    "split": split_name,
                    "source_dataset": "HAERAE-HUB/KMMLU",
                    "language": "korean",
                    "full_content": full_content,
                })
                item_id += 1

        n_items = sum(1 for r in all_rows if r["config"] == cfg)
        print(f"{n_items} items")

    return all_rows


def main():
    print("=" * 70)
    print("KMMLU Task Metadata Builder")
    print("=" * 70)

    rows = load_kmmlu()
    if not rows:
        print("ERROR: No data loaded.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\nTotal items loaded: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Configs: {df['config'].nunique()}")

    # ── task_metadata.csv ──
    meta_cols = [
        "item_id", "question_short", "answer_key", "category", "config",
        "human_accuracy", "split", "source_dataset", "language",
    ]
    meta_df = df[meta_cols].copy()
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved task_metadata.csv: {meta_path}")
    print(f"  Shape: {meta_df.shape}")

    # ── item_content.csv ──
    content_df = df[["item_id", "full_content"]].copy()
    content_df.columns = ["item_id", "content"]
    content_path = PROCESSED_DIR / "item_content.csv"
    content_df.to_csv(content_path, index=False)
    print(f"Saved item_content.csv: {content_path}")
    print(f"  Shape: {content_df.shape}")

    # ── response_matrix.csv (human accuracy pseudo-column) ──
    # We use the test split only for the response matrix
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) > 0:
        rm = test_df[["item_id", "human_accuracy"]].set_index("item_id")
        rm.columns = ["human_baseline"]
        rm_path = PROCESSED_DIR / "response_matrix.csv"
        rm.to_csv(rm_path)
        print(f"Saved response_matrix.csv: {rm_path}")
        print(f"  Shape: {rm.shape}")
        print(f"  Human accuracy stats (test split):")
        print(f"    Mean: {rm['human_baseline'].mean():.4f}")
        print(f"    Std:  {rm['human_baseline'].std():.4f}")
        print(f"    Min:  {rm['human_baseline'].min():.4f}")
        print(f"    Max:  {rm['human_baseline'].max():.4f}")
        n_zero = (rm["human_baseline"] == 0.0).sum()
        n_one = (rm["human_baseline"] == 1.0).sum()
        print(f"    Items with 0% human acc: {n_zero}")
        print(f"    Items with 100% human acc: {n_one}")
    else:
        print("WARNING: No test split items found.")

    # ── model_summary.csv ──
    summary_rows = []
    if len(test_df) > 0:
        overall_acc = test_df["human_accuracy"].mean()
        summary_rows.append({
            "model": "human_baseline",
            "source": "HAERAE-HUB/KMMLU",
            "overall_accuracy": overall_acc,
            "n_items": len(test_df),
            "notes": "Per-item human accuracy from KMMLU dataset",
        })
        # Per-category human accuracy
        for cat in sorted(test_df["config"].unique()):
            cat_df = test_df[test_df["config"] == cat]
            summary_rows.append({
                "model": f"human_baseline_{cat}",
                "source": "HAERAE-HUB/KMMLU",
                "overall_accuracy": cat_df["human_accuracy"].mean(),
                "n_items": len(cat_df),
                "notes": f"Human accuracy for {cat}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model_summary.csv: {summary_path}")
    print(f"  Entries: {len(summary_df)}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Total items: {len(df)}")
    print(f"  Test items:  {len(test_df)}")
    print(f"  Configs:     {df['config'].nunique()}")
    print(f"  Language:    Korean")
    print(f"  Has human accuracy: Yes (per-item)")

    if len(test_df) > 0:
        print(f"\n  Per-config test-set human accuracy:")
        for cfg in sorted(test_df["config"].unique()):
            cfg_df = test_df[test_df["config"] == cfg]
            acc = cfg_df["human_accuracy"].mean()
            print(f"    {cfg:45s}: {acc:.4f} ({len(cfg_df)} items)")

    print("\nDone!")


if __name__ == "__main__":
    main()
