#!/usr/bin/env python3
"""
Build task metadata and item content for C-Eval from HuggingFace.

Data source:
  ceval/ceval-exam on HuggingFace
  - 52 subject configs covering Chinese professional/academic exams
  - Splits: test, val, dev
  - Columns: id, question, A, B, C, D, answer (letter), explanation
  - Test set answers were released on HuggingFace

No per-model prediction data is available, so we build only:
  - task_metadata.csv  : item_id, question (first 200 chars), answer_key,
                         config, split, source_dataset, language
  - item_content.csv   : item_id, full question + options text
  - model_summary.csv  : empty placeholder (no model evaluations)
  - response_matrix.csv: placeholder with item_ids and answer keys only

All paths use Path(__file__).resolve().parent.parent
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
# C-Eval configs (52 subjects)
# ──────────────────────────────────────────────────────────────────────
CEVAL_CONFIGS = [
    "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
    "business_administration", "chinese_language_and_literature",
    "civil_servant", "clinical_medicine", "college_chemistry",
    "college_economics", "college_physics", "college_programming",
    "computer_architecture", "computer_network", "discrete_mathematics",
    "education_science", "electrical_engineer", "environmental_impact_assessment_engineer",
    "fire_engineer", "high_school_biology", "high_school_chemistry",
    "high_school_chinese", "high_school_geography", "high_school_history",
    "high_school_mathematics", "high_school_physics", "high_school_politics",
    "ideological_and_moral_cultivation", "law", "legal_professional",
    "logic", "mao_zedong_thought", "marxism", "metrology_engineer",
    "middle_school_biology", "middle_school_chemistry",
    "middle_school_geography", "middle_school_history",
    "middle_school_mathematics", "middle_school_physics",
    "middle_school_politics", "modern_chinese_history", "operating_system",
    "physician", "plant_protection", "probability_and_statistics",
    "professional_tour_guide", "sports_science", "tax_accountant",
    "teacher_qualification", "urban_and_rural_planner",
    "veterinary_medicine",
]


def load_ceval():
    """Load all C-Eval configs from HuggingFace."""
    from datasets import load_dataset

    all_rows = []
    item_id = 0

    for cfg in CEVAL_CONFIGS:
        print(f"  Loading {cfg}...", end=" ")
        try:
            ds = load_dataset("ceval/ceval-exam", cfg)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        for split_name in ["test", "val", "dev"]:
            if split_name not in ds:
                continue
            split_ds = ds[split_name]
            for row in split_ds:
                q = row["question"]
                a_key = str(row.get("answer", "")).strip()
                explanation = str(row.get("explanation", "")).strip()
                options_text = (
                    f"A: {row['A']}\nB: {row['B']}\n"
                    f"C: {row['C']}\nD: {row['D']}"
                )
                full_content = f"{q}\n\n{options_text}"
                if explanation:
                    full_content += f"\n\nExplanation: {explanation}"

                all_rows.append({
                    "item_id": f"ceval_{item_id:06d}",
                    "question_short": q[:200],
                    "answer_key": a_key,
                    "config": cfg,
                    "split": split_name,
                    "source_dataset": "ceval/ceval-exam",
                    "language": "chinese",
                    "has_explanation": bool(explanation),
                    "full_content": full_content,
                })
                item_id += 1

        n_items = sum(1 for r in all_rows if r["config"] == cfg)
        print(f"{n_items} items")

    return all_rows


def main():
    print("=" * 70)
    print("C-Eval Task Metadata Builder")
    print("=" * 70)

    rows = load_ceval()
    if not rows:
        print("ERROR: No data loaded.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\nTotal items loaded: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Configs: {df['config'].nunique()}")

    # ── task_metadata.csv ──
    meta_cols = [
        "item_id", "question_short", "answer_key", "config",
        "split", "source_dataset", "language", "has_explanation",
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

    # ── response_matrix.csv (gold answers only, no model responses) ──
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) > 0:
        rm = test_df[["item_id", "answer_key"]].set_index("item_id")
        rm.columns = ["gold_answer"]
        rm_path = PROCESSED_DIR / "response_matrix.csv"
        rm.to_csv(rm_path)
        print(f"Saved response_matrix.csv (gold answers): {rm_path}")
        print(f"  Shape: {rm.shape}")
        print(f"  Answer distribution:")
        print(f"    {test_df['answer_key'].value_counts().to_dict()}")
    else:
        print("WARNING: No test split items found.")

    # ── model_summary.csv (placeholder) ──
    summary_df = pd.DataFrame(columns=[
        "model", "source", "overall_accuracy", "n_items", "notes"
    ])
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model_summary.csv (empty): {summary_path}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Total items: {len(df)}")
    print(f"  Test items:  {len(test_df)}")
    print(f"  Configs:     {df['config'].nunique()}")
    print(f"  Language:    Chinese")
    print(f"  Has model predictions: No")
    print(f"  Has gold answers: Yes (test set answers released)")
    has_expl = df["has_explanation"].sum()
    print(f"  Items with explanation: {has_expl}")

    if len(test_df) > 0:
        print(f"\n  Per-config test item counts:")
        for cfg in sorted(test_df["config"].unique()):
            n = (test_df["config"] == cfg).sum()
            print(f"    {cfg:50s}: {n} items")

    print("\nDone!")


if __name__ == "__main__":
    main()
