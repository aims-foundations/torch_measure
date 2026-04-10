#!/usr/bin/env python3
"""
Build task metadata, item content, and response matrix for C-Eval.

Data sources:
  1. ceval/ceval-exam on HuggingFace
     - 52 subject configs covering Chinese professional/academic exams
     - Splits: test, val, dev
     - Columns: id, question, A, B, C, D, answer (letter), explanation
     - Test set answers were released on HuggingFace

  2. OpenCompass compass_academic_predictions on HuggingFace (ATTEMPTED)
     - Does NOT currently include C-Eval subjects (as of March 2026)
     - Contains CMMLU, MMLU, BBH, and other benchmarks, but NOT C-Eval

     Other searched sources with NO public per-item C-Eval results:
     - github.com/hkust-nlp/ceval: Code + data only, no model predictions
     - cevalbenchmark.com leaderboard: Only aggregate per-subject scores
     - OpenCompass: Has C-Eval in their eval configs but per-item predictions
       are not in the public compass_academic_predictions dataset
     - Qwen/ChatGLM/Baichuan repos: Only aggregate scores, no per-item data
     - FlagEval: No per-item results publicly available
     - C-Eval test set requires submission to cevalbenchmark.com for scoring

     Potential future sources:
     1. OpenCompass compass_academic_predictions may add C-Eval subjects
        (currently gated at huggingface.co/datasets/opencompass/compass_academic_predictions)
     2. Run lm-evaluation-harness locally on C-Eval val set
        (val set has ground truth answers; test set requires server submission)

Outputs (in ../processed/):
  - task_metadata.csv  : item_id, question (first 200 chars), answer_key,
                         config, split, source_dataset, language
  - item_content.csv   : item_id, full question + options text
  - model_summary.csv  : empty placeholder (no model evaluations)
  - response_matrix.csv: placeholder with item_ids and answer keys only

All paths use Path(__file__).resolve().parent
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
_BENCHMARK_DIR = Path(__file__).resolve().parent
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


def build_response_matrix_from_predictions():
    """Attempt to build response matrix from OpenCompass predictions.

    Currently reports that no public per-item model prediction data exists
    for C-Eval and prints recommendations for future data acquisition.
    """
    print("\n" + "=" * 70)
    print("C-Eval Response Matrix Builder (OpenCompass Source)")
    print("=" * 70)

    print()
    print("STATUS: No public per-item model prediction data found for C-Eval.")
    print()
    print("Searched sources (all negative):")
    print("  1. github.com/hkust-nlp/ceval -- Code only, no model predictions")
    print("  2. cevalbenchmark.com -- Aggregate leaderboard scores only")
    print("  3. OpenCompass compass_academic_predictions -- CMMLU yes, C-Eval no")
    print("  4. Qwen/ChatGLM/Baichuan repos -- Aggregate per-subject only")
    print("  5. FlagEval -- No per-item results publicly available")
    print("  6. HuggingFace Open LLM Leaderboard -- English benchmarks only")
    print()
    print("C-Eval test set is unique in that it requires server-side submission")
    print("to cevalbenchmark.com for scoring. Neither the questions' answers nor")
    print("the models' predictions are published per-item.")
    print()
    print("C-Eval val set (1,346 items across 52 subjects) has ground truth")
    print("answers and can be evaluated locally with lm-evaluation-harness.")
    print()
    print("Recommendation:")
    print("  1. Request access to opencompass/compass_academic_predictions")
    print("     (may add C-Eval in future)")
    print("  2. Run lm-evaluation-harness on C-Eval val set for target models")
    print("  3. Check cevalbenchmark.com periodically for data releases")
    print()
    print("Current files:")
    for f in sorted(PROCESSED_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


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

    # ── Attempt to build response matrix from predictions ──
    build_response_matrix_from_predictions()

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

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
