#!/usr/bin/env python3
"""
Build task metadata, item content, and response matrix for CMMLU (Chinese MMLU).

Data sources:
  1. haonan-li/cmmlu on HuggingFace
     - 67 subject configs covering Chinese academic/professional exams
     - Loaded from the cmmlu_v1_0_1.zip file (CSV files inside)
     - Columns: Question, A, B, C, D, Answer
     - Splits: test and dev

  2. opencompass/compass_academic_predictions on HuggingFace (GATED DATASET)
     - Per-item evaluation results for 20+ models on all 67 CMMLU subjects
     - Each subject has a folder: results_stations/cmmlu-{subject}/
     - Each model has a JSON file with per-item predictions
     - Models include: DeepSeek-V3, DeepSeek-R1, GPT-4o-mini, Qwen2.5 series,
       Gemma-2 series, InternLM series, LLaMA-3.x series, etc.
     - ACCESS: This dataset requires approval at:
       https://huggingface.co/datasets/opencompass/compass_academic_predictions

Outputs (in ../processed/):
  - task_metadata.csv  : item_id, question (first 200 chars), answer_key,
                         config, split, source_dataset, language
  - item_content.csv   : item_id, full question + options text
  - model_summary.csv  : per-model aggregate statistics (or empty placeholder)
  - response_matrix.csv: models x items matrix (binary 0/1), or placeholder
                         with item_ids and answer keys only if no access

All paths use Path(__file__).resolve().parent.parent
"""

import json
import sys
import warnings
import zipfile
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
# CMMLU subjects (67 total, matching OpenCompass folder names)
# ──────────────────────────────────────────────────────────────────────
CMMLU_SUBJECTS = [
    "agronomy", "anatomy", "ancient_chinese", "arts", "astronomy",
    "business_ethics", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
    "college_actuarial_science", "college_education",
    "college_engineering_hydrology", "college_law", "college_mathematics",
    "college_medical_statistics", "college_medicine", "computer_science",
    "computer_security", "conceptual_physics", "construction_project_management",
    "economics", "education", "electrical_engineering", "elementary_chinese",
    "elementary_commonsense", "elementary_information_and_technology",
    "elementary_mathematics", "ethnology", "food_science", "genetics",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_geography", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "human_sexuality", "international_law",
    "journalism", "jurisprudence", "legal_and_moral_basis", "logical",
    "machine_learning", "management", "marketing", "marxist_theory",
    "modern_chinese", "nutrition", "philosophy", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_study", "sociology", "sports_science",
    "traditional_chinese_medicine", "virology", "world_history",
    "world_religions",
]


def download_cmmlu_zip():
    """Download the CMMLU zip from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        "haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset"
    )
    return zip_path


def load_cmmlu(zip_path):
    """Load all CMMLU subjects from the zip file."""
    all_rows = []
    item_id = 0

    with zipfile.ZipFile(zip_path) as zf:
        csv_files = sorted([
            n for n in zf.namelist()
            if n.endswith(".csv") and "/" in n
        ])

        for csv_name in csv_files:
            parts = csv_name.split("/")
            if len(parts) < 2:
                continue
            split_name = parts[0]  # "test" or "dev"
            config = parts[1].replace(".csv", "")

            print(f"  Loading {split_name}/{config}...", end=" ")
            try:
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            # Normalize column names
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl == "question":
                    col_map[c] = "Question"
                elif cl == "answer":
                    col_map[c] = "Answer"
                elif cl in ("a", "b", "c", "d"):
                    col_map[c] = cl.upper()
            df = df.rename(columns=col_map)

            n_items = 0
            for _, row in df.iterrows():
                q = str(row.get("Question", ""))
                a_key = str(row.get("Answer", "")).strip()
                opt_a = str(row.get("A", ""))
                opt_b = str(row.get("B", ""))
                opt_c = str(row.get("C", ""))
                opt_d = str(row.get("D", ""))

                options_text = (
                    f"A: {opt_a}\nB: {opt_b}\n"
                    f"C: {opt_c}\nD: {opt_d}"
                )
                full_content = f"{q}\n\n{options_text}"

                all_rows.append({
                    "item_id": f"cmmlu_{item_id:06d}",
                    "question_short": q[:200],
                    "answer_key": a_key,
                    "config": config,
                    "split": split_name,
                    "source_dataset": "haonan-li/cmmlu",
                    "language": "chinese",
                    "full_content": full_content,
                })
                item_id += 1
                n_items += 1

            print(f"{n_items} items")

    return all_rows


def discover_models():
    """List available models in the OpenCompass CMMLU predictions."""
    from huggingface_hub import HfApi

    api = HfApi()
    print("Discovering models in OpenCompass CMMLU predictions...")

    # Check first subject folder for available model files
    subject = CMMLU_SUBJECTS[0]
    path = f"results_stations/cmmlu-{subject}"
    try:
        files = list(api.list_repo_tree(
            "opencompass/compass_academic_predictions",
            repo_type="dataset",
            path_in_repo=path,
        ))
        models = []
        for f in files:
            if hasattr(f, "path") and f.path.endswith(".json"):
                model_name = f.path.split("/")[-1].replace(".json", "")
                models.append(model_name)
        print(f"  Found {len(models)} models: {models}")
        return models
    except Exception as e:
        print(f"  Error: {e}")
        return []


def download_predictions(model_name, subject):
    """Download per-item predictions for a model on a CMMLU subject."""
    from huggingface_hub import hf_hub_download

    filename = f"results_stations/cmmlu-{subject}/{model_name}.json"
    try:
        path = hf_hub_download(
            "opencompass/compass_academic_predictions",
            filename,
            repo_type="dataset",
        )
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def extract_per_item_scores(data):
    """Extract per-item binary scores from OpenCompass JSON predictions.

    OpenCompass format (expected):
      List of dicts, each with fields like:
        - "gold" or "answer": the correct answer
        - "prediction" or "pred": model's predicted answer
        - "correct" or "score": binary correctness indicator
      OR a dict keyed by item index/ID with similar fields.

    Returns:
      List of int (0 or 1) per item, or None if format is unrecognized.
    """
    if isinstance(data, list):
        scores = []
        for item in data:
            if isinstance(item, dict):
                # Try common field names for correctness
                for key in ["correct", "score", "acc", "accuracy"]:
                    if key in item:
                        scores.append(int(bool(item[key])))
                        break
                else:
                    # Try matching prediction to gold
                    gold = item.get("gold", item.get("answer", item.get("gold_answer", "")))
                    pred = item.get("prediction", item.get("pred", item.get("response", "")))
                    if gold and pred:
                        scores.append(1 if str(gold).strip() == str(pred).strip() else 0)
                    else:
                        return None
            elif isinstance(item, (int, float)):
                scores.append(int(item))
            else:
                return None
        return scores

    elif isinstance(data, dict):
        # Dict format: keys are item indices or IDs
        scores = []
        for key in sorted(data.keys(), key=lambda k: int(k) if k.isdigit() else k):
            item = data[key]
            if isinstance(item, dict):
                for field in ["correct", "score", "acc"]:
                    if field in item:
                        scores.append(int(bool(item[field])))
                        break
                else:
                    gold = item.get("gold", item.get("answer", ""))
                    pred = item.get("prediction", item.get("pred", ""))
                    if gold and pred:
                        scores.append(1 if str(gold).strip() == str(pred).strip() else 0)
                    else:
                        return None
            elif isinstance(item, (int, float)):
                scores.append(int(item))
            else:
                return None
        return scores

    return None


def build_response_matrix_from_predictions(models):
    """Build the full CMMLU response matrix from OpenCompass predictions."""
    print("\n" + "=" * 70)
    print("Building CMMLU Response Matrix from OpenCompass")
    print("=" * 70)

    # Load existing task_metadata to align items
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        test_meta = meta_df[meta_df["split"] == "test"]
        print(f"  Existing task_metadata: {len(meta_df)} total, {len(test_meta)} test items")
    else:
        test_meta = None

    model_responses = {}
    subject_item_counts = {}

    for model_name in models:
        print(f"\n  Model: {model_name}")
        all_scores = []

        for subj in CMMLU_SUBJECTS:
            data = download_predictions(model_name, subj)
            if data is None:
                print(f"    {subj}: SKIP (download failed)")
                all_scores = None
                break

            scores = extract_per_item_scores(data)
            if scores is None:
                print(f"    {subj}: SKIP (format unrecognized)")
                all_scores = None
                break

            if subj not in subject_item_counts:
                subject_item_counts[subj] = len(scores)
            elif len(scores) != subject_item_counts[subj]:
                print(f"    {subj}: SKIP (item count mismatch)")
                all_scores = None
                break

            all_scores.extend(scores)

        if all_scores is not None:
            model_responses[model_name] = all_scores
            acc = np.mean(all_scores)
            print(f"    -> acc={acc:.3f} ({len(all_scores)} items)")

    if not model_responses:
        print("\nNo models loaded successfully!")
        return None, None

    # Build item IDs
    item_ids = []
    for subj in CMMLU_SUBJECTS:
        n = subject_item_counts.get(subj, 0)
        for j in range(n):
            item_ids.append(f"cmmlu_oc_{subj}_{j:04d}")

    # Build response matrix: rows=models, columns=items
    rm_df = pd.DataFrame(model_responses, index=item_ids).T
    rm_df.index.name = "model"

    # Build model summary
    summary_rows = []
    for model_name, scores in model_responses.items():
        summary_rows.append({
            "model": model_name,
            "source": "OpenCompass",
            "overall_accuracy": np.mean(scores),
            "n_items": len(scores),
            "notes": "compass_academic_predictions",
        })
    summary_df = pd.DataFrame(summary_rows)

    return rm_df, summary_df


def main():
    print("=" * 70)
    print("CMMLU Task Metadata Builder")
    print("=" * 70)

    # ── Step 1: Build item metadata from HuggingFace ──
    print("\nDownloading CMMLU zip from HuggingFace...")
    zip_path = download_cmmlu_zip()
    print(f"  Zip path: {zip_path}")

    print("\nLoading CMMLU data...")
    rows = load_cmmlu(zip_path)
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
        "split", "source_dataset", "language",
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

    # ── Step 2: Attempt to build response matrix from OpenCompass predictions ──
    models = discover_models()

    if not models:
        print("\n" + "!" * 70)
        print("CANNOT ACCESS: opencompass/compass_academic_predictions")
        print("This is a gated dataset. To get access:")
        print("  1. Visit: https://huggingface.co/datasets/opencompass/compass_academic_predictions")
        print("  2. Click 'Access repository' and accept the terms")
        print("  3. Re-run this script")
        print()
        print("Available CMMLU subjects in this dataset (67):")
        for subj in CMMLU_SUBJECTS:
            print(f"  cmmlu-{subj}")
        print()
        print("Known models (from directory listing):")
        print("  MiniMax-Text-01, deepseek-chat-v3, deepseek-r1,")
        print("  deepseek-v2_5-turbomind, gemma-2-27b-it-turbomind,")
        print("  gemma-2-9b-it-turbomind, gemma3_27b_it,")
        print("  gpt-4o-mini-2024-07-18, internlm2_5-20b-chat-turbomind,")
        print("  internlm2_5-7b-chat-turbomind, internlm3-8b-instruct-turbomind,")
        print("  llama-3_1-8b-instruct-turbomind, llama-3_2-3b-instruct-turbomind,")
        print("  llama-3_3-70b-instruct-turbomind, qwen-max-2025-01-25,")
        print("  qwen2.5-14b/32b/72b/7b-instruct-turbomind")
        print("!" * 70)
    else:
        # Build response matrix
        rm_df, oc_summary_df = build_response_matrix_from_predictions(models)

        if rm_df is not None:
            rm_path = PROCESSED_DIR / "response_matrix.csv"
            rm_df.to_csv(rm_path)
            print(f"\nSaved response_matrix.csv: {rm_path}")
            print(f"  Shape: {rm_df.shape}")

        if oc_summary_df is not None:
            summary_path = PROCESSED_DIR / "model_summary.csv"
            oc_summary_df.to_csv(summary_path, index=False)
            print(f"Saved model_summary.csv: {summary_path}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Total items: {len(df)}")
    print(f"  Test items:  {len(test_df)}")
    print(f"  Configs:     {df['config'].nunique()}")
    print(f"  Language:    Chinese")
    print(f"  Has model predictions: {'Yes' if models else 'No'}")
    print(f"  Has gold answers: Yes")

    if len(test_df) > 0:
        print(f"\n  Per-config test item counts:")
        for cfg in sorted(test_df["config"].unique()):
            n = (test_df["config"] == cfg).sum()
            print(f"    {cfg:50s}: {n} items")

    print("\nDone!")


if __name__ == "__main__":
    main()
