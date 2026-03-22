#!/usr/bin/env python3
"""
Build response matrix for CMMLU from OpenCompass per-item evaluation results.

Data source:
  opencompass/compass_academic_predictions on HuggingFace (GATED DATASET)
  - Per-item evaluation results for 20+ models on all 67 CMMLU subjects
  - Each subject has a folder: results_stations/cmmlu-{subject}/
  - Each model has a JSON file with per-item predictions
  - Models include: DeepSeek-V3, DeepSeek-R1, GPT-4o-mini, Qwen2.5 series,
    Gemma-2 series, InternLM series, LLaMA-3.x series, etc.

ACCESS: This dataset requires approval at:
  https://huggingface.co/datasets/opencompass/compass_academic_predictions

Once access is granted, run this script to build the response matrix.

Output:
  - response_matrix.csv : models x items matrix (binary 0/1)
  - model_summary.csv   : per-model aggregate statistics

All paths use Path(__file__).resolve().parent.parent
"""

import json
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


def build_response_matrix(models):
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
    print("CMMLU Response Matrix Builder (OpenCompass Source)")
    print("=" * 70)

    # Discover available models
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
        sys.exit(1)

    # Build response matrix
    rm_df, summary_df = build_response_matrix(models)

    if rm_df is not None:
        rm_path = PROCESSED_DIR / "response_matrix.csv"
        rm_df.to_csv(rm_path)
        print(f"\nSaved response_matrix.csv: {rm_path}")
        print(f"  Shape: {rm_df.shape}")

    if summary_df is not None:
        summary_path = PROCESSED_DIR / "model_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved model_summary.csv: {summary_path}")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    if rm_df is not None:
        print(f"  Models: {rm_df.shape[0]}")
        print(f"  Items: {rm_df.shape[1]}")
        print(f"  Mean accuracy: {summary_df['overall_accuracy'].mean():.3f}")
    print(f"  Source: OpenCompass compass_academic_predictions")
    print(f"  Note: Uses CMMLU test set items evaluated with gen-mode")
    print("\nDone!")


if __name__ == "__main__":
    main()
