"""
Build FrenchMedMCQA item metadata from French medical specialization exam MCQs.

Data source:
  - HuggingFace: qanastek/FrenchMedMCQA
    https://huggingface.co/datasets/qanastek/FrenchMedMCQA
  - 622 test items from the French Pharmacy specialization diploma exam
  - Each item has: id, question, correct_answers, nbr_correct_answers, subject_name

FrenchMedMCQA overview:
  - French medical multiple-choice questions
  - Pharmacy specialization diploma (DES Pharmacie)
  - Questions can have multiple correct answers

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answers, subject)
  - processed/item_content.csv: Full item content with answer options
"""

import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET = "qanastek/FrenchMedMCQA"


def download_raw():
    """Download FrenchMedMCQA dataset from HuggingFace."""
    output_dir = RAW_DIR / "FrenchMedMCQA"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / "test.json"

    if test_file.exists():
        print(f"  Already downloaded: {test_file}")
        return test_file

    print(f"  Downloading {HF_DATASET} from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split="test")
        # Save as JSON list
        items = [item for item in ds]
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  Downloaded {len(items)} test items")
    except ImportError:
        print("  ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        sys.exit(1)

    return test_file


def main():
    print("FrenchMedMCQA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    test_file = download_raw()

    print(f"  Loading items from: {test_file}")
    with open(test_file, encoding="utf-8") as f:
        data = json.load(f)

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "question", "correct_answers", "nbr_correct_answers",
            "subject", "language", "benchmark",
        ])
        for item in data:
            correct_answers = item.get("correct_answers", [])
            if isinstance(correct_answers, list):
                correct_answers = ",".join(correct_answers)
            writer.writerow([
                item.get("id", ""),
                item.get("question", ""),
                correct_answers,
                item.get("nbr_correct_answers", ""),
                item.get("subject_name", ""),
                "French",
                "FrenchMedMCQA",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "question", "answer_a", "answer_b", "answer_c",
            "answer_d", "answer_e", "correct_answers", "nbr_correct_answers", "subject",
        ])
        for item in data:
            answers = item.get("answers", {})
            correct_answers = item.get("correct_answers", [])
            if isinstance(correct_answers, list):
                correct_answers = ",".join(correct_answers)
            writer.writerow([
                item.get("id", ""),
                item.get("question", ""),
                answers.get("a", "") if isinstance(answers, dict) else "",
                answers.get("b", "") if isinstance(answers, dict) else "",
                answers.get("c", "") if isinstance(answers, dict) else "",
                answers.get("d", "") if isinstance(answers, dict) else "",
                answers.get("e", "") if isinstance(answers, dict) else "",
                correct_answers,
                item.get("nbr_correct_answers", ""),
                item.get("subject_name", ""),
            ])

    print(f"\n  FrenchMedMCQA: {len(data)} test items (questions only, no per-item model predictions)")
    print(f"  Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
