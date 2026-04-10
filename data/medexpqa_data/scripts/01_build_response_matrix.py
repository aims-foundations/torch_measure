"""
Build MedExpQA item metadata from Spanish Medical QA with explanations.

Data source:
  - HuggingFace: HiTZ/MedExpQA
    https://huggingface.co/datasets/HiTZ/MedExpQA
  - Spanish medical QA from CasiMedicos exam platform
  - Also available in English, French, and Italian translations

MedExpQA overview:
  - Spanish (primary) medical question answering with expert explanations
  - Based on CasiMedicos clinical case questions
  - RAG-augmented versions available
  - Multilingual: Spanish, English, French, Italian

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, correct_option)
  - processed/item_content.csv: Full item content
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# The original script used a GitHub repo; HuggingFace also hosts it
REPO_URL = "https://github.com/ixa-ehu/MedExpQA.git"
HF_DATASET = "HiTZ/MedExpQA"


def download_raw():
    """Clone the MedExpQA repository."""
    repo_dir = RAW_DIR / "MedExpQA"
    if repo_dir.exists():
        print(f"  Already cloned: {repo_dir}")
        return repo_dir
    print("  Cloning MedExpQA repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}")
        print("  Trying HuggingFace download instead...")
        return download_from_hf()
    return repo_dir


def download_from_hf():
    """Fallback: download from HuggingFace."""
    output_dir = RAW_DIR / "MedExpQA"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, "es", split="test")
        test_file = output_dir / "data" / "es" / "test.es.casimedicos.rag.jsonl"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Downloaded {len(ds)} Spanish test items from HuggingFace")
    except ImportError:
        print("  ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        sys.exit(1)

    return output_dir


def main():
    print("MedExpQA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    repo_dir = download_raw()

    # Find Spanish test file
    test_file = repo_dir / "data" / "es" / "test.es.casimedicos.rag.jsonl"
    if not test_file.exists():
        # Try alternate locations
        for candidate in [
            repo_dir / "data" / "es" / "test.jsonl",
            repo_dir / "es" / "test.es.casimedicos.rag.jsonl",
        ]:
            if candidate.exists():
                test_file = candidate
                break
        else:
            # Try to find any JSONL file
            jsonl_files = list(repo_dir.rglob("*.jsonl"))
            if jsonl_files:
                test_file = jsonl_files[0]
                print(f"  Using: {test_file}")
            else:
                print(f"  ERROR: No test file found in {repo_dir}")
                sys.exit(1)

    print(f"  Loading items from: {test_file}")
    items = []
    with open(test_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            items.append({
                "sample_id": f"medexpqa_es_{i:05d}",
                "correct_option": item.get("correct_option", ""),
                "full_question": item.get("full_question", ""),
                "question": item.get("question", ""),
            })

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "correct_option", "language", "benchmark"])
        for item in items:
            q = item["full_question"] or item["question"]
            writer.writerow([
                item["sample_id"],
                q,
                item["correct_option"],
                "Spanish",
                "MedExpQA",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "full_question", "correct_option"])
        for item in items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["full_question"],
                item["correct_option"],
            ])

    print(f"\n  MedExpQA Spanish: {len(items)} test items (questions only, no per-item model predictions)")
    print(f"  Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
