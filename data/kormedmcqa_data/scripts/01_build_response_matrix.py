"""
Build KorMedMCQA item metadata from Korean Healthcare Professional Licensing Exams.

Data source:
  - GitHub: sean-jang/KorMedMCQA
    https://github.com/sean-jang/KorMedMCQA
  - Test sets for 3 professional exams:
    Doctor (285 items), Nurse (587 items), Pharmacist (614 items)
  - CSV files: data/{doctor,nurse,pharmacist}-test.csv

KorMedMCQA overview:
  - Korean medical licensing exam benchmark
  - Multiple-choice with 5 options (A-E)
  - Each item has subject, year, question text, and options

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answer, subject, year)
  - processed/item_content.csv: Full item content with all options
"""

import csv
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

REPO_URL = "https://github.com/sean-jang/KorMedMCQA.git"


def download_raw():
    """Clone the KorMedMCQA repository."""
    repo_dir = RAW_DIR / "KorMedMCQA"
    if repo_dir.exists():
        print(f"  Already cloned: {repo_dir}")
        return repo_dir
    print("  Cloning KorMedMCQA repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}")
        sys.exit(1)
    return repo_dir


def main():
    print("KorMedMCQA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    repo_dir = download_raw()
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        print(f"  ERROR: data/ directory not found in {repo_dir}")
        sys.exit(1)

    print("  Loading test items...")
    all_items = []
    for test_file in sorted(data_dir.glob("*-test.csv")):
        subject = test_file.name.replace("-test.csv", "")
        with open(test_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                all_items.append({
                    "sample_id": f"kormed_{subject}_{i:05d}",
                    "subject": row.get("subject", subject),
                    "year": row.get("year", ""),
                    "question": row.get("question", ""),
                    "A": row.get("A", ""),
                    "B": row.get("B", ""),
                    "C": row.get("C", ""),
                    "D": row.get("D", ""),
                    "E": row.get("E", ""),
                    "answer": row.get("answer", ""),
                })
        print(f"  {subject}: {sum(1 for it in all_items if subject in it['sample_id'])} items")

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "subject", "year", "language", "benchmark"])
        for item in all_items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["subject"],
                item["year"],
                "Korean",
                "KorMedMCQA",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "A", "B", "C", "D", "E", "answer", "subject", "year"])
        for item in all_items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["A"],
                item["B"],
                item["C"],
                item["D"],
                item["E"],
                item["answer"],
                item["subject"],
                item["year"],
            ])

    print(f"\n  KorMedMCQA: {len(all_items)} test items (questions only, no per-item model predictions)")
    print(f"  Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
