"""
Build PerMedCQA item metadata from Persian Medical Consumer QA.

Data source:
  - GitHub: PerMedCQA/PerMedCQA
    https://github.com/PerMedCQA/PerMedCQA
  - Test set: Data/test.json
  - 3,512 Persian consumer medical QA items (free-text, not MCQ)

PerMedCQA overview:
  - Persian medical consumer question answering
  - Free-text QA format (not multiple choice)
  - Items have: Title, Category, Specialty, Sex, Age, dataset_source
  - Collected from Persian medical forums

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (title, category, specialty, demographics)
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

REPO_URL = "https://github.com/PerMedCQA/PerMedCQA.git"


def download_raw():
    """Clone the PerMedCQA repository."""
    repo_dir = RAW_DIR / "PerMedCQA"
    if repo_dir.exists():
        print(f"  Already cloned: {repo_dir}")
        return repo_dir
    print("  Cloning PerMedCQA repository...")
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
    print("PerMedCQA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    repo_dir = download_raw()

    test_file = repo_dir / "Data" / "test.json"
    if not test_file.exists():
        # Try alternate paths
        for candidate in [
            repo_dir / "data" / "test.json",
            repo_dir / "test.json",
        ]:
            if candidate.exists():
                test_file = candidate
                break
        else:
            print(f"  ERROR: test.json not found in {repo_dir}")
            sys.exit(1)

    print(f"  Loading items from: {test_file}")
    with open(test_file, encoding="utf-8") as f:
        data = json.load(f)

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "title", "category", "specialty", "sex", "age",
            "source", "language", "benchmark",
        ])
        for item in data:
            writer.writerow([
                item.get("instance_id", ""),
                item.get("Title", ""),
                item.get("Category", ""),
                item.get("Specialty", ""),
                item.get("Sex", ""),
                item.get("Age", ""),
                item.get("dataset_source", ""),
                "Persian",
                "PerMedCQA",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "title", "category", "specialty", "sex", "age", "source"])
        for item in data:
            writer.writerow([
                item.get("instance_id", ""),
                item.get("Title", ""),
                item.get("Category", ""),
                item.get("Specialty", ""),
                item.get("Sex", ""),
                item.get("Age", ""),
                item.get("dataset_source", ""),
            ])

    print(f"\n  PerMedCQA: {len(data)} test items (free-text QA, no per-item model predictions)")
    print(f"  Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
