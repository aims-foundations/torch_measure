"""
Build MedArabiQ item metadata from Arabic Medical Tasks.

Data source:
  - GitHub: Harithaaaa/MedArabiQ
    https://github.com/Harithaaaa/MedArabiQ
  - Multiple task types in datasets/ directory:
    MCQ, fill-in-blank, patient-doctor QA
  - CSV files with Question, Answer, Category columns

MedArabiQ overview:
  - Arabic medical question answering benchmark
  - Multiple task formats (MCQ, fill-in-blank, patient-doctor)
  - Covers various medical categories

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answer, task, category)
  - processed/item_content.csv: Full item content
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/Harithaaaa/MedArabiQ.git"


def download_raw():
    """Clone the MedArabiQ repository."""
    repo_dir = RAW_DIR / "MedArabiQ"
    if repo_dir.exists():
        print(f"  Already cloned: {repo_dir}")
        return repo_dir
    print("  Cloning MedArabiQ repository...")
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
    print("MedArabiQ Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    repo_dir = download_raw()
    data_dir = repo_dir / "datasets"
    if not data_dir.exists():
        print(f"  ERROR: datasets/ directory not found in {repo_dir}")
        sys.exit(1)

    print("  Loading items from CSV files...")
    all_items = []
    for csvfile in sorted(data_dir.glob("*.csv")):
        task = csvfile.stem
        try:
            with open(csvfile, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for i, row in enumerate(reader):
                    all_items.append({
                        "sample_id": f"medarabiq_{task}_{i:05d}",
                        "task": task,
                        "question": row.get("Question", ""),
                        "answer": row.get("Answer", ""),
                        "category": row.get("Category", ""),
                    })
                    count += 1
                print(f"    {task}: {count} items")
        except Exception as e:
            print(f"    WARNING: Error reading {csvfile}: {e}")

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "task", "category", "language", "benchmark"])
        for item in all_items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["task"],
                item["category"],
                "Arabic",
                "MedArabiQ",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "answer", "task", "category"])
        for item in all_items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["task"],
                item["category"],
            ])

    # Task breakdown
    from collections import Counter

    task_counts = Counter(item["task"] for item in all_items)
    print(f"\n  MedArabiQ: {len(all_items)} items across {len(task_counts)} tasks:")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count}")
    print(f"  (questions only, no per-item model predictions)")
    print(f"  Saved to {PROCESSED_DIR}")


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
