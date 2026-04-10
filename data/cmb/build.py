"""
Build CMB (Chinese Medical Benchmark) item metadata.

Data source:
  - GitHub: FreedomIntelligence/CMB
    https://github.com/FreedomIntelligence/CMB
  - Test set: CMB-Exam/CMB-test/CMB-test-choice-question-merge.json
  - Answers: data/CMB-test-choice-answer.json
  - 11,200 test items across 28 medical subcategories

CMB overview:
  - Chinese medical benchmark covering clinical and basic medical knowledge
  - Organized by exam_type, exam_class, exam_subject
  - Multiple-choice questions with detailed categorization

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answer, exam info)
  - processed/item_content.csv: Full item content with options
"""

import csv
import json
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

REPO_URL = "https://github.com/FreedomIntelligence/CMB.git"


def download_raw():
    """Clone the CMB repository."""
    repo_dir = RAW_DIR / "CMB"
    if repo_dir.exists():
        print(f"  Already cloned: {repo_dir}")
        return repo_dir
    print("  Cloning CMB repository...")
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
    print("CMB Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    repo_dir = download_raw()

    # Find test questions
    data_file = repo_dir / "CMB-Exam" / "CMB-test" / "CMB-test-choice-question-merge.json"
    if not data_file.exists():
        # Try alternate path
        data_file = repo_dir / "data" / "CMB-test-choice-question-merge.json"
    if not data_file.exists():
        print(f"  ERROR: Test data not found. Looked for:")
        print(f"    {repo_dir / 'CMB-Exam' / 'CMB-test' / 'CMB-test-choice-question-merge.json'}")
        sys.exit(1)

    print(f"  Loading questions from: {data_file}")
    with open(data_file) as f:
        questions = json.load(f)

    # Load answer key if available
    answer_file = repo_dir / "data" / "CMB-test-choice-answer.json"
    answers = {}
    if answer_file.exists():
        with open(answer_file) as f:
            ans_list = json.load(f)
            for a in ans_list:
                answers[a["id"]] = a["answer"]
        print(f"  Loaded {len(answers)} answers")

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "question", "answer", "exam_type", "exam_class",
            "exam_subject", "question_type", "language", "benchmark",
        ])
        for q in questions:
            writer.writerow([
                f"cmb_{q['id']:05d}",
                q.get("question", ""),
                answers.get(q["id"], ""),
                q.get("exam_type", ""),
                q.get("exam_class", ""),
                q.get("exam_subject", ""),
                q.get("question_type", ""),
                "Chinese",
                "CMB",
            ])

    # Write item_content.csv with full option text
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "question", "option_A", "option_B", "option_C",
            "option_D", "option_E", "answer", "exam_type", "exam_class",
            "exam_subject", "question_type",
        ])
        for q in questions:
            writer.writerow([
                f"cmb_{q['id']:05d}",
                q.get("question", ""),
                q.get("option", {}).get("A", "") if isinstance(q.get("option"), dict) else "",
                q.get("option", {}).get("B", "") if isinstance(q.get("option"), dict) else "",
                q.get("option", {}).get("C", "") if isinstance(q.get("option"), dict) else "",
                q.get("option", {}).get("D", "") if isinstance(q.get("option"), dict) else "",
                q.get("option", {}).get("E", "") if isinstance(q.get("option"), dict) else "",
                answers.get(q["id"], ""),
                q.get("exam_type", ""),
                q.get("exam_class", ""),
                q.get("exam_subject", ""),
                q.get("question_type", ""),
            ])

    print(f"\n  CMB: {len(questions)} test items (questions only, no per-item model predictions)")
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
