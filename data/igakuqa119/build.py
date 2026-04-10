"""
Build IgakuQA119 response matrix from the 119th Japanese Medical Licensing Exam.

Data source:
  - IgakuQA project: https://github.com/jungokasai/IgakuQA
  - 119th exam (Feb 2025): 400 items across 6 blocks (A-F)
  - Grading results for 5 models from demo evaluations
  - Correct answers from official answer key

Raw data layout (expected in raw/):
  questions/119{A..F}_json.json   - Question text and choices per block
  results/correct_answers.csv     - Official correct answers (問題番号, 解答)
  results/demo/*_grading_results.csv - Per-model grading results

Score format:
  - Binary 0/1: whether model answer matches correct answer

Outputs:
  - processed/response_matrix.csv: Items (rows) x models (columns), binary 0/1
  - processed/task_metadata.csv: Per-item metadata (question, answer, has_image)
  - processed/item_content.csv: Full item content (question text, choices)
"""

import csv
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/jungokasai/IgakuQA.git"


def download_raw():
    """Clone the IgakuQA repository for reference data."""
    igaku_dir = RAW_DIR / "IgakuQA"
    if igaku_dir.exists():
        print(f"  Already cloned: {igaku_dir}")
        return
    print("  Cloning IgakuQA repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(igaku_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}")
        print("  Continuing with local raw data...")


def main():
    print("IgakuQA119 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 0: Download reference repo
    download_raw()

    # Step 1: Load correct answers
    print("  Loading correct answers...")
    correct_answers_file = RAW_DIR / "results" / "correct_answers.csv"
    if not correct_answers_file.exists():
        print(f"  ERROR: correct_answers.csv not found at {correct_answers_file}")
        sys.exit(1)

    correct = {}
    with open(correct_answers_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column names are in Japanese: 問題番号 (question number), 解答 (answer)
            qnum = row.get("\u554f\u984c\u756a\u53f7", row.get("問題番号", ""))
            ans = row.get("\u89e3\u7b54", row.get("解答", ""))
            if qnum and ans:
                correct[qnum] = ans

    print(f"  Loaded {len(correct)} correct answers")

    # Step 2: Load questions from JSON files
    print("  Loading questions...")
    questions = {}
    questions_dir = RAW_DIR / "questions"
    if questions_dir.exists():
        for qfile in sorted(questions_dir.glob("*.json")):
            with open(qfile) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        qnum = item.get("number", "")
                        questions[qnum] = {
                            "question": item.get("question", ""),
                            "choices": json.dumps(item.get("choices", []), ensure_ascii=False),
                            "has_image": item.get("has_image", False),
                        }
        print(f"  Loaded {len(questions)} questions from JSON files")
    else:
        print("  WARNING: No questions directory found; metadata will have empty question text")

    # Step 3: Load grading results
    print("  Loading grading results...")
    predictions = defaultdict(dict)
    models = set()

    results_demo_dir = RAW_DIR / "results" / "demo"
    if results_demo_dir.exists():
        for gfile in sorted(results_demo_dir.glob("*_grading_results.csv")):
            with open(gfile, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    qnum = row.get("question_number", "")
                    model = row.get("model", "").split("/")[-1].replace(":latest", "")
                    is_correct = 1 if row.get("is_correct", "").lower() == "true" else 0
                    if qnum and model:
                        predictions[qnum][model] = is_correct
                        models.add(model)

    models = sorted(models)
    qnums = sorted(correct.keys())

    print(f"  Found {len(models)} models with grading results")
    print(f"  Models: {', '.join(models)}")

    # Step 4: Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "has_image", "language", "benchmark"])
        for qnum in qnums:
            q = questions.get(qnum, {})
            writer.writerow([
                qnum,
                q.get("question", ""),
                correct[qnum],
                q.get("has_image", ""),
                "Japanese",
                "IgakuQA119",
            ])

    # Step 5: Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "block", "question_text", "choices", "answer", "has_image"])
        for qnum in qnums:
            q = questions.get(qnum, {})
            # Extract block letter from question number (e.g., "119A1" -> "A")
            block = ""
            for ch in qnum:
                if ch.isalpha():
                    block = ch
                    break
            writer.writerow([
                qnum,
                block,
                q.get("question", ""),
                q.get("choices", "[]"),
                correct[qnum],
                q.get("has_image", False),
            ])

    # Step 6: Write response_matrix.csv
    print("  Writing response_matrix.csv...")
    with open(PROCESSED_DIR / "response_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"] + models)
        for qnum in qnums:
            row = [qnum]
            for model in models:
                row.append(predictions[qnum].get(model, ""))
            writer.writerow(row)

    # Step 7: Write model_summary.csv
    print("  Writing model_summary.csv...")
    with open(PROCESSED_DIR / "model_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_items", "num_correct", "accuracy"])
        for model in models:
            total = sum(1 for q in qnums if model in predictions[q])
            corr = sum(predictions[q].get(model, 0) for q in qnums if model in predictions[q])
            acc = corr / total if total > 0 else 0
            writer.writerow([model, total, corr, f"{acc:.4f}"])

    # Print summary
    print()
    print(f"  IgakuQA119: {len(qnums)} items x {len(models)} models")
    for model in models:
        total = sum(1 for q in qnums if model in predictions[q])
        corr = sum(predictions[q].get(model, 0) for q in qnums if model in predictions[q])
        acc = corr / total * 100 if total > 0 else 0
        print(f"    {model}: {corr}/{total} correct ({acc:.1f}%)")

    print(f"\n  Saved to {PROCESSED_DIR}")


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
