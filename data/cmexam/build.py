"""
Build CMExam item metadata from the Chinese National Medical Licensing Examination.

Data source:
  - HuggingFace: williamliu/CMExam
    https://huggingface.co/datasets/williamliu/CMExam
  - 6,811 test items with questions, answers, and explanations
  - Each item has: Question, Answer, Explanation fields

CMExam overview:
  - Chinese National Medical Licensing Examination dataset
  - Multiple-choice questions with detailed explanations
  - Covers broad medical knowledge domains

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answer, explanation)
  - processed/item_content.csv: Full item content
"""

import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET = "williamliu/CMExam"


def download_raw():
    """Download CMExam dataset from HuggingFace."""
    output_dir = RAW_DIR / "CMExam"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / "test.json"

    if test_file.exists():
        print(f"  Already downloaded: {test_file}")
        return test_file

    print(f"  Downloading {HF_DATASET} from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split="test")
        # Save as JSONL
        with open(test_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Downloaded {len(ds)} test items")
    except ImportError:
        print("  ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        sys.exit(1)

    return test_file


def main():
    print("CMExam Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    test_file = download_raw()

    print(f"  Loading items from: {test_file}")
    items = []
    with open(test_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            items.append({
                "sample_id": f"cmexam_{i:05d}",
                "question": item.get("Question", item.get("question", "")),
                "answer": item.get("Answer", item.get("answer", "")),
                "explanation": (item.get("Explanation", item.get("explanation", "")) or "")[:200],
            })

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "explanation_preview", "language", "benchmark"])
        for item in items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["explanation"],
                "Chinese",
                "CMExam",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "answer", "explanation_preview"])
        for item in items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["explanation"],
            ])

    print(f"\n  CMExam: {len(items)} test items (questions only, no per-item model predictions)")
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
