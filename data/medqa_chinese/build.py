"""
Build MedQA Chinese item metadata from Chinese medical licensing exam.

Data source:
  - GitHub: jind11/MedQA
    https://github.com/jind11/MedQA
  - Download: https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw
  - Mainland China 4-option MCQ test set: 3,426 items
  - JSONL format with question, options, answer, answer_idx, meta_info

MedQA overview:
  - Chinese National Medical Licensing Examination (CNMLE)
  - 4-option multiple choice (Mainland China version)
  - Also has Taiwan and US (USMLE) versions

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answer, meta_info)
  - processed/item_content.csv: Full item content with all options
"""

INFO = {
    'description': 'Build MedQA Chinese item metadata from Chinese medical licensing exam',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2009.13081',
    'data_source_url': 'https://huggingface.co/datasets/bigbio/med_qa',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{jin2020diseasedoespatienthave,
      title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams}, 
      author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
      year={2020},
      eprint={2009.13081},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2009.13081}, 
}""",
    'tags': ['pending'],
}


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

# MedQA data must be downloaded separately (Google Drive)
# GitHub repo for reference: https://github.com/jind11/MedQA
# Direct download requires extracting to raw/medqa/data_clean/questions/Mainland/4_options/test.jsonl
HF_DATASET = "bigbio/med_qa"  # Also available on HuggingFace


def download_raw():
    """Download MedQA Chinese data.

    The original data requires a Google Drive download. We try HuggingFace first.
    """
    output_dir = RAW_DIR / "medqa"
    test_file = output_dir / "test.jsonl"

    if test_file.exists():
        print(f"  Already downloaded: {test_file}")
        return test_file

    # Also check extracted path
    extracted = output_dir / "data_clean" / "questions" / "Mainland" / "4_options" / "test.jsonl"
    if extracted.exists():
        return extracted

    print("  Downloading MedQA Chinese from HuggingFace...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        # Try the bigbio version which has Chinese subset
        ds = load_dataset("bigbio/med_qa", "med_qa_zh_4options_bigbio_qa", split="test", trust_remote_code=True)
        with open(test_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Downloaded {len(ds)} test items")
        return test_file
    except Exception:
        pass

    try:
        from datasets import load_dataset

        ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
        print("  WARNING: Only USMLE (English) version available on HuggingFace")
        print("  For Chinese version, download from: https://github.com/jind11/MedQA")
        print("  Extract to: raw/medqa/data_clean/questions/Mainland/4_options/test.jsonl")
    except Exception:
        pass

    print("  ERROR: Could not download MedQA Chinese data.")
    print("  Please download manually from https://github.com/jind11/MedQA")
    print(f"  and extract to: {output_dir}/data_clean/questions/Mainland/4_options/test.jsonl")
    sys.exit(1)


def main():
    print("MedQA Chinese Response Matrix Builder")
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
                "sample_id": f"medqa_zh_{i:05d}",
                "question": item.get("question", ""),
                "options": json.dumps(item.get("options", {}), ensure_ascii=False),
                "answer": item.get("answer", ""),
                "answer_idx": item.get("answer_idx", ""),
                "meta_info": item.get("meta_info", ""),
            })

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "answer_idx", "meta_info", "language", "benchmark"])
        for item in items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["answer_idx"],
                item["meta_info"],
                "Chinese",
                "MedQA-Chinese",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "options", "answer", "answer_idx", "meta_info"])
        for item in items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["options"],
                item["answer"],
                item["answer_idx"],
                item["meta_info"],
            ])

    print(f"\n  MedQA Chinese: {len(items)} test items (questions only, no per-item model predictions)")
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
