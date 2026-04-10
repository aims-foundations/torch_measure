"""
Build MMedBench item metadata from Multilingual Medical Benchmark.

Data source:
  - HuggingFace: Henrychur/MMedBench
    https://huggingface.co/datasets/Henrychur/MMedBench
  - Test items in 6 languages:
    Chinese (3,426), Japanese (199), Spanish (2,742),
    French (622), Russian (256), English (1,273)
  - JSONL files per language in Test/ directory

MMedBench overview:
  - Multilingual medical benchmark across 6 languages
  - Medical MCQ format with options and answer keys
  - Covers both clinical and basic medical knowledge

Note:
  - Questions only; no per-item model predictions yet.
  - When model predictions are collected, response_matrix.csv will be added.

Outputs:
  - processed/task_metadata.csv: Per-item metadata (question, answer, language)
  - processed/item_content.csv: Full item content with options
"""

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET = "Henrychur/MMedBench"


def download_raw():
    """Download MMedBench dataset from HuggingFace."""
    output_dir = RAW_DIR / "MMedBench"
    test_dir = output_dir / "Test"

    if test_dir.exists() and any(test_dir.glob("*.jsonl")):
        print(f"  Already downloaded: {test_dir}")
        return test_dir

    print(f"  Downloading {HF_DATASET} from HuggingFace...")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        # MMedBench has language-specific configs
        languages = ["Chinese", "Japanese", "Spanish", "French", "Russian", "English"]
        for lang in languages:
            try:
                ds = load_dataset(HF_DATASET, lang, split="test")
                lang_file = test_dir / f"{lang}.jsonl"
                with open(lang_file, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"    {lang}: {len(ds)} test items")
            except Exception as e:
                print(f"    WARNING: Could not download {lang}: {e}")
    except ImportError:
        print("  ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        sys.exit(1)

    return test_dir


def main():
    print("MMedBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    test_dir = download_raw()

    print("  Loading items from JSONL files...")
    all_items = []
    for lang_file in sorted(test_dir.glob("*.jsonl")):
        lang = lang_file.stem
        with open(lang_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                all_items.append({
                    "sample_id": f"mmedbench_{lang.lower()}_{i:05d}",
                    "question": item.get("question", ""),
                    "options": json.dumps(item.get("options", {}), ensure_ascii=False),
                    "answer": item.get("answer", ""),
                    "answer_idx": item.get("answer_idx", ""),
                    "meta_info": item.get("meta_info", ""),
                    "language": lang,
                })

    # Write task_metadata.csv
    print("  Writing task_metadata.csv...")
    with open(PROCESSED_DIR / "task_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "answer_idx", "meta_info", "language", "benchmark"])
        for item in all_items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["answer"],
                item["answer_idx"],
                item["meta_info"],
                item["language"],
                "MMedBench",
            ])

    # Write item_content.csv
    print("  Writing item_content.csv...")
    with open(PROCESSED_DIR / "item_content.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "options", "answer", "answer_idx", "meta_info", "language"])
        for item in all_items:
            writer.writerow([
                item["sample_id"],
                item["question"],
                item["options"],
                item["answer"],
                item["answer_idx"],
                item["meta_info"],
                item["language"],
            ])

    # Language breakdown
    lang_counts = Counter(item["language"] for item in all_items)
    print(f"\n  MMedBench: {len(all_items)} test items across {len(lang_counts)} languages:")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
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
