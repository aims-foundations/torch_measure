#!/usr/bin/env python3
"""
Build item collections from downloaded agriculture benchmark datasets.

Produces:
- processed/task_metadata.csv: Per-dataset summary statistics
- processed/item_content.csv: Per-item content with identifiers
- processed/summary_stats.csv: Aggregate statistics

Format mirrors other *eval_data directories (afrieval_data, legaleval_data).
"""

import os
import json
import csv
import glob
from collections import defaultdict

RAW_DIR = "/lfs/skampere1/0/sttruong/torch_measure/data/agreval_data/raw"
PROC_DIR = "/lfs/skampere1/0/sttruong/torch_measure/data/agreval_data/processed"


def load_jsonl(path):
    """Load a JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def load_json(path):
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_kisanvaani(raw_dir, items_out, meta_out):
    """Process KisanVaani agriculture QA dataset."""
    dataset_name = "kisanvaani"
    data_dir = os.path.join(raw_dir, "KisanVaani_agriculture-qa-english-only")

    for jsonl_file in glob.glob(os.path.join(data_dir, "*.jsonl")):
        data = load_jsonl(jsonl_file)
        split = os.path.basename(jsonl_file).replace(".jsonl", "")

        for i, item in enumerate(data):
            q = item.get("question", "")
            a = item.get("answers", "")
            item_id = f"{dataset_name}_{split}_{i}"
            content = f"Q: {q}\nA: {a}"

            items_out.append({
                "item_id": item_id,
                "content": q,
            })
            meta_out.append({
                "item_id": item_id,
                "text": q,
                "label": a[:200] if a else "",
                "task_type": "agricultural_qa",
                "language": "eng",
                "language_variety": "English",
                "source_dataset": dataset_name,
                "split": split,
                "n_chars": len(q),
            })

        print(f"  KisanVaani: {len(data)} items from {split}")
    return len(meta_out)


def process_bhashabench_krishi(raw_dir, items_out, meta_out):
    """Process BhashaBench-Krishi agricultural MCQ benchmark."""
    dataset_name = "bhashabench_krishi"
    data_dir = os.path.join(raw_dir, "bharatgenai_BhashaBench-Krishi")

    if not os.path.exists(data_dir):
        print("  BhashaBench-Krishi: NOT DOWNLOADED (gated dataset)")
        return 0

    count = 0
    for jsonl_file in glob.glob(os.path.join(data_dir, "*.jsonl")):
        data = load_jsonl(jsonl_file)
        split = os.path.basename(jsonl_file).replace(".jsonl", "")

        for i, item in enumerate(data):
            q = item.get("question", item.get("Question", ""))
            lang = "hin" if "hindi" in split.lower() else "eng"
            lang_name = "Hindi" if lang == "hin" else "English"
            answer = item.get("answer", item.get("Answer", ""))
            domain = item.get("subject", item.get("domain", ""))
            difficulty = item.get("difficulty", "")

            item_id = f"{dataset_name}_{lang}_{split}_{i}"

            items_out.append({
                "item_id": item_id,
                "content": q,
            })
            meta_out.append({
                "item_id": item_id,
                "text": q,
                "label": str(answer),
                "task_type": "agricultural_mcq",
                "language": lang,
                "language_variety": lang_name,
                "source_dataset": dataset_name,
                "split": split,
                "n_chars": len(q),
                "domain": domain,
                "difficulty": difficulty,
            })
            count += 1

    print(f"  BhashaBench-Krishi: {count} items")
    return count


def process_mirage(raw_dir, items_out, meta_out):
    """Process MIRAGE agricultural consultation benchmark."""
    dataset_name = "mirage"
    data_dir = os.path.join(raw_dir, "MIRAGE-Benchmark_MIRAGE")

    if not os.path.exists(data_dir):
        print("  MIRAGE: NOT DOWNLOADED")
        return 0

    count = 0
    for jsonl_file in glob.glob(os.path.join(data_dir, "*.jsonl")):
        data = load_jsonl(jsonl_file)
        split = os.path.basename(jsonl_file).replace(".jsonl", "")

        for i, item in enumerate(data):
            q = item.get("question", item.get("dialog_context", ""))
            a = item.get("answer", item.get("utterance", ""))
            category = item.get("category", "")
            entity_type = item.get("entity_type", "")

            item_id = f"{dataset_name}_{split}_{i}"

            items_out.append({
                "item_id": item_id,
                "content": q[:500] if q else "",
            })
            meta_out.append({
                "item_id": item_id,
                "text": q[:500] if q else "",
                "label": a[:200] if a else "",
                "task_type": "agricultural_consultation",
                "language": "eng",
                "language_variety": "English",
                "source_dataset": dataset_name,
                "split": split,
                "n_chars": len(q) if q else 0,
                "category": category,
                "entity_type": entity_type,
            })
            count += 1

    print(f"  MIRAGE: {count} items")
    return count


def process_plantvillagevqa(raw_dir, items_out, meta_out):
    """Process PlantVillageVQA benchmark."""
    dataset_name = "plantvillagevqa"
    data_dir = os.path.join(raw_dir, "SyedNazmusSakib_PlantVillageVQA")

    if not os.path.exists(data_dir):
        print("  PlantVillageVQA: NOT DOWNLOADED")
        return 0

    count = 0
    for jsonl_file in glob.glob(os.path.join(data_dir, "*.jsonl")):
        data = load_jsonl(jsonl_file)
        split = os.path.basename(jsonl_file).replace(".jsonl", "")

        for i, item in enumerate(data):
            q = item.get("question", "")
            a = item.get("answer", "")
            qtype = item.get("question_type", "")

            item_id = f"{dataset_name}_{split}_{i}"

            items_out.append({
                "item_id": item_id,
                "content": q,
            })
            meta_out.append({
                "item_id": item_id,
                "text": q,
                "label": a[:200] if a else "",
                "task_type": "plant_disease_vqa",
                "language": "eng",
                "language_variety": "English",
                "source_dataset": dataset_name,
                "split": split,
                "n_chars": len(q),
                "question_type": qtype,
            })
            count += 1

    print(f"  PlantVillageVQA: {count} items")
    return count


def process_crop_dataset(raw_dir, items_out, meta_out):
    """Process CROP-dataset (Chinese/English crop science)."""
    dataset_name = "crop_dataset"
    data_dir = os.path.join(raw_dir, "AI4Agr_CROP-dataset")

    if not os.path.exists(data_dir):
        print("  CROP-dataset: NOT DOWNLOADED")
        return 0

    count = 0
    for jsonl_file in glob.glob(os.path.join(data_dir, "*.jsonl")):
        data = load_jsonl(jsonl_file)
        split = os.path.basename(jsonl_file).replace(".jsonl", "")

        for i, item in enumerate(data):
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            # Detect language
            lang = "zho" if any('\u4e00' <= c <= '\u9fff' for c in instruction[:50]) else "eng"
            lang_name = "Chinese" if lang == "zho" else "English"

            item_id = f"{dataset_name}_{lang}_{split}_{i}"

            items_out.append({
                "item_id": item_id,
                "content": instruction[:500],
            })
            meta_out.append({
                "item_id": item_id,
                "text": instruction[:500],
                "label": output[:200] if output else "",
                "task_type": "crop_science_qa",
                "language": lang,
                "language_variety": lang_name,
                "source_dataset": dataset_name,
                "split": split,
                "n_chars": len(instruction),
            })
            count += 1

    print(f"  CROP-dataset: {count} items")
    return count


def build_summary_stats(meta_items):
    """Compute summary statistics per dataset+language."""
    groups = defaultdict(list)
    for item in meta_items:
        key = (item["source_dataset"], item["language"], item.get("language_variety", ""))
        groups[key].append(item)

    stats = []
    for (dataset, lang, lang_name), items in sorted(groups.items()):
        n_total = len(items)
        splits = defaultdict(int)
        for it in items:
            splits[it["split"]] += 1

        labels = set(it["label"] for it in items)
        char_lens = [it["n_chars"] for it in items]
        avg_chars = sum(char_lens) / len(char_lens) if char_lens else 0
        import statistics
        median_chars = statistics.median(char_lens) if char_lens else 0

        stats.append({
            "source_dataset": dataset,
            "language": lang,
            "language_name": lang_name,
            "n_items": n_total,
            "n_train": splits.get("train", 0),
            "n_validation": splits.get("dev", splits.get("validation", 0)),
            "n_test": splits.get("test", 0),
            "task_type": items[0]["task_type"],
            "n_labels": len(labels),
            "avg_n_chars": avg_chars,
            "median_n_chars": median_chars,
        })

    return stats


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    items_out = []
    meta_out = []

    print("Processing agriculture benchmark datasets...")
    print("="*60)

    process_kisanvaani(RAW_DIR, items_out, meta_out)
    process_bhashabench_krishi(RAW_DIR, items_out, meta_out)
    process_mirage(RAW_DIR, items_out, meta_out)
    process_plantvillagevqa(RAW_DIR, items_out, meta_out)
    process_crop_dataset(RAW_DIR, items_out, meta_out)

    print(f"\nTotal items collected: {len(items_out)}")

    if not items_out:
        print("No items to process. Run 01_download_agriculture_benchmarks.py first.")
        return

    # Write item_content.csv
    ic_path = os.path.join(PROC_DIR, "item_content.csv")
    with open(ic_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "content"])
        writer.writeheader()
        for item in items_out:
            writer.writerow(item)
    print(f"Wrote {ic_path}")

    # Write task_metadata.csv
    base_fields = ["item_id", "text", "label", "task_type", "language",
                    "language_variety", "source_dataset", "split", "n_chars"]
    tm_path = os.path.join(PROC_DIR, "task_metadata.csv")
    with open(tm_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction='ignore')
        writer.writeheader()
        for item in meta_out:
            writer.writerow(item)
    print(f"Wrote {tm_path}")

    # Write summary_stats.csv
    stats = build_summary_stats(meta_out)
    ss_path = os.path.join(PROC_DIR, "summary_stats.csv")
    with open(ss_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_dataset", "language", "language_name", "n_items",
            "n_train", "n_validation", "n_test", "task_type",
            "n_labels", "avg_n_chars", "median_n_chars"
        ])
        writer.writeheader()
        for row in stats:
            writer.writerow(row)
    print(f"Wrote {ss_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
