#!/usr/bin/env python3
"""
Download agriculture benchmark datasets and build item collections.

Downloads publicly available agriculture AI benchmark datasets from HuggingFace,
organizes them for item-level analysis, and produces:
- processed/task_metadata.csv: Per-dataset summary statistics
- processed/item_content.csv: Per-item content with identifiers
- processed/summary_stats.csv: Aggregate statistics

Format mirrors other *eval_data directories (afrieval_data, legaleval_data).

Focuses on datasets with:
1. Per-item evaluation questions (MCQ, VQA, NER)
2. Global South coverage (India, Africa, Southeast Asia)
3. Text-based benchmarks suitable for LLM evaluation

DATASET INVENTORY (ranked by per-item model prediction availability):
===========================================================================

TIER 1: Per-item QA benchmarks downloadable from HuggingFace
-------------------------------------------------------------
1. BhashaBench-Krishi (BBK) - Indian agricultural MCQ benchmark
   - 15,405 MCQs from 55+ government agricultural exams
   - English (12,648) + Hindi (2,757)
   - 25+ agricultural domains, 270+ topics
   - 29+ models evaluated (GPT-4o, Qwen3-235B, etc.)
   - GATED dataset: requires HF login + agreement
   - HF: bharatgenai/BhashaBench-Krishi

2. KisanVaani Agriculture QA - Indian agricultural Q&A
   - 22,615 QA pairs from Kisan Call Center (Uganda-focused)
   - Open-ended answers, crop management, pest/disease
   - HF: KisanVaani/agriculture-qa-english-only
   - Apache 2.0, no per-item model predictions

3. AgriCoT / AgroCoT - Chain-of-thought VLM agriculture benchmark
   - 4,535 curated samples with CoT reasoning annotations
   - 5 dimensions: object detection, quantitative analysis,
     disease monitoring, spatial understanding, environmental mgmt
   - 26 VLMs evaluated
   - HF: wenyb/AgriCoT (4,821 rows, 4 GB with images)

4. AgroBench - Agricultural VLM benchmark (ICCV 2025)
   - 4,342 examples, 682 disease categories, 203 crop types
   - 7 VQA tasks annotated by expert agronomists
   - GATED dataset: requires agreement
   - HF: risashinoda/AgroBench

5. AgroMind - Agricultural remote sensing benchmark
   - 28,482 QA pairs from 20,850 images
   - 13 task types across 4 dimensions
   - 24 models evaluated (20 open-source + 4 closed)
   - HF: AgroMind/AgroMind (CC BY-SA 4.0)

6. MIRAGE - Agricultural expert consultation benchmark
   - 35,000+ authentic consultations with images
   - 7,000+ unique biological entities
   - Single-turn (25.7k) + Multi-turn (5.6k) splits
   - HF: MIRAGE-Benchmark/MIRAGE (CC BY-SA 4.0)

7. PlantVillageVQA - Plant disease VQA benchmark
   - 193,609 QA pairs, 55,448 leaf images
   - 14 crops, 38 diseases, 9 question categories
   - 3 cognitive complexity levels
   - HF: SyedNazmusSakib/PlantVillageVQA (CC BY 4.0)

8. AgMMU - Agricultural Multimodal Understanding
   - 746 MCQs + 746 open-ended questions
   - 57,079-item knowledge base from USDA Extension
   - 5 agricultural question types
   - HF: AgMMU/AgMMU_v1

9. CROP-dataset - Crop science instruction fine-tuning
   - 210,000+ QA pairs in Chinese + English
   - Rice, corn/maize diseases, pest management
   - HF: AI4Agr/CROP-dataset (CC BY-NC 4.0)

TIER 2: Benchmark datasets on GitHub (no per-item predictions)
--------------------------------------------------------------
10. AgriBench (ECCV 2024) - MM-LLM agriculture benchmark
    - 1,784 images from MM-LUCAS dataset (27 EU countries)
    - 5 task complexity levels
    - NO per-item results published yet (TODO on their roadmap)
    - GitHub: Yutong-Zhou-cv/AgriBench
    - Data: Google Drive

11. CDDMBench - Crop Disease Diagnosis Multimodal benchmark
    - 137,000 images + 1M QA pairs
    - Must run inference yourself
    - GitHub: UnicomAI/UnicomBenchmark/tree/main/CDDMBench

12. AgCNER - Chinese Agricultural NER for diseases/pests
    - 13 categories, 206,992 entities, 66,553 samples
    - Aggregate F1 results only (93.58% BiLSTM-CRF)
    - GitHub: guojson/AgCNER

13. CropHarvest - Global satellite crop classification
    - 95,186 datapoints, 33,205 with multiclass labels
    - 12 Sub-Saharan African countries
    - GitHub: nasaharvest/cropharvest

14. Fields of the World (AAAI 2025) - Field boundary segmentation
    - 70,462 samples across 24 countries (4 continents)
    - GitHub: fieldsoftheworld/ftw-baselines

TIER 3: Specialized / Emerging datasets
----------------------------------------
15. AgriNER (contributions-ner-agri) - Agricultural scholarly NER
    - 15,261 entity annotations from 5,500 paper titles
    - 7 contribution-centric entities
    - GitHub: jd-coderepos/contributions-ner-agri

16. LeafNet/LeafBench - Plant disease VLM benchmark
    - 186,000+ images, 22 crops, 62 diseases
    - 13,950 QA pairs, 12 VLMs evaluated
    - Paper: arxiv.org/abs/2602.13662

17. FoodSky / CDE Benchmark - Chinese food domain LLM eval
    - Chef + Dietetic examination benchmark
    - FoodEarth mini: 20K instances on Zenodo
    - GitHub: LanceZPF/FoodSky

18. NGQA - Nutritional Graph QA (ACL 2025)
    - Health-aware nutritional reasoning
    - NHANES + FNDDS data

19. SustainBench - Crop yield prediction
    - US (857 counties), Argentina (135), Brazil (32)
    - GitHub: sustainlab-group/sustainbench

20. Lacuna Fund Africa Crop Disease Images
    - 127,046 images + 39,300 spectral data points
    - Cassava, maize, beans, bananas, cocoa
    - Harvard Dataverse

OPENCOMPASS DOMAIN BENCHMARKS (per-item prediction potential):
=============================================================
OpenCompass stores per-item predictions in:
  output/.../predictions/<model>/<dataset>.json

Domain benchmarks in OpenCompass configs/datasets:
- FinanceIQ: Chinese financial exam MCQs
- MedBench: Chinese medical benchmark (300,901 questions)
- MedQA: Medical QA
- MedCalc_Bench: Medical calculation
- MedXpertQA: Medical expert QA
- Medbullets: Medical board questions
- LawBench: Chinese legal benchmark
- CARDBiomedBench: Biomedical NLP
- ClinicBench: Clinical NLP
- PubMedQA: Biomedical literature QA

NO agriculture benchmarks in OpenCompass as of March 2026.
"""

import sys
from pathlib import Path
import os
import json
import csv
import glob
import statistics
from collections import defaultdict

BASE_DIR = "/lfs/skampere1/0/sttruong/torch_measure/data/agreval_data/raw"
PROC_DIR = "/lfs/skampere1/0/sttruong/torch_measure/data/agreval_data/processed"


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_hf_dataset(repo_id, output_dir, subset=None, split=None, token_required=False):
    """Download a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets required")
        return False

    dest = os.path.join(output_dir, repo_id.replace("/", "_"))
    os.makedirs(dest, exist_ok=True)

    try:
        kwargs = {}
        if subset:
            kwargs["name"] = subset
        if split:
            kwargs["split"] = split
        if token_required:
            kwargs["token"] = True

        print(f"Downloading {repo_id}...")
        ds = load_dataset(repo_id, **kwargs)

        # Save as JSON Lines
        if hasattr(ds, 'keys'):
            for split_name in ds.keys():
                out_path = os.path.join(dest, f"{split_name}.jsonl")
                ds[split_name].to_json(out_path)
                print(f"  Saved {split_name}: {len(ds[split_name])} rows -> {out_path}")
        else:
            out_path = os.path.join(dest, "data.jsonl")
            ds.to_json(out_path)
            print(f"  Saved: {len(ds)} rows -> {out_path}")
        return True

    except Exception as e:
        print(f"  ERROR downloading {repo_id}: {e}")
        return False


def download():
    """Download all publicly available agriculture benchmarks."""
    os.makedirs(BASE_DIR, exist_ok=True)

    results = {}

    # ---- OPEN DATASETS (no gating) ----

    # 1. KisanVaani Agriculture QA (Apache 2.0, open)
    results["KisanVaani"] = download_hf_dataset(
        "KisanVaani/agriculture-qa-english-only",
        BASE_DIR,
        split="train"
    )

    # 2. AgroMind (CC BY-SA 4.0, open)
    results["AgroMind"] = download_hf_dataset(
        "AgroMind/AgroMind",
        BASE_DIR
    )

    # 3. MIRAGE (CC BY-SA 4.0, open)
    for config in ["MMST_Standard", "MMST_Contextual", "MMMT_Direct", "MMMT_Decomp"]:
        results[f"MIRAGE_{config}"] = download_hf_dataset(
            "MIRAGE-Benchmark/MIRAGE",
            BASE_DIR,
            subset=config
        )

    # 4. PlantVillageVQA (CC BY 4.0, open)
    results["PlantVillageVQA"] = download_hf_dataset(
        "SyedNazmusSakib/PlantVillageVQA",
        BASE_DIR
    )

    # 5. CROP-dataset (CC BY-NC 4.0, open)
    results["CROP_dataset"] = download_hf_dataset(
        "AI4Agr/CROP-dataset",
        BASE_DIR
    )

    # ---- GATED DATASETS (require HF token + agreement) ----

    # 6. BhashaBench-Krishi (gated, CC BY 4.0)
    # Requires accepting terms at huggingface.co/datasets/bharatgenai/BhashaBench-Krishi
    for lang in ["English", "Hindi"]:
        results[f"BhashaBench_Krishi_{lang}"] = download_hf_dataset(
            "bharatgenai/BhashaBench-Krishi",
            BASE_DIR,
            subset=lang,
            split="test",
            token_required=True
        )

    # 7. AgroBench (gated)
    results["AgroBench"] = download_hf_dataset(
        "risashinoda/AgroBench",
        BASE_DIR,
        token_required=True
    )

    # 8. AgriCoT (likely open, large with images ~4GB)
    results["AgriCoT"] = download_hf_dataset(
        "wenyb/AgriCoT",
        BASE_DIR
    )

    # 9. AgMMU (open)
    results["AgMMU"] = download_hf_dataset(
        "AgMMU/AgMMU_v1",
        BASE_DIR
    )

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")


# ---------------------------------------------------------------------------
# Build logic
# ---------------------------------------------------------------------------

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
    # Step 1: Download datasets
    download()

    # Step 2: Build item collections
    os.makedirs(PROC_DIR, exist_ok=True)

    items_out = []
    meta_out = []

    print("Processing agriculture benchmark datasets...")
    print("="*60)

    process_kisanvaani(BASE_DIR, items_out, meta_out)
    process_bhashabench_krishi(BASE_DIR, items_out, meta_out)
    process_mirage(BASE_DIR, items_out, meta_out)
    process_plantvillagevqa(BASE_DIR, items_out, meta_out)
    process_crop_dataset(BASE_DIR, items_out, meta_out)

    print(f"\nTotal items collected: {len(items_out)}")

    if not items_out:
        print("No items to process. Check download step for errors.")
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
