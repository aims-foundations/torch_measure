#!/usr/bin/env python3
"""
Download agriculture benchmark datasets for validity analysis.

This script downloads publicly available agriculture AI benchmark datasets
and organizes them for item-level analysis. Focuses on datasets with:
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

import os
import json
import subprocess
import sys

BASE_DIR = "/lfs/skampere1/0/sttruong/torch_measure/data/agreval_data/raw"

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


def download_all():
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


if __name__ == "__main__":
    download_all()
