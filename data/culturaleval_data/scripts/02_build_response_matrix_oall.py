#!/usr/bin/env python3
"""
Build response matrices for cultural eval benchmarks using the Open Arabic LLM
Leaderboard (OALL) per-item evaluation details.

Data source:
  OALL detail datasets on HuggingFace (huggingface.co/OALL)
  - Per-item evaluation results for 900+ models
  - Each model's results are in: OALL/details_{org}__{model}
  - Configs include: community_arabic_exams_0, community_arabic_mmlu_*_0,
    community_acva_*_0 (Arabic Culture & Values Assessment)
  - Each row has: gold_index, metrics (acc_norm), predictions, etc.

The OALL Arabic MMLU is a ChatGPT-translated version of English MMLU (NOT the
native ArabicMMLU from MBZUAI). We build it as a separate response matrix.

The OALL Arabic EXAMS is a curated Arabic-only subset (537 items) from EXAMS.

Outputs (in ../processed/):
  - response_matrix_oall_arabic_exams.csv  : models x items matrix (binary 0/1)
  - response_matrix_oall_arabic_mmlu.csv   : models x items matrix (binary 0/1)
  - model_summary_oall.csv                 : per-model aggregate statistics

All paths use Path(__file__).resolve().parent.parent
"""

import sys
import warnings
import time
from pathlib import Path
from collections import defaultdict

import json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Arabic MMLU subjects (same as English MMLU, translated to Arabic)
# ──────────────────────────────────────────────────────────────────────
ARABIC_MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def _extract_acc_norm(row):
    """Robustly extract acc_norm from a row's metrics, handling dict or JSON str."""
    metrics = row.get("metrics") if isinstance(row, dict) else row["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    if isinstance(metrics, dict):
        return int(metrics.get("acc_norm", 0))
    return 0


def discover_oall_models(max_models=100):
    """Discover OALL detail datasets (models evaluated on Arabic benchmarks)."""
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Discovering OALL detail datasets (limit={max_models})...")
    datasets = list(api.list_datasets(author="OALL", search="details"))
    detail_datasets = [
        ds.id for ds in datasets
        if ds.id.startswith("OALL/details_")
    ]
    print(f"  Found {len(detail_datasets)} total detail datasets")

    # Filter to a manageable subset -- prioritize well-known models
    priority_patterns = [
        "meta-llama__Meta-Llama-3",
        "meta-llama__Llama-2",
        "mistralai__Mistral",
        "mistralai__Mixtral",
        "Qwen__Qwen",
        "google__gemma",
        "CohereForAI__c4ai",
        "CohereForAI__aya",
        "core42__jais",
        "01-ai__Yi",
        "microsoft__Phi",
        "tiiuae__falcon",
        "FreedomIntelligence__AceGPT",
        "bigscience__bloom",
        "internlm",
        "deepseek",
    ]

    priority = []
    others = []
    for ds_id in detail_datasets:
        model_part = ds_id.replace("OALL/details_", "")
        is_priority = any(pat in model_part for pat in priority_patterns)
        if is_priority:
            priority.append(ds_id)
        else:
            others.append(ds_id)

    selected = priority[:max_models]
    remaining = max_models - len(selected)
    if remaining > 0:
        selected.extend(others[:remaining])

    print(f"  Selected {len(selected)} models ({len(priority)} priority)")
    return selected


def load_oall_details(dataset_id, config, max_retries=2):
    """Load per-item details from an OALL detail dataset for a given config."""
    from datasets import load_dataset

    for attempt in range(max_retries + 1):
        try:
            ds = load_dataset(dataset_id, config, split="latest")
            return ds
        except Exception:
            try:
                # Some datasets use different split names
                ds = load_dataset(dataset_id, config)
                splits = list(ds.keys())
                if "latest" in splits:
                    return ds["latest"]
                elif splits:
                    return ds[splits[-1]]
            except Exception as e:
                if attempt == max_retries:
                    return None
                time.sleep(1)
    return None


def extract_model_name(dataset_id):
    """Extract a clean model name from OALL dataset ID."""
    name = dataset_id.replace("OALL/details_", "")
    name = name.replace("__", "/")
    return name


def build_arabic_exams_matrix(model_datasets, max_models=100):
    """Build response matrix for Arabic EXAMS from OALL per-item details."""
    print("\n" + "=" * 70)
    print("Building Arabic EXAMS Response Matrix")
    print("=" * 70)

    config = "community_arabic_exams_0"
    n_items = None
    model_responses = {}
    model_accuracies = {}

    for i, ds_id in enumerate(model_datasets[:max_models]):
        model_name = extract_model_name(ds_id)
        print(f"  [{i+1}/{min(len(model_datasets), max_models)}] {model_name}...", end=" ")

        ds = load_oall_details(ds_id, config)
        if ds is None:
            print("SKIP (load failed)")
            continue

        if n_items is None:
            n_items = len(ds)
            print(f"({n_items} items)", end=" ")
        elif len(ds) != n_items:
            print(f"SKIP (item count mismatch: {len(ds)} vs {n_items})")
            continue

        # Extract binary correct/incorrect per item
        try:
            responses = []
            for row in ds:
                responses.append(_extract_acc_norm(row))
        except Exception as e:
            print(f"SKIP (extraction error: {e})")
            continue

        model_responses[model_name] = responses
        model_accuracies[model_name] = np.mean(responses)
        print(f"acc={model_accuracies[model_name]:.3f}")

    if not model_responses:
        print("  No models loaded successfully!")
        return None, None

    # Build item IDs
    item_ids = [f"oall_arabic_exams_{i:04d}" for i in range(n_items)]

    # Build response matrix: rows=models, columns=items
    rm_df = pd.DataFrame(model_responses, index=item_ids).T
    rm_df.index.name = "model"

    # Build model summary
    summary_rows = []
    for model_name, acc in model_accuracies.items():
        summary_rows.append({
            "model": model_name,
            "source": "OALL",
            "benchmark": "arabic_exams",
            "overall_accuracy": acc,
            "n_items": n_items,
        })
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n  Response matrix shape: {rm_df.shape}")
    print(f"  Models: {rm_df.shape[0]}")
    print(f"  Items: {rm_df.shape[1]}")
    print(f"  Mean accuracy: {summary_df['overall_accuracy'].mean():.3f}")

    return rm_df, summary_df


def build_arabic_mmlu_matrix(model_datasets, max_models=100):
    """Build response matrix for Arabic (translated) MMLU from OALL details."""
    print("\n" + "=" * 70)
    print("Building Arabic MMLU (Translated) Response Matrix")
    print("=" * 70)

    # First, determine item counts per subject
    subject_items = {}
    subject_configs = {
        subj: f"community_arabic_mmlu_{subj}_0"
        for subj in ARABIC_MMLU_SUBJECTS
    }

    model_responses = {}
    model_accuracies = {}
    total_items = None

    for i, ds_id in enumerate(model_datasets[:max_models]):
        model_name = extract_model_name(ds_id)
        print(f"  [{i+1}/{min(len(model_datasets), max_models)}] {model_name}...", end=" ")

        all_responses = []
        all_correct = 0
        all_total = 0
        success = True

        for subj in ARABIC_MMLU_SUBJECTS:
            config = subject_configs[subj]
            ds = load_oall_details(ds_id, config)
            if ds is None:
                success = False
                break

            if subj not in subject_items:
                subject_items[subj] = len(ds)

            try:
                for row in ds:
                    acc = _extract_acc_norm(row)
                    all_responses.append(acc)
                    all_correct += acc
                    all_total += 1
            except Exception as e:
                print(f"SKIP (extraction error in {subj}: {e})")
                success = False
                break

        if not success or all_total == 0:
            print("SKIP (load failed for some subjects)")
            continue

        if total_items is None:
            total_items = all_total
        elif all_total != total_items:
            print(f"SKIP (item count mismatch: {all_total} vs {total_items})")
            continue

        model_responses[model_name] = all_responses
        model_accuracies[model_name] = all_correct / all_total
        print(f"acc={model_accuracies[model_name]:.3f} ({all_total} items)")

    if not model_responses:
        print("  No models loaded successfully!")
        return None, None

    # Build item IDs with subject prefixes
    item_ids = []
    for subj in ARABIC_MMLU_SUBJECTS:
        n = subject_items.get(subj, 0)
        for j in range(n):
            item_ids.append(f"oall_ar_mmlu_{subj}_{j:04d}")

    # Build response matrix
    rm_df = pd.DataFrame(model_responses, index=item_ids).T
    rm_df.index.name = "model"

    # Build model summary
    summary_rows = []
    for model_name, acc in model_accuracies.items():
        summary_rows.append({
            "model": model_name,
            "source": "OALL",
            "benchmark": "arabic_mmlu_translated",
            "overall_accuracy": acc,
            "n_items": total_items,
        })
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n  Response matrix shape: {rm_df.shape}")
    print(f"  Models: {rm_df.shape[0]}")
    print(f"  Items: {rm_df.shape[1]}")
    print(f"  Subjects: {len(subject_items)}")
    print(f"  Mean accuracy: {summary_df['overall_accuracy'].mean():.3f}")

    return rm_df, summary_df


def main():
    print("=" * 70)
    print("Cultural Evaluation Response Matrix Builder (OALL Source)")
    print("=" * 70)

    # Parse arguments
    max_models = 50  # Default: top 50 models
    if len(sys.argv) > 1:
        try:
            max_models = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Max models to process: {max_models}")

    # Discover models
    model_datasets = discover_oall_models(max_models=max_models)

    # Build Arabic EXAMS response matrix
    exams_rm, exams_summary = build_arabic_exams_matrix(
        model_datasets, max_models=max_models
    )
    if exams_rm is not None:
        path = PROCESSED_DIR / "response_matrix_oall_arabic_exams.csv"
        exams_rm.to_csv(path)
        print(f"\n  Saved: {path}")

    # Build Arabic MMLU (translated) response matrix
    mmlu_rm, mmlu_summary = build_arabic_mmlu_matrix(
        model_datasets, max_models=max_models
    )
    if mmlu_rm is not None:
        path = PROCESSED_DIR / "response_matrix_oall_arabic_mmlu.csv"
        mmlu_rm.to_csv(path)
        print(f"\n  Saved: {path}")

    # Combine model summaries
    summaries = []
    if exams_summary is not None:
        summaries.append(exams_summary)
    if mmlu_summary is not None:
        summaries.append(mmlu_summary)

    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        summary_path = PROCESSED_DIR / "model_summary_oall.csv"
        all_summary.to_csv(summary_path, index=False)
        print(f"\nSaved model_summary_oall.csv: {summary_path}")
        print(f"  Shape: {all_summary.shape}")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    if exams_rm is not None:
        print(f"  Arabic EXAMS: {exams_rm.shape[0]} models x {exams_rm.shape[1]} items")
    else:
        print(f"  Arabic EXAMS: No data")
    if mmlu_rm is not None:
        print(f"  Arabic MMLU (translated): {mmlu_rm.shape[0]} models x {mmlu_rm.shape[1]} items")
    else:
        print(f"  Arabic MMLU (translated): No data")

    print(f"\nNotes:")
    print(f"  - Arabic MMLU is a translated version of English MMLU, NOT native ArabicMMLU")
    print(f"  - Arabic EXAMS is a curated Arabic-only subset of the EXAMS benchmark")
    print(f"  - Source: Open Arabic LLM Leaderboard (OALL) on HuggingFace")
    print(f"  - Per-item binary (0/1) correct/incorrect from normalized log-likelihood eval")

    print("\nDone!")


if __name__ == "__main__":
    main()
