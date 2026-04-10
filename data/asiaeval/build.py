#!/usr/bin/env python3
"""
Build task metadata, item content CSVs, and response matrices for Southeast
Asian and South Asian NLP benchmarks.

=============================================================================
PART 1: Item Collection (from HuggingFace)
=============================================================================

Data sources:
  1. NusaX       -- thonyyy/nusax_sentiment (12 Indonesian local languages, sentiment)
                    (mirror of indonlp/NusaX-senti without requiring trust_remote_code)
  2. Belebele    -- facebook/belebele (reading comprehension, Global South subset)
  3. XCOPA       -- xcopa (causal reasoning, 11 languages)
  4. Global-MMLU -- CohereForAI/Global-MMLU (multilingual MMLU, subset)
  5. IndicCOPA   -- ai4bharat/IndicCOPA (15+ Indic languages, causal reasoning)
                    (loaded via direct JSONL downloads from HuggingFace Hub)

Outputs (in processed/):
  - task_metadata.csv      : item_id, text (first 200 chars), label, task_type,
                              language, source_dataset, split
  - item_content.csv       : item_id, full content
  - summary_stats.csv      : per-language/dataset/split summary statistics
  - summary_by_language.csv: high-level per-language summary

=============================================================================
PART 2: Per-Item Prediction Collection
=============================================================================

SOURCE INVESTIGATION SUMMARY (conducted 2026-03-21)

1. IndoMMLU (github.com/fajri91/IndoMMLU)
   - 14,981 Indonesian questions, 24 models evaluated
   - evaluate.py DOES save per-item predictions as CSV with columns:
     [input, golds, options, preds, probs]
   - Output path: {output_folder}/result_{model_name}_{by_letter}.csv
   - STATUS: Per-item prediction CSVs are NOT published in the repo.
     The output/ directory is in .gitignore. Must re-run evaluation.
   - ACTION: We can re-run evaluate.py to generate per-item predictions.
     Alternatively, contact Fajri Koto (fajri91) for the CSV outputs.

2. SEA-HELM (github.com/aisingapore/SEA-HELM)
   - Built on Stanford HELM framework; generates per_instance_stats.json
     and scenario_state.json locally per evaluation run.
   - STATUS: Per-instance results are NOT published. The leaderboard at
     leaderboard.sea-lion.ai shows only aggregate scores. No download API.
   - ACTION: Must run SEA-HELM locally to generate per-instance outputs.

3. NusaX / IndoNLP (github.com/IndoNLP/nusax)
   - Repository contains only the benchmark data (datasets/sentiment/) and
     evaluation notebooks. No model prediction files are published.
   - STATUS: No per-item model predictions available.
   - ACTION: Run models on NusaX-senti via lm-evaluation-harness with
     --log_samples to generate per-item JSONL outputs.

4. Open Thai LLM Leaderboard (huggingface.co/ThaiLLM-Leaderboard)
   - HuggingFace org has 3 datasets: requests, results, mt-bench-thai
   - results/ contains per-model JSON files organized as:
     results/{LLM,MC,NLG,NLU}/{model_name}/results.json
   - STATUS: Contains aggregate scores only (not per-item predictions).
     The OpenThaiGPT eval repo (github.com/OpenThaiGPT/openthaigpt_eval)
     has an outputs/ directory that is EMPTY in the published repo.
   - ACTION: Run evaluations locally using openthaigpt_eval or
     lm-evaluation-harness with Thai benchmarks + --log_samples.

5. VMLU (github.com/ZaloAI-Jaist/VMLU)
   - 10,880 Vietnamese multiple-choice questions across 58 subjects
   - test_gpt.py saves per-item predictions as CSV with columns:
     [id, prompt, question, answer] to all_res/gpt_result/raw_result_N.csv
   - Final submission.csv contains [id, answer] only.
   - STATUS: Per-item prediction CSVs are NOT published in the repo.
     Submissions go to vmlu.ai/submit (not publicly accessible).
   - ACTION: Re-run test_gpt.py or test_prompt.py to generate predictions.
     The VMLU dataset is on HuggingFace but without model outputs.

6. Global-MMLU (CohereForAI/Global-MMLU on HuggingFace)
   - 42 languages, cultural sensitivity annotations (CS/CA).
   - Global-MMLU is integrated into lm-evaluation-harness as a task.
   - STATUS: No per-item model predictions published. Dataset contains
     only questions + answers + cultural sensitivity labels.
   - ACTION: Run lm-evaluation-harness with --tasks global_mmlu_*
     and --log_samples to generate per-item predictions.

7. IndicGLUE / IndicBERT (github.com/AI4Bharat/IndicBERT)
   - Fine-tuning scripts for NER, paraphrase, QA, sentiment tasks.
   - STATUS: No per-item prediction files published. Repository contains
     only training/fine-tuning code and model configs.
   - ACTION: Run fine-tuning scripts which output predictions during eval.

8. Belebele (github.com/facebookresearch/belebele, facebook/belebele on HF)
   - 900 questions x 122 languages, parallel reading comprehension.
   - STATUS: No per-item model predictions published. The repo contains
     only the dataset and assembly scripts.
   - ACTION: Run lm-evaluation-harness with --tasks belebele_*
     and --log_samples to generate per-item predictions.

RECOMMENDED COLLECTION STRATEGY

The most practical approach is to run lm-evaluation-harness (EleutherAI)
with the --log_samples flag on all target benchmarks. This generates
per-sample JSONL files at:
  {output_path}/{model_name}/samples_{task_name}_{timestamp}.jsonl

Each JSONL line contains:
  - doc_id: question index
  - doc: original question data
  - target: correct answer
  - filtered_resps: model's filtered responses
  - acc: whether the model got it correct (0/1)

Additionally, Stanford HELM raw results are downloadable from:
  gs://crfm-helm-public/{project}/benchmark_output/
  (Requires gcloud CLI; data is several hundred GB per project)

HELM output per-run includes:
  - scenario_state.json: every request to and response from the model
  - per_instance_stats.json: metrics for each individual item
  - stats.json: aggregated statistics

Additional outputs (in processed/):
  - all_predictions.csv    : raw per-item predictions from all sources
  - response_matrix.csv    : items x models, values = 1/0/NaN
  - alignment_report.json  : alignment statistics
"""

import os
import sys
import json
import glob
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _SCRIPT_DIR
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PREDICTIONS_DIR = RAW_DIR / "predictions"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Global accumulators (used by item-collection functions)
all_metadata_rows = []
all_content_rows = []
item_counter = 0


# ======================================================================
# Part 1 helpers: Item collection
# ======================================================================

def next_item_id():
    global item_counter
    iid = f"item_{item_counter:06d}"
    item_counter += 1
    return iid


def truncate(text, max_len=200):
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ──────────────────────────────────────────────────────────────────────
# 1. NusaX -- via thonyyy/nusax_sentiment (parquet mirror, no script)
# ──────────────────────────────────────────────────────────────────────
def load_nusax():
    print("=" * 70)
    print("DATASET 1: NusaX (thonyyy/nusax_sentiment)")
    print("  Mirror of indonlp/NusaX-senti without trust_remote_code")
    print("=" * 70)

    from datasets import load_dataset

    total_items = 0
    try:
        # This mirror has all languages in a single dataset with a 'language' column
        ds = load_dataset("thonyyy/nusax_sentiment")
        for split_name in ds.keys():
            split_ds = ds[split_name]
            n = len(split_ds)
            print(f"  {split_name}: {n} items")
            print(f"  Columns: {split_ds.column_names}")

            # Check first sample to understand structure
            if n > 0:
                sample = split_ds[0]
                print(f"  Sample: {sample}")

            for row in split_ds:
                iid = next_item_id()
                text = str(row.get("text", ""))
                label = row.get("label", "")
                language = str(row.get("language", "unknown"))
                orig_split = str(row.get("split", split_name))

                # Label is already string in this mirror: negative/neutral/positive
                label_str = str(label)

                all_metadata_rows.append({
                    "item_id": iid,
                    "text": truncate(text),
                    "label": label_str,
                    "task_type": "sentiment_analysis",
                    "language": language,
                    "source_dataset": "NusaX",
                    "split": orig_split,
                })
                all_content_rows.append({
                    "item_id": iid,
                    "content": text,
                })
                total_items += 1

    except Exception as e:
        print(f"  ERROR loading thonyyy/nusax_sentiment: {e}")
        traceback.print_exc()

    print(f"  Total NusaX items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 2. Belebele -- facebook/belebele (streaming mode for reliability)
# ──────────────────────────────────────────────────────────────────────
def load_belebele():
    print("=" * 70)
    print("DATASET 2: Belebele (facebook/belebele)")
    print("  Using streaming mode to avoid brotli decoder errors")
    print("=" * 70)

    from datasets import load_dataset, get_dataset_config_names

    target_langs = [
        "hin_Deva", "ben_Beng", "tam_Taml", "tha_Thai", "vie_Latn",
        "ind_Latn", "tgl_Latn", "swa_Latn", "yor_Latn", "ara_Arab",
        "zho_Hans",
    ]

    try:
        configs = get_dataset_config_names("facebook/belebele")
        print(f"  Available configs ({len(configs)}): {configs[:10]}...")
        available_target = [l for l in target_langs if l in configs]
        missing = [l for l in target_langs if l not in configs]
        if missing:
            print(f"  Missing configs: {missing}")
        target_langs_final = available_target
    except Exception as e:
        print(f"  Could not get configs: {e}")
        target_langs_final = target_langs

    total_items = 0
    for lang in target_langs_final:
        try:
            # Use streaming to avoid brotli decoder issues
            ds_stream = load_dataset(
                "facebook/belebele", lang, split="test", streaming=True
            )
            items = list(ds_stream)
            n = len(items)
            print(f"  {lang}/test: {n} items")

            for row in items:
                iid = next_item_id()

                passage = str(row.get("flores_passage", ""))
                question = str(row.get("question", ""))
                mc1 = str(row.get("mc_answer1", ""))
                mc2 = str(row.get("mc_answer2", ""))
                mc3 = str(row.get("mc_answer3", ""))
                mc4 = str(row.get("mc_answer4", ""))
                correct = row.get("correct_answer_num", "")

                full_content = (
                    f"Passage: {passage}\n"
                    f"Question: {question}\n"
                    f"A: {mc1}\nB: {mc2}\nC: {mc3}\nD: {mc4}\n"
                    f"Correct: {correct}"
                )

                all_metadata_rows.append({
                    "item_id": iid,
                    "text": truncate(f"{question} | {passage}"),
                    "label": str(correct),
                    "task_type": "reading_comprehension",
                    "language": lang,
                    "source_dataset": "Belebele",
                    "split": "test",
                })
                all_content_rows.append({
                    "item_id": iid,
                    "content": full_content,
                })
                total_items += 1

        except Exception as e:
            print(f"  ERROR loading {lang}: {e}")

    print(f"  Total Belebele items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 3. XCOPA -- xcopa (causal reasoning)
# ──────────────────────────────────────────────────────────────────────
def load_xcopa():
    print("=" * 70)
    print("DATASET 3: XCOPA (causal reasoning)")
    print("=" * 70)

    from datasets import load_dataset, get_dataset_config_names

    hf_names = ["xcopa", "cambridgeltl/xcopa"]

    configs = None
    chosen_name = None
    for hf_name in hf_names:
        try:
            configs = get_dataset_config_names(hf_name)
            chosen_name = hf_name
            print(f"  Using dataset: {hf_name}")
            print(f"  Available configs: {configs}")
            break
        except Exception as e:
            print(f"  Tried {hf_name}: {e}")

    if configs is None:
        print("  Could not load XCOPA from any source. Skipping.")
        return

    total_items = 0
    for lang in configs:
        try:
            ds = load_dataset(chosen_name, lang)
            for split_name in ds.keys():
                split_ds = ds[split_name]
                n = len(split_ds)
                print(f"  {lang}/{split_name}: {n} items")

                for row in split_ds:
                    iid = next_item_id()

                    premise = str(row.get("premise", ""))
                    choice1 = str(row.get("choice1", ""))
                    choice2 = str(row.get("choice2", ""))
                    question_type = str(row.get("question", ""))
                    label = row.get("label", "")

                    full_content = (
                        f"Premise: {premise}\n"
                        f"Question type: {question_type}\n"
                        f"Choice 1: {choice1}\n"
                        f"Choice 2: {choice2}\n"
                        f"Label: {label}"
                    )

                    all_metadata_rows.append({
                        "item_id": iid,
                        "text": truncate(f"{premise} | {choice1} | {choice2}"),
                        "label": str(label),
                        "task_type": "causal_reasoning",
                        "language": lang,
                        "source_dataset": "XCOPA",
                        "split": split_name,
                    })
                    all_content_rows.append({
                        "item_id": iid,
                        "content": full_content,
                    })
                    total_items += 1

        except Exception as e:
            print(f"  ERROR loading {lang}: {e}")

    print(f"  Total XCOPA items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 4. Global-MMLU -- CohereForAI/Global-MMLU
# ──────────────────────────────────────────────────────────────────────
def load_global_mmlu():
    print("=" * 70)
    print("DATASET 4: Global-MMLU (CohereForAI/Global-MMLU)")
    print("=" * 70)

    from datasets import load_dataset, get_dataset_config_names

    target_langs = ["ar", "hi", "bn", "th", "vi", "id", "sw", "yo", "es", "pt", "zh"]

    hf_names = ["CohereForAI/Global-MMLU", "CohereLabs/Global-MMLU"]
    configs = None
    chosen_name = None

    for hf_name in hf_names:
        try:
            configs = get_dataset_config_names(hf_name)
            chosen_name = hf_name
            print(f"  Using dataset: {hf_name}")
            print(f"  Available configs ({len(configs)}): {configs[:20]}...")
            break
        except Exception as e:
            print(f"  Tried {hf_name}: {e}")

    if configs is None:
        print("  Could not load Global-MMLU. Skipping.")
        return

    available_target = [l for l in target_langs if l in configs]
    missing = [l for l in target_langs if l not in configs]
    if missing:
        print(f"  Missing language configs: {missing}")
    print(f"  Will load languages: {available_target}")

    total_items = 0
    for lang in available_target:
        try:
            print(f"  Loading {lang}...", end=" ", flush=True)
            ds = load_dataset(chosen_name, lang, split="test")
            n = len(ds)
            print(f"test: {n} items")

            # Cap at 5000 per language
            max_items = 5000
            indices = range(min(n, max_items))
            if n > max_items:
                print(f"    (sampling {max_items} of {n} items)")

            for idx in indices:
                row = ds[int(idx)]
                iid = next_item_id()

                question = str(row.get("question", ""))
                subject = str(row.get("subject", ""))
                answer = str(row.get("answer", ""))
                choices = row.get("choices", [])
                if isinstance(choices, list):
                    choices_str = " | ".join(str(c) for c in choices)
                else:
                    choices_str = str(choices)

                answer_map = {0: "A", 1: "B", 2: "C", 3: "D",
                              "0": "A", "1": "B", "2": "C", "3": "D",
                              "A": "A", "B": "B", "C": "C", "D": "D"}
                answer_str = answer_map.get(answer, str(answer))

                full_content = (
                    f"Subject: {subject}\n"
                    f"Question: {question}\n"
                    f"Choices: {choices_str}\n"
                    f"Answer: {answer_str}"
                )

                all_metadata_rows.append({
                    "item_id": iid,
                    "text": truncate(f"[{subject}] {question}"),
                    "label": answer_str,
                    "task_type": "multiple_choice_qa",
                    "language": lang,
                    "source_dataset": "Global-MMLU",
                    "split": "test",
                })
                all_content_rows.append({
                    "item_id": iid,
                    "content": full_content,
                })
                total_items += 1

        except Exception as e:
            print(f"\n  ERROR loading {lang}: {e}")
            traceback.print_exc()

    print(f"  Total Global-MMLU items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 5. IndicCOPA -- ai4bharat/IndicCOPA (direct JSONL download)
# ──────────────────────────────────────────────────────────────────────
def load_indiccopa():
    print("=" * 70)
    print("DATASET 5: IndicCOPA (ai4bharat/IndicCOPA)")
    print("  Loading via direct JSONL file downloads (dataset script unsupported)")
    print("=" * 70)

    from huggingface_hub import hf_hub_download, list_repo_files

    # Discover available language files
    try:
        all_files = list_repo_files("ai4bharat/IndicCOPA", repo_type="dataset")
        jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
        print(f"  Found {len(jsonl_files)} JSONL files: {jsonl_files}")
    except Exception as e:
        print(f"  ERROR listing repo files: {e}")
        return

    # Parse language codes from filenames like "data/test.hi.jsonl"
    lang_files = {}
    for f in jsonl_files:
        parts = f.split("/")[-1].replace(".jsonl", "").split(".")
        if len(parts) >= 2:
            split_name = parts[0]
            lang_code = parts[1]
            lang_files.setdefault(lang_code, []).append((split_name, f))

    print(f"  Languages found: {sorted(lang_files.keys())}")

    total_items = 0
    for lang in sorted(lang_files.keys()):
        for split_name, filepath in lang_files[lang]:
            try:
                local_path = hf_hub_download(
                    "ai4bharat/IndicCOPA", filepath, repo_type="dataset"
                )
                with open(local_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                n = len(lines)
                print(f"  {lang}/{split_name}: {n} items")

                for line in lines:
                    row = json.loads(line.strip())
                    iid = next_item_id()

                    premise = str(row.get("premise", ""))
                    choice1 = str(row.get("choice1", ""))
                    choice2 = str(row.get("choice2", ""))
                    question_type = str(row.get("question", ""))
                    label = row.get("label", "")

                    full_content = (
                        f"Premise: {premise}\n"
                        f"Question type: {question_type}\n"
                        f"Choice 1: {choice1}\n"
                        f"Choice 2: {choice2}\n"
                        f"Label: {label}"
                    )

                    all_metadata_rows.append({
                        "item_id": iid,
                        "text": truncate(f"{premise} | {choice1} | {choice2}"),
                        "label": str(label),
                        "task_type": "causal_reasoning",
                        "language": lang,
                        "source_dataset": "IndicCOPA",
                        "split": split_name,
                    })
                    all_content_rows.append({
                        "item_id": iid,
                        "content": full_content,
                    })
                    total_items += 1

            except Exception as e:
                print(f"  ERROR loading {lang}/{split_name}: {e}")
                traceback.print_exc()

    print(f"  Total IndicCOPA items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# Summary Statistics
# ──────────────────────────────────────────────────────────────────────
def build_summary_stats(metadata_df):
    """Build per-language, per-dataset summary statistics."""
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = metadata_df.groupby(
        ["source_dataset", "language", "task_type", "split"]
    ).agg(
        n_items=("item_id", "count"),
        n_labels=("label", "nunique"),
        label_distribution=("label", lambda x: dict(x.value_counts().head(5))),
    ).reset_index()

    high_level = metadata_df.groupby(
        ["source_dataset", "language"]
    ).agg(
        n_items=("item_id", "count"),
        n_splits=("split", "nunique"),
        task_types=("task_type", lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()

    print(f"\n  Datasets loaded: {metadata_df['source_dataset'].nunique()}")
    print(f"  Total items: {len(metadata_df)}")
    print(f"  Languages: {metadata_df['language'].nunique()}")
    print(f"  Task types: {sorted(metadata_df['task_type'].unique())}")

    print("\n  Items per dataset:")
    for ds, grp in metadata_df.groupby("source_dataset"):
        langs = sorted(grp["language"].unique())
        print(f"    {ds}: {len(grp)} items, {len(langs)} languages")
        print(f"      Languages: {', '.join(langs)}")

    print("\n  Items per language (across all datasets):")
    lang_counts = metadata_df.groupby("language")["item_id"].count().sort_values(
        ascending=False
    )
    for lang, count in lang_counts.items():
        datasets = metadata_df[metadata_df["language"] == lang][
            "source_dataset"
        ].unique()
        print(f"    {lang:15s}: {count:6d} items  ({', '.join(sorted(datasets))})")

    return summary, high_level


# ======================================================================
# Part 2 helpers: Prediction collection and response matrix building
# ======================================================================

# Task definitions for lm-evaluation-harness
LM_EVAL_TASKS = {
    # Belebele (reading comprehension, 122 languages)
    # Available as belebele_{lang_code} in lm-eval-harness
    "belebele": {
        "task_pattern": "belebele_{lang}",
        "languages": [
            "hin_Deva", "ben_Beng", "tam_Taml", "tha_Thai", "vie_Latn",
            "ind_Latn", "tgl_Latn", "swa_Latn", "yor_Latn", "ara_Arab",
            "zho_Hans",
        ],
        "source_dataset": "Belebele",
    },
    # XCOPA (causal reasoning)
    "xcopa": {
        "task_pattern": "xcopa_{lang}",
        "languages": [
            "et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh",
        ],
        "source_dataset": "XCOPA",
    },
    # Global-MMLU (multilingual MMLU)
    "global_mmlu": {
        "task_pattern": "global_mmlu_{lang}",
        "languages": [
            "ar", "hi", "bn", "th", "vi", "id", "sw", "yo", "es", "pt", "zh",
        ],
        "source_dataset": "Global-MMLU",
    },
    # IndoMMLU (Indonesian MMLU, via SEACrowd/lm-eval integration)
    "indommlu": {
        "task_pattern": "indommlu",
        "languages": ["id"],
        "source_dataset": "IndoMMLU",
    },
}

# Models to evaluate (representative set across size categories)
TARGET_MODELS = [
    # Large frontier models (API-based)
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    # Open-weight large models
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "google/gemma-2-27b-it",
    # Open-weight medium models
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    # Southeast Asian specialized models
    "aisingapore/Gemma-SEA-LION-v3-9B",
    "scb10x/typhoon-v1.5-7b-instruct",
    # South Asian specialized models
    "ai4bharat/Airavata",
]


def generate_lm_eval_commands(
    output_dir: str = None,
    models: list = None,
    num_fewshot: int = 0,
):
    """
    Generate shell commands to run lm-evaluation-harness with --log_samples
    for all target benchmarks and models.

    Args:
        output_dir: Directory to save evaluation outputs.
        models: List of model identifiers.
        num_fewshot: Number of few-shot examples.

    Returns:
        List of shell command strings.
    """
    if output_dir is None:
        output_dir = str(PREDICTIONS_DIR / "lm_eval_outputs")
    if models is None:
        models = TARGET_MODELS

    commands = []

    for benchmark_key, benchmark_info in LM_EVAL_TASKS.items():
        # Build task list
        if "{lang}" in benchmark_info["task_pattern"]:
            tasks = [
                benchmark_info["task_pattern"].format(lang=lang)
                for lang in benchmark_info["languages"]
            ]
        else:
            tasks = [benchmark_info["task_pattern"]]

        task_str = ",".join(tasks)

        for model in models:
            cmd = (
                f"lm_eval "
                f"--model hf "
                f"--model_args pretrained={model} "
                f"--tasks {task_str} "
                f"--num_fewshot {num_fewshot} "
                f"--batch_size auto "
                f"--output_path {output_dir}/{model.split('/')[-1]} "
                f"--log_samples"
            )
            commands.append(cmd)

    return commands


def write_eval_script(output_path: str = None):
    """Write a bash script with all evaluation commands."""
    if output_path is None:
        output_path = str(_SCRIPT_DIR / "run_evaluations.sh")

    commands = generate_lm_eval_commands()

    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated evaluation commands for AsiaEval benchmarks\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("# Requires: pip install lm-eval\n")
        f.write("#\n")
        f.write("# Each command runs lm-evaluation-harness with --log_samples\n")
        f.write("# to produce per-item JSONL prediction files.\n")
        f.write("#\n")
        f.write("# Output structure:\n")
        f.write("#   {output_dir}/{model}/samples_{task}_{timestamp}.jsonl\n")
        f.write("#\n")
        f.write("set -e\n\n")

        for i, cmd in enumerate(commands):
            f.write(f"echo '=== Command {i+1}/{len(commands)} ==='\n")
            f.write(f"{cmd}\n\n")

    os.chmod(output_path, 0o755)
    print(f"Wrote {len(commands)} evaluation commands to {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Parse lm-evaluation-harness per-sample JSONL outputs
# ──────────────────────────────────────────────────────────────────────

def parse_lm_eval_samples(samples_dir: str) -> pd.DataFrame:
    """
    Parse lm-evaluation-harness --log_samples JSONL output files.

    The harness saves files as:
      {output_path}/{model}/samples_{task}_{timestamp}.jsonl

    Each JSONL line contains:
      - doc_id: int (question index within the task)
      - doc: dict (original question data)
      - target: str/int (correct answer)
      - filtered_resps: list (model's filtered responses)
      - acc: float (1.0 if correct, 0.0 if incorrect)
      - (optionally) acc_norm, exact_match, etc.

    Returns:
        DataFrame with columns:
        [model, task, doc_id, target, prediction, correct, raw_response]
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        print(f"  WARNING: {samples_dir} does not exist.")
        return pd.DataFrame()

    rows = []

    # Find all model output directories
    model_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name

        # Find all samples_*.jsonl files
        jsonl_files = sorted(model_dir.glob("samples_*.jsonl"))

        for jsonl_file in jsonl_files:
            # Extract task name from filename
            # Format: samples_{task_name}_{timestamp}.jsonl
            fname = jsonl_file.stem  # e.g., samples_belebele_hin_Deva_2024-01-01...
            parts = fname.split("_", 1)
            if len(parts) < 2:
                continue
            # task name is everything between "samples_" and the timestamp
            task_part = parts[1]
            # Remove timestamp suffix (last part after the date pattern)
            # Timestamps look like: _2024-06-28T00-00-00.00001
            import re
            task_name = re.sub(
                r"_\d{4}-\d{2}-\d{2}T\d{2}.*$", "", task_part
            )

            print(f"  Parsing {model_name}/{jsonl_file.name} (task={task_name})")

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    doc_id = record.get("doc_id", None)
                    target = record.get("target", None)
                    acc = record.get("acc", None)
                    acc_norm = record.get("acc_norm", None)
                    filtered_resps = record.get("filtered_resps", None)

                    # Extract the model's top prediction
                    prediction = None
                    if filtered_resps and isinstance(filtered_resps, list):
                        if len(filtered_resps) > 0:
                            resp = filtered_resps[0]
                            if isinstance(resp, list) and len(resp) > 0:
                                prediction = resp[0]
                            else:
                                prediction = resp

                    correct = None
                    if acc is not None:
                        correct = int(float(acc) >= 0.5)
                    elif acc_norm is not None:
                        correct = int(float(acc_norm) >= 0.5)

                    rows.append({
                        "model": model_name,
                        "task": task_name,
                        "doc_id": doc_id,
                        "target": str(target) if target is not None else None,
                        "prediction": str(prediction) if prediction is not None else None,
                        "correct": correct,
                        "raw_response": json.dumps(filtered_resps) if filtered_resps else None,
                    })

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} per-item predictions from lm-eval-harness outputs")
    return df


# ──────────────────────────────────────────────────────────────────────
# Parse IndoMMLU-format CSV outputs
# ──────────────────────────────────────────────────────────────────────

def parse_indommlu_results(results_dir: str) -> pd.DataFrame:
    """
    Parse IndoMMLU evaluate.py CSV outputs.

    The evaluate.py script saves CSVs as:
      {output_folder}/result_{model_name}_{by_letter}.csv

    CSV columns: [input, golds, options, preds, probs]
    - golds: ground truth label (0-4 for options A-E)
    - preds: model prediction (0-4)
    - probs: confidence probabilities

    Returns:
        DataFrame with columns:
        [model, task, doc_id, target, prediction, correct, raw_response]
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"  WARNING: {results_dir} does not exist.")
        return pd.DataFrame()

    rows = []
    csv_files = sorted(results_dir.glob("result_*.csv"))

    for csv_file in csv_files:
        # Extract model name from filename: result_{model}_{letter_mode}.csv
        fname = csv_file.stem
        parts = fname.replace("result_", "").rsplit("_", 1)
        model_name = parts[0] if len(parts) >= 1 else fname

        print(f"  Parsing IndoMMLU results: {csv_file.name} (model={model_name})")

        try:
            df_raw = pd.read_csv(csv_file)
        except Exception as e:
            print(f"    ERROR reading {csv_file}: {e}")
            continue

        for idx, row in df_raw.iterrows():
            gold = row.get("golds", None)
            pred = row.get("preds", None)
            probs = row.get("probs", None)

            # Convert numeric labels to letters
            label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
            target_str = label_map.get(gold, str(gold))
            pred_str = label_map.get(pred, str(pred))

            correct = 1 if gold == pred else 0

            rows.append({
                "model": model_name,
                "task": "indommlu",
                "doc_id": idx,
                "target": target_str,
                "prediction": pred_str,
                "correct": correct,
                "raw_response": str(probs) if probs is not None else None,
            })

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} per-item predictions from IndoMMLU outputs")
    return df


# ──────────────────────────────────────────────────────────────────────
# Parse VMLU-format CSV outputs
# ──────────────────────────────────────────────────────────────────────

def parse_vmlu_results(results_dir: str) -> pd.DataFrame:
    """
    Parse VMLU test_gpt.py CSV outputs.

    The test_gpt.py script saves intermediate CSVs as:
      all_res/gpt_result/raw_result_{N}.csv

    CSV columns: [id, prompt, question, answer]
    - id: question identifier
    - answer: model's answer (single letter a-e)

    Returns:
        DataFrame with columns:
        [model, task, doc_id, target, prediction, correct, raw_response]
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"  WARNING: {results_dir} does not exist.")
        return pd.DataFrame()

    rows = []
    csv_files = sorted(results_dir.glob("raw_result_*.csv"))

    if not csv_files:
        # Also check for submission.csv format
        csv_files = sorted(results_dir.glob("submission*.csv"))

    for csv_file in csv_files:
        model_name = csv_file.parent.name  # Use parent dir as model name
        print(f"  Parsing VMLU results: {csv_file.name} (model={model_name})")

        try:
            df_raw = pd.read_csv(csv_file)
        except Exception as e:
            print(f"    ERROR reading {csv_file}: {e}")
            continue

        for idx, row in df_raw.iterrows():
            qid = row.get("id", idx)
            answer = str(row.get("answer", "")).strip().upper()

            rows.append({
                "model": model_name,
                "task": "vmlu",
                "doc_id": qid,
                "target": None,  # Ground truth not in submission files
                "prediction": answer,
                "correct": None,  # Cannot determine without ground truth
                "raw_response": None,
            })

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} per-item predictions from VMLU outputs")
    return df


# ──────────────────────────────────────────────────────────────────────
# Parse HELM per_instance_stats.json outputs
# ──────────────────────────────────────────────────────────────────────

def parse_helm_per_instance(runs_dir: str) -> pd.DataFrame:
    """
    Parse Stanford HELM per_instance_stats.json files.

    HELM output per-run directory contains:
      - run_spec.json: scenario + adapter specification
      - scenario_state.json: every request to and response from the model
      - per_instance_stats.json: metrics for each individual item
      - stats.json: aggregated statistics

    Download HELM raw results from:
      gs://crfm-helm-public/{project}/benchmark_output/

    Returns:
        DataFrame with columns:
        [model, task, doc_id, target, prediction, correct, raw_response]
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        print(f"  WARNING: {runs_dir} does not exist.")
        return pd.DataFrame()

    rows = []

    # Each subdirectory in runs/ is a specific (model, scenario) run
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]

    for run_dir in sorted(run_dirs):
        per_instance_path = run_dir / "per_instance_stats.json"
        scenario_state_path = run_dir / "scenario_state.json"
        run_spec_path = run_dir / "run_spec.json"

        if not per_instance_path.exists():
            continue

        # Get model and scenario from run_spec
        model_name = run_dir.name
        task_name = "unknown"
        if run_spec_path.exists():
            try:
                with open(run_spec_path, "r") as f:
                    run_spec = json.load(f)
                model_name = run_spec.get("adapter_spec", {}).get("model", model_name)
                task_name = run_spec.get("scenario_spec", {}).get("class_name", task_name)
            except Exception:
                pass

        print(f"  Parsing HELM run: {run_dir.name} (model={model_name}, task={task_name})")

        try:
            with open(per_instance_path, "r") as f:
                per_instance_stats = json.load(f)
        except Exception as e:
            print(f"    ERROR reading {per_instance_path}: {e}")
            continue

        for item in per_instance_stats:
            instance_id = item.get("instance_id", None)
            train_trial_index = item.get("train_trial_index", 0)

            # Extract metrics
            stats = item.get("stats", [])
            correct = None
            for stat in stats:
                name = stat.get("name", {})
                if isinstance(name, dict) and name.get("name") == "exact_match":
                    correct = int(stat.get("sum", 0))
                    break

            rows.append({
                "model": model_name,
                "task": task_name,
                "doc_id": instance_id,
                "target": None,
                "prediction": None,
                "correct": correct,
                "raw_response": None,
            })

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} per-item predictions from HELM outputs")
    return df


# ──────────────────────────────────────────────────────────────────────
# Build unified response matrix
# ──────────────────────────────────────────────────────────────────────

def build_response_matrix(predictions_df: pd.DataFrame, metadata_path: str = None):
    """
    Build a response matrix from collected predictions.

    The response matrix has:
      Rows = items (aligned with task_metadata.csv item_ids)
      Columns = models
      Values = 1 (correct) / 0 (incorrect) / NaN (not evaluated)

    Args:
        predictions_df: DataFrame with columns
            [model, task, doc_id, target, prediction, correct]
        metadata_path: Path to task_metadata.csv

    Returns:
        response_matrix: DataFrame (items x models)
        alignment_report: dict with alignment statistics
    """
    if metadata_path is None:
        metadata_path = str(PROCESSED_DIR / "task_metadata.csv")

    metadata = pd.read_csv(metadata_path)
    print(f"  Loaded metadata: {len(metadata)} items")

    if len(predictions_df) == 0:
        print("  WARNING: No predictions to build response matrix from.")
        return pd.DataFrame(), {"status": "no_predictions"}

    # Get unique models
    models = sorted(predictions_df["model"].unique())
    print(f"  Models found: {len(models)}")

    # Initialize response matrix with NaN
    response_matrix = pd.DataFrame(
        np.nan,
        index=metadata["item_id"],
        columns=models,
    )

    # Task-to-dataset mapping for alignment
    task_dataset_map = {
        "belebele": "Belebele",
        "xcopa": "XCOPA",
        "global_mmlu": "Global-MMLU",
        "indommlu": "IndoMMLU",
        "vmlu": "VMLU",
    }

    alignment_report = {
        "total_predictions": len(predictions_df),
        "models": models,
        "aligned": 0,
        "unaligned": 0,
        "per_model": {},
    }

    # For each model, align predictions to metadata items
    for model in models:
        model_preds = predictions_df[predictions_df["model"] == model]

        aligned_count = 0
        for _, pred_row in model_preds.iterrows():
            task = pred_row["task"]
            doc_id = pred_row["doc_id"]
            correct = pred_row["correct"]

            if correct is None:
                continue

            # Try to find matching item in metadata
            # This requires task-specific alignment logic
            source_dataset = None
            for task_key, ds_name in task_dataset_map.items():
                if task_key in str(task).lower():
                    source_dataset = ds_name
                    break

            if source_dataset is None:
                continue

            # Find items from the same dataset
            mask = metadata["source_dataset"] == source_dataset
            dataset_items = metadata[mask]

            if doc_id is not None and isinstance(doc_id, (int, float)):
                doc_id = int(doc_id)
                if doc_id < len(dataset_items):
                    item_id = dataset_items.iloc[doc_id]["item_id"]
                    response_matrix.loc[item_id, model] = correct
                    aligned_count += 1

        alignment_report["per_model"][model] = {
            "total_predictions": len(model_preds),
            "aligned": aligned_count,
        }
        alignment_report["aligned"] += aligned_count

    alignment_report["unaligned"] = (
        alignment_report["total_predictions"] - alignment_report["aligned"]
    )

    # Summary statistics
    n_evaluated = response_matrix.notna().sum().sum()
    n_total = response_matrix.shape[0] * response_matrix.shape[1]
    coverage = n_evaluated / n_total if n_total > 0 else 0

    print(f"\n  Response Matrix Summary:")
    print(f"    Items: {response_matrix.shape[0]}")
    print(f"    Models: {response_matrix.shape[1]}")
    print(f"    Evaluated cells: {n_evaluated:,} / {n_total:,} ({coverage:.1%})")
    print(f"    Aligned predictions: {alignment_report['aligned']}")
    print(f"    Unaligned predictions: {alignment_report['unaligned']}")

    return response_matrix, alignment_report


# ======================================================================
# Main
# ======================================================================

def main():
    # ==================================================================
    # Part 1: Collect items from HuggingFace
    # ==================================================================
    print("AsiaEval Response Matrix Builder")
    print("=" * 70)
    print("Building task metadata and item content for SE/South Asian benchmarks")
    print("=" * 70 + "\n")

    # Load each dataset
    load_nusax()
    load_belebele()
    load_xcopa()
    load_global_mmlu()
    load_indiccopa()

    # Build DataFrames
    print("\n" + "=" * 70)
    print("BUILDING OUTPUT FILES")
    print("=" * 70)

    metadata_df = pd.DataFrame(all_metadata_rows)
    content_df = pd.DataFrame(all_content_rows)

    if len(metadata_df) == 0:
        print("ERROR: No data was loaded. Exiting.")
        sys.exit(1)

    print(f"  Total metadata rows: {len(metadata_df)}")
    print(f"  Total content rows: {len(content_df)}")

    # Build summary stats
    summary, high_level = build_summary_stats(metadata_df)

    # -- Save outputs --
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    out_meta = PROCESSED_DIR / "task_metadata.csv"
    metadata_df.to_csv(out_meta, index=False)
    print(f"  Saved: {out_meta} ({len(metadata_df)} rows)")

    out_content = PROCESSED_DIR / "item_content.csv"
    content_df.to_csv(out_content, index=False)
    print(f"  Saved: {out_content} ({len(content_df)} rows)")

    out_summary = PROCESSED_DIR / "summary_stats.csv"
    summary.to_csv(out_summary, index=False)
    print(f"  Saved: {out_summary}")

    out_high = PROCESSED_DIR / "summary_by_language.csv"
    high_level.to_csv(out_high, index=False)
    print(f"  Saved: {out_high}")

    # -- Final report for Part 1 --
    print("\n" + "=" * 70)
    print("ITEM COLLECTION REPORT")
    print("=" * 70)
    print(f"  Datasets: {metadata_df['source_dataset'].nunique()}")
    print(f"  Languages: {metadata_df['language'].nunique()}")
    print(f"  Total items: {len(metadata_df)}")
    print(f"  Task types: {sorted(metadata_df['task_type'].unique())}")
    print(f"\n  Output directory: {PROCESSED_DIR}")

    # ==================================================================
    # Part 2: Collect predictions and build response matrices
    # ==================================================================
    print()
    print("=" * 72)
    print("AsiaEval Per-Item Prediction Collector")
    print("=" * 72)
    print()

    # Step 1: Generate evaluation commands
    print("STEP 1: Generating lm-evaluation-harness commands")
    print("-" * 72)
    script_path = write_eval_script()
    print()

    # Step 2: Check for any existing prediction outputs
    print("STEP 2: Scanning for existing prediction outputs")
    print("-" * 72)

    all_predictions = []

    # Check for lm-eval-harness outputs
    lm_eval_dir = PREDICTIONS_DIR / "lm_eval_outputs"
    if lm_eval_dir.exists():
        print(f"  Found lm-eval-harness outputs at {lm_eval_dir}")
        df = parse_lm_eval_samples(str(lm_eval_dir))
        if len(df) > 0:
            all_predictions.append(df)
    else:
        print(f"  No lm-eval-harness outputs found at {lm_eval_dir}")
        print(f"  Run the generated script: {script_path}")

    # Check for IndoMMLU outputs
    indommlu_dir = PREDICTIONS_DIR / "indommlu_outputs"
    if indommlu_dir.exists():
        print(f"  Found IndoMMLU outputs at {indommlu_dir}")
        df = parse_indommlu_results(str(indommlu_dir))
        if len(df) > 0:
            all_predictions.append(df)
    else:
        print(f"  No IndoMMLU outputs found at {indommlu_dir}")

    # Check for VMLU outputs
    vmlu_dir = PREDICTIONS_DIR / "vmlu_outputs"
    if vmlu_dir.exists():
        print(f"  Found VMLU outputs at {vmlu_dir}")
        df = parse_vmlu_results(str(vmlu_dir))
        if len(df) > 0:
            all_predictions.append(df)
    else:
        print(f"  No VMLU outputs found at {vmlu_dir}")

    # Check for HELM outputs
    helm_dir = PREDICTIONS_DIR / "helm_outputs"
    if helm_dir.exists():
        print(f"  Found HELM outputs at {helm_dir}")
        df = parse_helm_per_instance(str(helm_dir))
        if len(df) > 0:
            all_predictions.append(df)
    else:
        print(f"  No HELM outputs found at {helm_dir}")

    print()

    # Step 3: Build response matrix if predictions exist
    print("STEP 3: Building response matrix")
    print("-" * 72)

    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        print(f"  Total predictions collected: {len(combined_df)}")

        # Save raw predictions
        pred_out = PROCESSED_DIR / "all_predictions.csv"
        combined_df.to_csv(pred_out, index=False)
        print(f"  Saved raw predictions to {pred_out}")

        # Build and save response matrix
        response_matrix, report = build_response_matrix(combined_df)

        if len(response_matrix) > 0:
            matrix_out = PROCESSED_DIR / "response_matrix.csv"
            response_matrix.to_csv(matrix_out)
            print(f"  Saved response matrix to {matrix_out}")

            report_out = PROCESSED_DIR / "alignment_report.json"
            with open(report_out, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"  Saved alignment report to {report_out}")
    else:
        print("  No predictions found yet.")
        print()
        print("  To collect per-item predictions, you have several options:")
        print()
        print("  OPTION A: Run lm-evaluation-harness (recommended)")
        print(f"    1. pip install lm-eval")
        print(f"    2. bash {script_path}")
        print(f"    3. Re-run this script to parse the outputs")
        print()
        print("  OPTION B: Download HELM raw results")
        print("    1. pip install gcloud")
        print("    2. gcloud storage rsync -r \\")
        print("       gs://crfm-helm-public/capabilities/benchmark_output/ \\")
        print(f"       {PREDICTIONS_DIR}/helm_outputs/")
        print("    3. Re-run this script to parse the outputs")
        print()
        print("  OPTION C: Re-run benchmark-specific evaluation scripts")
        print("    - IndoMMLU: cd IndoMMLU && python evaluate.py --output_folder")
        print(f"      {PREDICTIONS_DIR}/indommlu_outputs/")
        print("    - VMLU: cd VMLU && python code_benchmark/test_gpt.py")
        print(f"      (copy raw_result_*.csv to {PREDICTIONS_DIR}/vmlu_outputs/)")
        print()
        print("  OPTION D: Run SEA-HELM locally")
        print("    1. git clone https://github.com/aisingapore/SEA-HELM")
        print("    2. Follow SEA-HELM setup and run evaluation")
        print(f"    3. Copy per_instance_stats.json files to {PREDICTIONS_DIR}/helm_outputs/")

    # Step 4: Data availability summary
    print()
    print("=" * 72)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 72)
    print()
    print("Source                  | Per-Item Data? | Status")
    print("-" * 72)
    print("IndoMMLU               | YES (via eval) | Must re-run evaluate.py")
    print("SEA-HELM               | YES (via eval) | Must run locally; no published outputs")
    print("NusaX                  | NO             | No published model predictions")
    print("Thai LLM Leaderboard   | NO             | Aggregate scores only in results/")
    print("VMLU                   | YES (via eval) | Must re-run test_gpt.py")
    print("Global-MMLU            | NO             | Dataset only (no model outputs)")
    print("IndicGLUE/IndicBERT    | NO             | Training code only")
    print("Belebele               | NO             | Dataset only (no model outputs)")
    print()
    print("BEST PATH FORWARD:")
    print("  Run lm-evaluation-harness with --log_samples on all benchmarks.")
    print("  This is the only unified way to get per-item predictions across")
    print("  all target benchmarks with consistent formatting.")
    print()
    print("  Supported tasks in lm-eval-harness:")
    print("    - belebele_{lang}    (122 language variants)")
    print("    - xcopa_{lang}       (11 languages)")
    print("    - global_mmlu_{lang} (42 languages)")
    print("    - indommlu           (Indonesian)")
    print()
    print("  Additional benchmarks require custom integration:")
    print("    - NusaX sentiment    (via SEACrowd loader)")
    print("    - IndicCOPA          (via custom task YAML)")
    print("    - VMLU               (via custom task YAML or direct script)")
    print()

    print("Done!")


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
