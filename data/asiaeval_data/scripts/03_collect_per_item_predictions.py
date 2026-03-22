#!/usr/bin/env python3
"""
Collect per-item model predictions for Southeast Asian and South Asian benchmarks.

This script attempts to gather per-item (per-question) model evaluation results
from publicly available sources. It complements the existing task_metadata.csv
(92,046 items across 64 languages) by adding model response data.

=============================================================================
SOURCE INVESTIGATION SUMMARY (conducted 2026-03-21)
=============================================================================

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
     The framework caches results in the user-specified --output_dir.

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

=============================================================================
RECOMMENDED COLLECTION STRATEGY
=============================================================================

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

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================

This script provides utilities to:
1. Generate lm-evaluation-harness commands for all target benchmarks
2. Parse lm-eval-harness per-sample JSONL outputs into a response matrix
3. Parse IndoMMLU-format CSV outputs into a response matrix
4. Parse VMLU-format CSV outputs into a response matrix
5. Merge all per-item predictions into a unified response matrix

The response matrix format:
  Rows = items (aligned with task_metadata.csv item_ids)
  Columns = models
  Values = 1 (correct) / 0 (incorrect) / NaN (not evaluated)
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent
RAW_DIR = _BASE_DIR / "raw"
PROCESSED_DIR = _BASE_DIR / "processed"
PREDICTIONS_DIR = RAW_DIR / "predictions"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1. Generate lm-evaluation-harness commands
# ──────────────────────────────────────────────────────────────────────

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
# 2. Parse lm-evaluation-harness per-sample JSONL outputs
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
# 3. Parse IndoMMLU-format CSV outputs
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
# 4. Parse VMLU-format CSV outputs
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
# 5. Parse HELM per_instance_stats.json outputs
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
# 6. Build unified response matrix
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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
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
