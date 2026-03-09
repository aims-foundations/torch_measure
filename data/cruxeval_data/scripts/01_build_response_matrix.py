"""
Build response matrices for CRUXEval benchmark.

CRUXEval is a code reasoning benchmark from Facebook Research with 800 Python
functions evaluated on two tasks:
  - CRUXEval-I (input prediction): predict the input that produces a given output
  - CRUXEval-O (output prediction): predict the output for a given input

This script:
  1. Loads pre-evaluated per-model per-task results from the CRUXEval repo
     (samples/evaluation_results_unzipped and samples/evaluation_results).
  2. Builds response matrices where rows = tasks (sample_0..sample_799),
     columns = model configurations, values = pass@1 score (fraction of
     correct generations per sample).
  3. Saves separate CSVs for CRUXEval-I and CRUXEval-O, plus a combined matrix.
  4. Saves a model_summary.csv with per-model aggregate statistics.

Source: https://github.com/facebookresearch/cruxeval
Paper: https://arxiv.org/abs/2401.03065
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw" / "cruxeval"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation result directories
EVAL_DIR_ZIP = RAW_DIR / "samples" / "evaluation_results_unzipped" / "evaluation_results"
EVAL_DIR_SAMPLE = RAW_DIR / "samples" / "evaluation_results"

# Ground truth
DATA_PATH = RAW_DIR / "data" / "cruxeval.jsonl"

NUM_SAMPLES = 800


def load_ground_truth():
    """Load the CRUXEval dataset with code, input, output for all 800 samples."""
    with open(DATA_PATH) as f:
        dataset = [json.loads(line) for line in f]
    assert len(dataset) == NUM_SAMPLES, f"Expected {NUM_SAMPLES} samples, got {len(dataset)}"
    return dataset


def discover_evaluation_files():
    """
    Find all evaluation result files from both the unzipped archive and the
    pre-existing sample scored files.

    Returns:
        dict: config_name -> file_path
    """
    results = {}

    # From unzipped evaluation results (primary source)
    if EVAL_DIR_ZIP.exists():
        for f in sorted(EVAL_DIR_ZIP.iterdir()):
            if f.suffix == ".json":
                config = f.stem
                try:
                    data = json.loads(f.read_text())
                    if "raw_scored_generations" in data and len(data["raw_scored_generations"]) > 0:
                        results[config] = str(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"  WARNING: Skipping corrupt file {f.name}: {e}")

    # From sample evaluation results (secondary source, don't override)
    if EVAL_DIR_SAMPLE.exists():
        for f in sorted(EVAL_DIR_SAMPLE.iterdir()):
            if f.suffix == ".json" and f.name != ".gitkeep":
                config = f.stem.replace("sample_scored_", "")
                if config not in results:
                    try:
                        data = json.loads(f.read_text())
                        if "raw_scored_generations" in data and len(data["raw_scored_generations"]) > 0:
                            results[config] = str(f)
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"  WARNING: Skipping corrupt file {f.name}: {e}")

    return results


def parse_config_name(config):
    """
    Parse a config name like 'codellama-13b+cot_temp0.2_input' into components.

    Returns:
        tuple: (model_name, temperature, task_type)
    """
    # Split off task type (last part: input or output)
    parts = config.rsplit("_", 1)
    task_type = parts[1]  # 'input' or 'output'

    # Split off temperature
    model_temp = parts[0]
    temp_parts = model_temp.rsplit("_", 1)
    model_name = temp_parts[0]
    temperature = temp_parts[1]  # 'temp0.2' or 'temp0.8'

    return model_name, temperature, task_type


def load_scored_generations(filepath):
    """
    Load raw_scored_generations from an evaluation result file.

    Returns:
        dict: sample_id -> list of True/False per generation
    """
    with open(filepath) as f:
        data = json.load(f)
    return data["raw_scored_generations"]


def compute_pass_at_1_per_sample(scored_gens):
    """
    Compute pass@1 for each sample.

    pass@1 = fraction of generations that are correct for that sample.
    This is equivalent to the expected probability of getting a correct
    answer when sampling 1 generation.

    Returns:
        dict: sample_id -> float (0.0 to 1.0)
    """
    result = {}
    for sample_id, scores in scored_gens.items():
        if len(scores) == 0:
            result[sample_id] = 0.0
        else:
            result[sample_id] = sum(scores) / len(scores)
    return result


def compute_any_correct_per_sample(scored_gens):
    """
    Compute binary pass (any generation correct) for each sample.

    Returns:
        dict: sample_id -> int (0 or 1)
    """
    result = {}
    for sample_id, scores in scored_gens.items():
        result[sample_id] = 1 if any(scores) else 0
    return result


def build_response_matrix(eval_files, task_filter=None, use_pass_at_1=True):
    """
    Build response matrix from evaluation files.

    Args:
        eval_files: dict of config_name -> filepath
        task_filter: 'input', 'output', or None (both)
        use_pass_at_1: if True, values are pass@1 fractions; if False, binary any-correct

    Returns:
        pd.DataFrame: rows=sample_ids, columns=config_names, values=scores
    """
    sample_ids = [f"sample_{i}" for i in range(NUM_SAMPLES)]
    matrix_data = {}

    for config, filepath in sorted(eval_files.items()):
        model, temp, task_type = parse_config_name(config)

        if task_filter and task_type != task_filter:
            continue

        scored_gens = load_scored_generations(filepath)

        if use_pass_at_1:
            scores = compute_pass_at_1_per_sample(scored_gens)
        else:
            scores = compute_any_correct_per_sample(scored_gens)

        # Build column values in order
        col_values = []
        for sid in sample_ids:
            col_values.append(scores.get(sid, np.nan))

        matrix_data[config] = col_values

    df = pd.DataFrame(matrix_data, index=sample_ids)
    df.index.name = "sample_id"
    return df


def build_model_summary(eval_files):
    """
    Build summary statistics per model configuration.

    Returns:
        pd.DataFrame with columns: model, temperature, task, pass_at_1, pass_at_5,
                                    n_samples, n_correct_any, accuracy_any
    """
    rows = []
    for config, filepath in sorted(eval_files.items()):
        model, temp, task_type = parse_config_name(config)

        with open(filepath) as f:
            data = json.load(f)

        scored_gens = data["raw_scored_generations"]
        pass_at_1 = data.get("pass_at_1", np.nan)
        pass_at_5 = data.get("pass_at_5", np.nan)

        n_samples = len(scored_gens)
        n_correct_any = sum(1 for scores in scored_gens.values() if any(scores))
        accuracy_any = n_correct_any / n_samples * 100 if n_samples > 0 else 0

        # Count average generations per sample
        n_gens_list = [len(scores) for scores in scored_gens.values()]
        avg_gens = np.mean(n_gens_list) if n_gens_list else 0

        rows.append({
            "config": config,
            "model": model,
            "temperature": temp,
            "task": task_type,
            "pass_at_1": round(pass_at_1, 2),
            "pass_at_5": round(pass_at_5, 2),
            "n_samples": n_samples,
            "n_generations_per_sample": int(avg_gens),
            "n_correct_any": n_correct_any,
            "accuracy_any_pct": round(accuracy_any, 2),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("CRUXEval Response Matrix Builder")
    print("=" * 70)

    # ── Step 1: Load ground truth ──────────────────────────────────────────
    print("\n[1/5] Loading ground truth dataset...")
    dataset = load_ground_truth()
    print(f"  Loaded {len(dataset)} samples from {DATA_PATH}")

    # ── Step 2: Discover evaluation files ──────────────────────────────────
    print("\n[2/5] Discovering evaluation result files...")
    eval_files = discover_evaluation_files()
    print(f"  Found {len(eval_files)} model configurations with evaluation results")

    for config in sorted(eval_files.keys()):
        model, temp, task = parse_config_name(config)
        print(f"    {config} -> model={model}, temp={temp}, task={task}")

    # ── Step 3: Build response matrices ────────────────────────────────────
    print("\n[3/5] Building response matrices...")

    # CRUXEval-I (input prediction)
    df_input = build_response_matrix(eval_files, task_filter="input", use_pass_at_1=True)
    print(f"  CRUXEval-I: {df_input.shape[0]} samples x {df_input.shape[1]} models")

    # CRUXEval-O (output prediction)
    df_output = build_response_matrix(eval_files, task_filter="output", use_pass_at_1=True)
    print(f"  CRUXEval-O: {df_output.shape[0]} samples x {df_output.shape[1]} models")

    # Combined (all configs)
    df_combined = build_response_matrix(eval_files, task_filter=None, use_pass_at_1=True)
    print(f"  Combined:   {df_combined.shape[0]} samples x {df_combined.shape[1]} models")

    # Binary versions (any correct)
    df_input_binary = build_response_matrix(eval_files, task_filter="input", use_pass_at_1=False)
    df_output_binary = build_response_matrix(eval_files, task_filter="output", use_pass_at_1=False)
    df_combined_binary = build_response_matrix(eval_files, task_filter=None, use_pass_at_1=False)

    # ── Step 4: Build model summary ────────────────────────────────────────
    print("\n[4/5] Building model summary...")
    df_summary = build_model_summary(eval_files)
    print(f"  Summary: {len(df_summary)} model configurations")

    # ── Step 5: Save outputs ───────────────────────────────────────────────
    print("\n[5/5] Saving outputs...")

    # Response matrices (pass@1 fraction)
    path_input = PROCESSED_DIR / "response_matrix_input.csv"
    df_input.to_csv(path_input)
    print(f"  Saved CRUXEval-I response matrix to {path_input}")

    path_output = PROCESSED_DIR / "response_matrix_output.csv"
    df_output.to_csv(path_output)
    print(f"  Saved CRUXEval-O response matrix to {path_output}")

    path_combined = PROCESSED_DIR / "response_matrix.csv"
    df_combined.to_csv(path_combined)
    print(f"  Saved combined response matrix to {path_combined}")

    # Binary response matrices
    path_input_bin = PROCESSED_DIR / "response_matrix_input_binary.csv"
    df_input_binary.to_csv(path_input_bin)
    print(f"  Saved CRUXEval-I binary matrix to {path_input_bin}")

    path_output_bin = PROCESSED_DIR / "response_matrix_output_binary.csv"
    df_output_binary.to_csv(path_output_bin)
    print(f"  Saved CRUXEval-O binary matrix to {path_output_bin}")

    path_combined_bin = PROCESSED_DIR / "response_matrix_binary.csv"
    df_combined_binary.to_csv(path_combined_bin)
    print(f"  Saved combined binary matrix to {path_combined_bin}")

    # Model summary
    path_summary = PROCESSED_DIR / "model_summary.csv"
    df_summary.to_csv(path_summary, index=False)
    print(f"  Saved model summary to {path_summary}")

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBenchmark: CRUXEval (800 Python functions)")
    print(f"Tasks:     CRUXEval-I (input prediction) + CRUXEval-O (output prediction)")
    print(f"Models:    {len(eval_files)} configurations across "
          f"{df_summary['model'].nunique()} unique models")

    print(f"\nResponse matrices (pass@1 fraction, 0.0-1.0):")
    print(f"  CRUXEval-I: {df_input.shape}")
    print(f"  CRUXEval-O: {df_output.shape}")
    print(f"  Combined:   {df_combined.shape}")

    print(f"\nBinary response matrices (any generation correct):")
    print(f"  CRUXEval-I: {df_input_binary.shape}")
    print(f"  CRUXEval-O: {df_output_binary.shape}")
    print(f"  Combined:   {df_combined_binary.shape}")

    print("\n--- Model Summary (sorted by pass@1) ---")
    print(df_summary.sort_values("pass_at_1", ascending=False)[
        ["config", "task", "pass_at_1", "pass_at_5", "accuracy_any_pct"]
    ].to_string(index=False))

    print("\n--- CRUXEval-I: Per-model pass@1 ---")
    if df_input.shape[1] > 0:
        input_means = df_input.mean().sort_values(ascending=False) * 100
        for col, val in input_means.items():
            print(f"  {col}: {val:.2f}%")

    print("\n--- CRUXEval-O: Per-model pass@1 ---")
    if df_output.shape[1] > 0:
        output_means = df_output.mean().sort_values(ascending=False) * 100
        for col, val in output_means.items():
            print(f"  {col}: {val:.2f}%")

    # Task difficulty analysis
    print("\n--- Task Difficulty (combined, fraction of models that get it right) ---")
    task_difficulty = df_combined_binary.mean(axis=1)
    print(f"  Easiest tasks (all models correct): "
          f"{(task_difficulty == 1.0).sum()}")
    print(f"  Hardest tasks (no model correct):   "
          f"{(task_difficulty == 0.0).sum()}")
    print(f"  Mean task difficulty (fraction correct): "
          f"{task_difficulty.mean():.4f}")

    print("\n" + "=" * 70)
    print("Done! All files saved to:", PROCESSED_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
