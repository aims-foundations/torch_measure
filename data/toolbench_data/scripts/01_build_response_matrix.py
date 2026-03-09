"""
Build ToolBench response matrices from publicly available per-item evaluation data.

Data sources:
  1. StableToolBench baselines (per-item binary pass/fail):
     - HuggingFace: stabletoolbench/baselines (data_baselines.zip, ~37 MB)
     - 10 model configurations (5 models x 2 strategies: CoT, DFS)
     - 6 test scenarios: G1_instruction, G1_category, G1_tool,
       G2_category, G2_instruction, G3_instruction
     - 765 unique task instances across all scenarios
     - Each item has a binary "win" field (pass/fail)

  2. StableToolBench leaderboard (aggregate scores):
     - HuggingFace: stabletoolbench/StableToolBench_data/leaderboard_data.json
     - SolvablePassRate and SolvableWinRate scores per model per scenario

  3. SambaNova ToolBench paper Table 9 (aggregate success rates):
     - Xu et al., "On the Tool Manipulation Capability of Open-source LLMs"
     - arXiv:2305.16504
     - 27 models x 8 tool domains (3-shot scenario)
     - Scores are success rates (%), not per-item binary

Outputs:
  - response_matrix.csv: Binary (models x tasks) matrix from StableToolBench
    per-item data, combining all scenarios into one matrix
  - response_matrix_by_scenario.csv: Binary matrix with scenario prefix on task IDs
  - stabletoolbench_leaderboard.csv: Aggregate leaderboard scores
  - sambanova_toolbench_paper_table9.csv: Paper Table 9 aggregate scores (27 models)
  - model_summary.csv: Combined per-model statistics

Note on data availability:
  The StableToolBench baselines provide TRUE per-item binary pass/fail data for
  10 model configurations. The SambaNova ToolBench paper Table 9 provides only
  aggregate success rates (not per-item), so it is stored separately.
"""

import os
import sys
import json
import re
import glob
import subprocess
import urllib.request
import zipfile

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Data URLs
BASELINES_URL = (
    "https://huggingface.co/datasets/stabletoolbench/baselines"
    "/resolve/main/data_baselines.zip"
)
LEADERBOARD_URL = (
    "https://huggingface.co/datasets/stabletoolbench/StableToolBench_data"
    "/resolve/main/leaderboard_data.json"
)

# StableToolBench scenarios
SCENARIOS = [
    "G1_instruction", "G1_category", "G1_tool",
    "G2_category", "G2_instruction", "G3_instruction",
]

# SambaNova ToolBench paper Table 9 data (arXiv:2305.16504)
# 27 models evaluated in 3-shot scenario across 8 tool domains
# Scores are success rates (%) except VirtualHome (executability / LCS)
# and WebShop (reward). Tabletop is few-shot only.
PAPER_TABLE9_TASKS = [
    "OpenWeather", "TheCatAPI", "HomeSearch", "TripBooking",
    "GoogleSheets", "VirtualHome_exec", "VirtualHome_LCS",
    "WebShop_Long", "WebShop_Short", "Tabletop",
]

PAPER_TABLE9_DATA = {
    # Closed-source models
    "gpt4": [93.0, 96.0, 97.0, 96.7, 62.9, 23.0, 23.5, 0.0, None, 81.0],
    "text-davinci-003": [99.0, 98.0, 97.0, 89.2, 62.9, 31.0, 25.1, 0.0, None, 66.7],
    "gpt-3.5-turbo": [90.0, 92.0, 80.0, 85.8, 51.4, 20.0, 18.9, 0.0, 1.8, 33.3],
    "text-curie-001": [8.0, 58.0, 6.0, 6.7, 1.4, 12.0, 4.1, 0.0, 0.0, 1.0],
    # Open-source models
    "llama-65b": [90.0, 80.0, 84.0, 65.8, 32.9, 32.0, 20.3, 0.0, 41.2, 30.5],
    "llama-30b": [78.0, 84.0, 66.0, 45.0, 37.1, 27.0, 21.7, 0.0, 30.6, 34.3],
    "llama-13b": [70.0, 74.0, 45.0, 35.8, 5.7, 28.0, 18.9, 0.0, 27.6, 17.1],
    "llama-13b-alpaca": [62.0, 43.0, 44.0, 40.8, 11.4, 1.0, 1.6, 0.0, 2.7, 9.5],
    "starcoder": [91.0, 84.0, 82.0, 51.7, 48.0, 23.0, 19.4, 2.6, 0.0, 21.9],
    "starcoderbase": [90.0, 86.0, 79.0, 63.3, 42.9, 24.0, 16.3, 5.8, 23.1, 17.1],
    "codegen-16B-nl": [51.0, 75.0, 37.0, 21.7, 7.1, 43.0, 18.0, 0.0, 0.0, 16.2],
    "codegen-16B-multi": [56.0, 75.0, 47.0, 7.5, 21.4, 31.0, 14.1, 0.0, 0.5, 8.6],
    "codegen-16B-mono": [63.7, 72.0, 52.0, 28.3, 31.5, 28.0, 15.7, 1.5, 6.6, 15.2],
    "bloomz": [58.0, 85.0, 36.0, 22.5, 14.3, 9.0, 4.9, 0.0, 1.0, 1.0],
    "opt-iml-30b": [44.0, 48.0, 5.0, 3.3, 2.9, 13.0, 8.3, 0.0, 0.0, 1.0],
    "opt-30b": [46.0, 35.0, 2.0, 3.3, 8.6, 24.0, 11.7, 0.0, 0.0, 1.0],
    "opt-iml-1.3b": [20.0, 28.0, 0.0, 0.0, 4.3, 13.0, 3.1, 0.0, 0.0, 1.0],
    "opt-1.3b": [18.0, 30.0, 0.0, 0.0, 1.4, 31.0, 9.7, 0.0, 0.0, 1.0],
    "neox-20b": [55.0, 69.0, 27.0, 10.8, 18.6, 28.0, 15.3, 0.0, 8.8, 6.7],
    "GPT-NeoXT-Chat-Base-20B": [43.0, 73.0, 28.0, 10.8, 4.3, 26.0, 13.1, 0.0, 0.7, 7.6],
    "pythia-12b": [53.0, 65.0, 12.0, 0.8, 11.4, 17.0, 12.1, 0.0, 0.0, 1.9],
    "dolly-v2-12b": [0.0, 1.0, 10.0, 5.0, 7.1, 11.0, 8.9, 0.0, 0.0, 7.6],
    "pythia-6.9b": [41.0, 72.0, 8.0, 7.5, 4.3, 29.0, 14.0, 0.0, 0.0, 8.6],
    "pythia-2.8b": [49.0, 54.0, 7.0, 3.3, 12.9, 24.0, 14.8, 0.0, 0.0, 7.6],
    "pythia-1.4b": [37.0, 48.0, 4.0, 5.0, 10.0, 22.0, 10.7, 0.0, 5.2, 7.6],
    "stablelm-base-alpha-7b": [22.0, 47.0, 0.0, 0.0, 4.3, 28.0, 10.3, 0.0, 0.0, 2.9],
    "stablelm-tuned-alpha-7b": [23.0, 38.0, 0.0, 0.0, 1.4, 26.0, 7.3, 0.0, 0.0, 3.8],
    "stablelm-base-alpha-3b": [6.0, 28.0, 0.0, 0.0, 1.4, 29.0, 5.3, 0.0, 0.0, 1.0],
    "stablelm-tuned-alpha-3b": [14.0, 31.0, 0.0, 0.8, 0.0, 8.0, 5.6, 0.0, 0.0, 1.0],
}


def download_file(url, dest_path, desc=""):
    """Download a file if not already present."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Already exists: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")
        return True
    print(f"  Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Downloaded: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return False


def download_raw_data():
    """Download all raw data files."""
    print("Downloading raw data...")

    # 1. StableToolBench baselines
    zip_path = os.path.join(RAW_DIR, "data_baselines.zip")
    ok1 = download_file(BASELINES_URL, zip_path, "StableToolBench baselines")

    # Extract if needed
    baselines_dir = os.path.join(RAW_DIR, "data_baselines")
    if ok1 and not os.path.exists(baselines_dir):
        print("  Extracting data_baselines.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Skip __MACOSX
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(RAW_DIR, members=members)
        print(f"  Extracted to {baselines_dir}")

    # 2. StableToolBench leaderboard
    lb_path = os.path.join(RAW_DIR, "leaderboard_data.json")
    ok2 = download_file(LEADERBOARD_URL, lb_path, "StableToolBench leaderboard")

    return ok1, ok2


def parse_task_id(filename):
    """Extract numeric task ID from a result filename."""
    match = re.match(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None


def build_per_item_response_matrices():
    """Build binary response matrices from StableToolBench per-item data.

    Returns dict of scenario -> (matrix_df, model_names, task_ids).
    """
    baselines_dir = os.path.join(RAW_DIR, "data_baselines")
    if not os.path.exists(baselines_dir):
        print("ERROR: data_baselines not found. Run download first.")
        return {}

    # Discover model directories
    model_dirs = sorted([
        d for d in os.listdir(baselines_dir)
        if os.path.isdir(os.path.join(baselines_dir, d)) and not d.startswith('.')
    ])

    # Map directory names to clean model names
    model_name_map = {
        "chatgpt_cot": "GPT-3.5-Turbo-0613 (CoT)",
        "chatgpt_dfs": "GPT-3.5-Turbo-0613 (DFS)",
        "gpt-3.5-turbo-1106_cot": "GPT-3.5-Turbo-1106 (CoT)",
        "gpt-3.5-turbo-1106_dfs": "GPT-3.5-Turbo-1106 (DFS)",
        "gpt-4-0613_cot": "GPT-4-0613 (CoT)",
        "gpt-4-0613_dfs": "GPT-4-0613 (DFS)",
        "gpt-4-turbo-preview_cot": "GPT-4-Turbo-Preview (CoT)",
        "gpt-4-turbo-preview_dfs": "GPT-4-Turbo-Preview (DFS)",
        "toolllama_cot": "ToolLLaMA v2 (CoT)",
        "toolllama_dfs": "ToolLLaMA v2 (DFS)",
    }

    results_by_scenario = {}

    for scenario in SCENARIOS:
        print(f"\n  Processing scenario: {scenario}")

        # Collect all task IDs across models for this scenario
        all_task_ids = set()
        model_results = {}

        for model_dir in model_dirs:
            scenario_path = os.path.join(baselines_dir, model_dir, scenario)
            if not os.path.exists(scenario_path):
                continue

            model_name = model_name_map.get(model_dir, model_dir)
            results = {}

            for json_file in glob.glob(os.path.join(scenario_path, "*.json")):
                task_id = parse_task_id(json_file)
                if task_id is None:
                    continue
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    # A small number of files are truncated/malformed;
                    # skip them rather than crashing.
                    continue
                results[task_id] = 1 if data.get("win", False) else 0
                all_task_ids.add(task_id)

            model_results[model_name] = results

        # Build matrix
        task_ids = sorted(all_task_ids)
        model_names = sorted(model_results.keys())

        matrix = np.full((len(model_names), len(task_ids)), np.nan)
        for i, model_name in enumerate(model_names):
            for j, task_id in enumerate(task_ids):
                if task_id in model_results[model_name]:
                    matrix[i, j] = model_results[model_name][task_id]

        task_cols = [str(tid) for tid in task_ids]
        matrix_df = pd.DataFrame(matrix, index=model_names, columns=task_cols)
        matrix_df.index.name = "Model"

        results_by_scenario[scenario] = (matrix_df, model_names, task_ids)
        print(f"    {len(model_names)} models x {len(task_ids)} tasks")

    return results_by_scenario


def build_combined_response_matrix(results_by_scenario):
    """Combine all scenario matrices into one with prefixed task IDs."""
    all_dfs = []
    for scenario, (matrix_df, _, _) in sorted(results_by_scenario.items()):
        # Prefix columns with scenario name
        prefixed = matrix_df.copy()
        prefixed.columns = [f"{scenario}_{col}" for col in prefixed.columns]
        all_dfs.append(prefixed)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, axis=1)
    # Sort columns
    combined = combined[sorted(combined.columns)]
    return combined


def print_matrix_stats(matrix_df, label):
    """Print comprehensive statistics for a response matrix."""
    matrix = matrix_df.values
    model_names = list(matrix_df.index)
    n_models, n_tasks = matrix.shape
    total_cells = n_models * n_tasks
    n_valid = int(np.sum(~np.isnan(matrix)))
    n_pass = int(np.nansum(matrix))
    n_fail = n_valid - n_pass
    fill_rate = n_valid / total_cells if total_cells > 0 else 0
    mean_pass_rate = np.nanmean(matrix) if n_valid > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Models:          {n_models}")
    print(f"  Tasks:           {n_tasks}")
    print(f"  Matrix dims:     {n_models} x {n_tasks}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  Valid cells:     {n_valid:,}")
    print(f"  Pass cells:      {n_pass:,} ({n_pass/n_valid*100:.1f}%)"
          if n_valid > 0 else "  Pass cells:      0")
    print(f"  Fail cells:      {n_fail:,} ({n_fail/n_valid*100:.1f}%)"
          if n_valid > 0 else "  Fail cells:      0")
    print(f"  Fill rate:       {fill_rate*100:.1f}%")
    print(f"  Mean pass rate:  {mean_pass_rate*100:.1f}%")

    # Per-model stats
    per_model_pass = np.nanmean(matrix, axis=1)
    print(f"\n  Per-model pass rate:")
    best_idx = np.nanargmax(per_model_pass)
    worst_idx = np.nanargmin(per_model_pass)
    print(f"    Min:    {per_model_pass[worst_idx]*100:.1f}%"
          f" ({model_names[worst_idx]})")
    print(f"    Max:    {per_model_pass[best_idx]*100:.1f}%"
          f" ({model_names[best_idx]})")
    print(f"    Median: {np.nanmedian(per_model_pass)*100:.1f}%")
    print(f"    Std:    {np.nanstd(per_model_pass)*100:.1f}%")

    # Per-task stats
    per_task_solve = np.nanmean(matrix, axis=0)
    print(f"\n  Per-task solve rate:")
    print(f"    Min:    {np.nanmin(per_task_solve)*100:.1f}%")
    print(f"    Max:    {np.nanmax(per_task_solve)*100:.1f}%")
    print(f"    Median: {np.nanmedian(per_task_solve)*100:.1f}%")
    print(f"    Std:    {np.nanstd(per_task_solve)*100:.1f}%")

    # Task difficulty distribution
    unsolved = int(np.sum(per_task_solve == 0))
    easy = int(np.sum(per_task_solve > 0.9))
    hard = int(np.sum(per_task_solve < 0.1))
    print(f"\n  Task difficulty distribution:")
    print(f"    Unsolved (0%):   {unsolved}")
    print(f"    Hard (<10%):     {hard}")
    print(f"    Easy (>90%):     {easy}")

    return {
        "n_models": n_models,
        "n_tasks": n_tasks,
        "mean_pass_rate": mean_pass_rate,
        "fill_rate": fill_rate,
        "model_names": model_names,
        "per_model_pass": per_model_pass,
    }


def build_leaderboard_csv():
    """Parse the StableToolBench leaderboard JSON into a CSV."""
    lb_path = os.path.join(RAW_DIR, "leaderboard_data.json")
    if not os.path.exists(lb_path):
        print("WARNING: leaderboard_data.json not found, skipping.")
        return None

    with open(lb_path, 'r') as f:
        data = json.load(f)

    rows = []
    for entry in data.get("SolvablePassRateScores", []):
        method = entry.get("Method", "")
        # Skip placeholder entries
        if method.startswith("Method Name"):
            continue
        scores = entry.get("Scores", {})
        row = {"Model": method, "Metric": "SolvablePassRate"}
        for key, val in scores.items():
            if not key.endswith(" SE"):
                row[key] = val
        rows.append(row)

    for entry in data.get("SolvableWinRateScores", []):
        method = entry.get("Method", "")
        if method.startswith("Method Name"):
            continue
        scores = entry.get("Scores", {})
        row = {"Model": method, "Metric": "SolvableWinRate"}
        for key, val in scores.items():
            if not key.endswith(" SE"):
                row[key] = val
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    output_path = os.path.join(PROCESSED_DIR, "stabletoolbench_leaderboard.csv")
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  StableToolBench Leaderboard")
    print(f"{'='*60}")
    print(f"  Entries: {len(df)}")
    print(f"  Models:  {df['Model'].nunique()}")
    print(f"  Metrics: {df['Metric'].unique().tolist()}")
    print(f"\n  SolvablePassRate scores:")
    spr = df[df["Metric"] == "SolvablePassRate"].copy()
    if "Average" in spr.columns:
        spr = spr.sort_values("Average", ascending=False)
        for _, r in spr.iterrows():
            print(f"    {r['Model']:40s}  Avg={r['Average']:.1f}%")
    print(f"\n  Saved: {output_path}")
    return df


def build_paper_table9_csv():
    """Build CSV from the SambaNova ToolBench paper Table 9."""
    rows = []
    for model_name, scores in PAPER_TABLE9_DATA.items():
        row = {"Model": model_name}
        for task, score in zip(PAPER_TABLE9_TASKS, scores):
            row[task] = score
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    output_path = os.path.join(
        PROCESSED_DIR, "sambanova_toolbench_paper_table9.csv"
    )
    df.to_csv(output_path)

    print(f"\n{'='*60}")
    print(f"  SambaNova ToolBench Paper Table 9 (arXiv:2305.16504)")
    print(f"{'='*60}")
    print(f"  Models:  {len(df)}")
    print(f"  Tasks:   {len(PAPER_TABLE9_TASKS)}")
    print(f"  Source:  3-shot evaluation across 8 tool domains")
    print(f"  Note:    Aggregate success rates (%), NOT per-item binary")

    # Compute average excluding None
    numeric_cols = [c for c in df.columns if c not in ["Model"]]
    df_numeric = df[numeric_cols].astype(float)
    avg_scores = df_numeric.mean(axis=1, skipna=True)
    sorted_models = avg_scores.sort_values(ascending=False)

    print(f"\n  Top 10 models (by average score):")
    for model_name in sorted_models.head(10).index:
        avg = sorted_models[model_name]
        print(f"    {model_name:35s}  avg={avg:.1f}%")

    print(f"\n  Bottom 5 models (by average score):")
    for model_name in sorted_models.tail(5).index:
        avg = sorted_models[model_name]
        print(f"    {model_name:35s}  avg={avg:.1f}%")

    print(f"\n  Saved: {output_path}")
    return df


def build_model_summary(per_item_stats, combined_stats):
    """Build a comprehensive model summary CSV."""
    rows = []

    # StableToolBench per-item models
    if combined_stats:
        for i, model_name in enumerate(combined_stats["model_names"]):
            rows.append({
                "model": model_name,
                "source": "StableToolBench",
                "data_type": "per_item_binary",
                "n_tasks": combined_stats["n_tasks"],
                "pass_rate": combined_stats["per_model_pass"][i],
            })

    # Paper Table 9 models
    for model_name, scores in PAPER_TABLE9_DATA.items():
        valid_scores = [s for s in scores if s is not None]
        avg_score = np.mean(valid_scores) / 100.0 if valid_scores else None
        rows.append({
            "model": model_name,
            "source": "SambaNova_paper",
            "data_type": "aggregate_pct",
            "n_tasks": len(PAPER_TABLE9_TASKS),
            "pass_rate": avg_score,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("pass_rate", ascending=False, na_position="last")

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total model entries: {len(df)}")
    stb = df[df["source"] == "StableToolBench"]
    paper = df[df["source"] == "SambaNova_paper"]
    print(f"  StableToolBench models (per-item): {len(stb)}")
    print(f"  SambaNova paper models (aggregate): {len(paper)}")
    print(f"\n  Top 10 models overall (by pass rate):")
    for _, r in df.head(10).iterrows():
        pr = r["pass_rate"] * 100 if pd.notna(r["pass_rate"]) else 0
        print(f"    {r['model']:40s}  {pr:.1f}%  [{r['source']}]")

    print(f"\n  Saved: {output_path}")
    return df


def main():
    print("ToolBench Response Matrix Builder")
    print("=" * 60)
    print()
    print("Data sources:")
    print("  1. StableToolBench baselines (per-item binary pass/fail)")
    print("  2. StableToolBench leaderboard (aggregate scores)")
    print("  3. SambaNova ToolBench paper Table 9 (aggregate %)")
    print()

    # Step 1: Download raw data
    ok_baselines, ok_leaderboard = download_raw_data()

    # Step 2: Build per-item response matrices from StableToolBench
    per_item_stats = {}
    combined_stats = None

    if ok_baselines:
        print("\n" + "=" * 60)
        print("  BUILDING PER-ITEM RESPONSE MATRICES")
        print("=" * 60)

        results_by_scenario = build_per_item_response_matrices()

        # Save per-scenario matrices
        for scenario, (matrix_df, _, _) in sorted(results_by_scenario.items()):
            output_name = f"response_matrix_{scenario}.csv"
            output_path = os.path.join(PROCESSED_DIR, output_name)
            matrix_df.to_csv(output_path)

            stats = print_matrix_stats(
                matrix_df, f"StableToolBench - {scenario}"
            )
            per_item_stats[scenario] = stats
            print(f"\n  Saved: {output_path}")

        # Build combined matrix (all scenarios)
        combined_df = build_combined_response_matrix(results_by_scenario)
        if combined_df is not None:
            output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
            combined_df.to_csv(output_path)
            combined_stats = print_matrix_stats(
                combined_df,
                "StableToolBench - ALL SCENARIOS COMBINED (PRIMARY)"
            )
            print(f"\n  Saved: {output_path}")

    # Step 3: Build leaderboard CSV
    if ok_leaderboard:
        build_leaderboard_csv()

    # Step 4: Build paper Table 9 CSV
    build_paper_table9_csv()

    # Step 5: Build model summary
    build_model_summary(per_item_stats, combined_stats)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")

    if combined_stats:
        print(f"\n  PRIMARY response matrix (StableToolBench per-item):")
        print(f"    Dimensions: {combined_stats['n_models']} models"
              f" x {combined_stats['n_tasks']} tasks")
        print(f"    Fill rate:  {combined_stats['fill_rate']*100:.1f}%")
        print(f"    Mean pass:  {combined_stats['mean_pass_rate']*100:.1f}%")
        print(f"    Data type:  Binary pass/fail per item")

    print(f"\n  SECONDARY data (SambaNova paper Table 9):")
    print(f"    Dimensions: {len(PAPER_TABLE9_DATA)} models"
          f" x {len(PAPER_TABLE9_TASKS)} tasks")
    print(f"    Data type:  Aggregate success rates (%)")
    print(f"    Note:       NOT per-item binary; stored separately")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:50s}  {size_kb:.1f} KB")

    print(f"\n  Data provenance:")
    print(f"    StableToolBench: huggingface.co/datasets/stabletoolbench/baselines")
    print(f"    Leaderboard:     huggingface.co/datasets/stabletoolbench/"
          f"StableToolBench_data")
    print(f"    Paper:           arxiv.org/abs/2305.16504")


if __name__ == "__main__":
    main()
