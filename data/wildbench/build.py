"""
Build WildBench response matrices from per-model per-task evaluation scores.

Data sources:
  - allenai/WildBench GitHub repo: eval_results/ directory contains per-model
    per-task LLM-as-judge scores (1-10 scale) for each of 1,024 WildBench tasks.
  - allenai/WildBench HuggingFace dataset: Task metadata (session_id, primary_tag,
    intent, checklist) for the 1,024 evaluation items.

Evaluation versions:
  - v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09: 48 models, 1024 tasks
  - v2.0625/score.v2/eval=gpt-4o-2024-05-13: 63 models, 1023 tasks (most complete)

Score format:
  - Raw scores: 1-10 integer (LLM-as-judge quality rating)
  - WB Score (rescaled): (raw - 5) * 2, range [-8, 10]

Outputs:
  - response_matrix.csv: Raw scores (models x tasks) for v2.0625 gpt-4o evaluator
  - response_matrix_v2_0522.csv: Raw scores for v2.0522 gpt-4-turbo evaluator
  - response_matrix_rescaled.csv: WB-rescaled scores for v2.0625
  - task_metadata.csv: Per-task metadata from HuggingFace dataset
  - model_summary.csv: Per-model aggregate statistics
"""

import os
import json
import urllib.request
import urllib.error
import time
import sys

import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# GitHub raw base URL for eval results
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/allenai/WildBench/main"

# Evaluation configurations to download
EVAL_CONFIGS = [
    {
        "version": "v2.0625",
        "eval_type": "score.v2",
        "evaluator": "gpt-4o-2024-05-13",
        "label": "v2.0625-gpt4o",
        "primary": True,
        "models": [
            "Athene-70B",
            "Hermes-2-Theta-Llama-3-8B",
            "Llama-2-70b-chat-hf",
            "Llama-2-7b-chat-hf",
            "Llama-3-8B-Magpie-Align-v0.1",
            "Llama-3-Instruct-8B-SimPO-ExPO",
            "Llama-3-Instruct-8B-SimPO-v0.2",
            "Llama-3-Instruct-8B-SimPO",
            "Meta-Llama-3-70B-Instruct",
            "Meta-Llama-3-8B-Instruct",
            "Mistral-7B-Instruct-v0.2",
            "Mistral-Large-2",
            "Mistral-Nemo-Instruct-2407",
            "Mixtral-8x7B-Instruct-v0.1",
            "Nous-Hermes-2-Mixtral-8x7B-DPO",
            "Phi-3-medium-128k-instruct",
            "Phi-3-mini-128k-instruct",
            "Qwen1.5-72B-Chat-greedy",
            "Qwen1.5-7B-Chat@together",
            "Qwen2-72B-Instruct",
            "SELM-Llama-3-8B-Instruct-iter-3",
            "SELM-Zephyr-7B-iter-3",
            "Starling-LM-7B-beta-ExPO",
            "Starling-LM-7B-beta",
            "Yi-1.5-34B-Chat",
            "Yi-1.5-6B-Chat",
            "Yi-1.5-9B-Chat",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "command-r-plus",
            "command-r",
            "dbrx-instruct@together",
            "deepseek-coder-v2",
            "deepseek-v2-chat-0628",
            "deepseek-v2-coder-0628",
            "deepseekv2-chat",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemma-2-27b-it@together",
            "gemma-2-2b-it",
            "gemma-2-9b-it-DPO",
            "gemma-2-9b-it-SimPO",
            "gemma-2-9b-it",
            "gemma-2b-it",
            "gemma-7b-it",
            "glm-4-9b-chat",
            "gpt-3.5-turbo-0125",
            "gpt-4-0125-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "mistral-large-2402",
            "nemotron-4-340b-instruct",
            "neo_7b_instruct_v0.1-ExPO",
            "neo_7b_instruct_v0.1",
            "reka-core-20240501",
            "reka-edge",
            "reka-flash-20240226",
            "tulu-2-dpo-70b",
            "yi-large-preview",
            "yi-large",
        ],
    },
    {
        "version": "v2.0522",
        "eval_type": "score.v2",
        "evaluator": "gpt-4-turbo-2024-04-09",
        "label": "v2.0522-gpt4turbo",
        "primary": False,
        "models": [
            "Hermes-2-Theta-Llama-3-8B",
            "Llama-2-70b-chat-hf",
            "Llama-2-7b-chat-hf",
            "Llama-3-8B-OpenHermes-243K",
            "Llama-3-8B-ShareGPT-112K",
            "Llama-3-8B-Tulu-330K",
            "Llama-3-8B-Ultrachat-200K",
            "Llama-3-8B-WildChat",
            "Llama-3-8B-WizardLM-196K",
            "Llama-3-Instruct-8B-SimPO-ExPO",
            "Llama-3-Instruct-8B-SimPO",
            "Magpie-Pro-SFT-v0.1",
            "Meta-Llama-3-70B-Instruct",
            "Meta-Llama-3-8B-Instruct",
            "Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B-Instruct-v0.1",
            "Nous-Hermes-2-Mixtral-8x7B-DPO",
            "Phi-3-medium-128k-instruct",
            "Phi-3-mini-128k-instruct",
            "Qwen1.5-72B-Chat-greedy",
            "Qwen1.5-72B-Chat",
            "Qwen1.5-7B-Chat@together",
            "Qwen2-72B-Instruct",
            "SELM-Zephyr-7B-iter-3",
            "Starling-LM-7B-beta-ExPO",
            "Starling-LM-7B-beta",
            "Yi-1.5-34B-Chat",
            "Yi-1.5-6B-Chat",
            "Yi-1.5-9B-Chat",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "command-r-plus",
            "command-r",
            "dbrx-instruct@together",
            "deepseekv2-chat",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemma-2b-it",
            "gemma-7b-it",
            "gpt-3.5-turbo-0125",
            "gpt-4-0125-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13",
            "mistral-large-2402",
            "reka-flash-20240226",
            "tulu-2-dpo-70b",
            "yi-large",
        ],
    },
]


def download_file(url, dest_path, retries=3, delay=1.0):
    """Download a file from URL with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"    FAILED after {retries} attempts: {e}")
                return False
    return False


def download_eval_results(config):
    """Download all model evaluation JSON files for a given config."""
    version = config["version"]
    eval_type = config["eval_type"]
    evaluator = config["evaluator"]
    label = config["label"]
    models = config["models"]

    # Create versioned raw directory
    raw_subdir = os.path.join(RAW_DIR, label)
    os.makedirs(raw_subdir, exist_ok=True)

    print(f"\nDownloading {label}: {len(models)} models")
    print(f"  Source: {GITHUB_RAW_BASE}/eval_results/{version}/{eval_type}/eval={evaluator}/")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, model in enumerate(models):
        dest = os.path.join(raw_subdir, f"{model}.json")

        # Skip if already downloaded
        if os.path.exists(dest) and os.path.getsize(dest) > 100:
            skipped += 1
            continue

        url = (
            f"{GITHUB_RAW_BASE}/eval_results/{version}/{eval_type}"
            f"/eval={evaluator}/{model}.json"
        )

        success = download_file(url, dest)
        if success:
            downloaded += 1
        else:
            failed += 1

        # Rate limiting: small delay between requests
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(models)}")
            time.sleep(0.5)

    print(f"  Downloaded: {downloaded}, Skipped (cached): {skipped}, Failed: {failed}")
    return raw_subdir


def download_task_metadata():
    """Download WildBench task metadata from HuggingFace."""
    print("\nDownloading task metadata from HuggingFace...")

    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/WildBench", "v2", split="test")

        rows = []
        for item in ds:
            row = {
                "session_id": item["session_id"],
                "primary_tag": item.get("primary_tag", ""),
                "intent": item.get("intent", ""),
                "length": item.get("length", None),
            }
            # Extract secondary tags
            sec_tags = item.get("secondary_tags", [])
            row["secondary_tags"] = "|".join(sec_tags) if sec_tags else ""
            # Extract checklist count
            checklist = item.get("checklist", [])
            row["n_checklist_items"] = len(checklist) if checklist else 0
            rows.append(row)

        meta_df = pd.DataFrame(rows)
        meta_path = os.path.join(RAW_DIR, "task_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"  Saved {len(meta_df)} tasks to {meta_path}")
        return meta_df

    except ImportError:
        print("  WARNING: 'datasets' library not available.")
        print("  Attempting direct parquet download...")
        return download_task_metadata_direct()
    except Exception as e:
        print(f"  WARNING: HuggingFace download failed: {e}")
        print("  Attempting direct parquet download...")
        return download_task_metadata_direct()


def download_task_metadata_direct():
    """Download task metadata via direct parquet URL as fallback."""
    url = (
        "https://huggingface.co/datasets/allenai/WildBench/resolve/main/"
        "v2/test-00000-of-00001.parquet"
    )
    dest = os.path.join(RAW_DIR, "wildbench_v2_test.parquet")

    if not os.path.exists(dest):
        print(f"  Downloading parquet from {url}")
        success = download_file(url, dest, retries=3, delay=2.0)
        if not success:
            print("  FAILED to download task metadata.")
            return None

    try:
        df = pd.read_parquet(dest)
        meta_rows = []
        for _, row in df.iterrows():
            meta_rows.append({
                "session_id": row.get("session_id", ""),
                "primary_tag": row.get("primary_tag", ""),
                "intent": row.get("intent", ""),
                "length": row.get("length", None),
                "secondary_tags": (
                    "|".join(row["secondary_tags"])
                    if isinstance(row.get("secondary_tags"), list)
                    else ""
                ),
                "n_checklist_items": (
                    len(row["checklist"])
                    if isinstance(row.get("checklist"), list)
                    else 0
                ),
            })
        meta_df = pd.DataFrame(meta_rows)
        meta_path = os.path.join(RAW_DIR, "task_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"  Saved {len(meta_df)} tasks to {meta_path}")
        return meta_df
    except Exception as e:
        print(f"  Failed to parse parquet: {e}")
        return None


def parse_model_scores(json_path):
    """Parse a model's evaluation JSON into a dict of {session_id: score}."""
    with open(json_path, "r") as f:
        data = json.load(f)

    scores = {}
    for item in data:
        sid = item.get("session_id", "")
        raw_score = item.get("score", None)
        if raw_score is not None and str(raw_score) not in ["", "-1"]:
            try:
                scores[sid] = int(raw_score)
            except (ValueError, TypeError):
                try:
                    scores[sid] = float(raw_score)
                except (ValueError, TypeError):
                    scores[sid] = None
        else:
            scores[sid] = None
    return scores


def build_response_matrix(config, raw_subdir):
    """Build a response matrix from downloaded JSON files."""
    label = config["label"]
    models = config["models"]

    print(f"\n{'='*60}")
    print(f"  Building response matrix: {label}")
    print(f"{'='*60}")

    # Parse all model scores
    all_scores = {}
    all_session_ids = set()
    successful_models = []

    for model in models:
        json_path = os.path.join(raw_subdir, f"{model}.json")
        if not os.path.exists(json_path):
            print(f"  WARNING: Missing {model}.json, skipping")
            continue

        try:
            scores = parse_model_scores(json_path)
            all_scores[model] = scores
            all_session_ids.update(scores.keys())
            successful_models.append(model)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Failed to parse {model}.json: {e}")

    # Sort session IDs for consistent ordering
    session_ids = sorted(all_session_ids)

    # Build matrix DataFrame
    matrix_data = {}
    for model in successful_models:
        model_scores = all_scores[model]
        matrix_data[model] = [model_scores.get(sid, None) for sid in session_ids]

    matrix_df = pd.DataFrame(matrix_data, index=session_ids)
    matrix_df.index.name = "session_id"

    n_models = len(successful_models)
    n_tasks = len(session_ids)
    total_cells = n_models * n_tasks
    n_valid = matrix_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0

    # Score statistics (on valid cells only)
    all_valid_scores = matrix_df.values[~np.isnan(
        matrix_df.values.astype(float, copy=True)
    )]

    print(f"  Models:          {n_models}")
    print(f"  Tasks:           {n_tasks}")
    print(f"  Matrix dims:     {n_models} x {n_tasks}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  Valid cells:     {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells:   {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:       {fill_rate*100:.1f}%")

    if len(all_valid_scores) > 0:
        print(f"\n  Score distribution (raw 1-10):")
        print(f"    Mean:   {np.nanmean(all_valid_scores):.2f}")
        print(f"    Median: {np.nanmedian(all_valid_scores):.1f}")
        print(f"    Std:    {np.nanstd(all_valid_scores):.2f}")
        print(f"    Min:    {np.nanmin(all_valid_scores):.0f}")
        print(f"    Max:    {np.nanmax(all_valid_scores):.0f}")

        # Score histogram
        print(f"\n  Score histogram:")
        for score_val in range(1, 11):
            count = np.sum(all_valid_scores == score_val)
            pct = count / len(all_valid_scores) * 100
            bar = "#" * int(pct)
            print(f"    {score_val:2d}: {count:6,} ({pct:5.1f}%) {bar}")

    # Per-model stats
    per_model_mean = matrix_df.mean(axis=0)
    print(f"\n  Per-model mean score (raw):")
    print(
        f"    Min:    {per_model_mean.min():.2f} ({per_model_mean.idxmin()})"
    )
    print(
        f"    Max:    {per_model_mean.max():.2f} ({per_model_mean.idxmax()})"
    )
    print(f"    Median: {per_model_mean.median():.2f}")
    print(f"    Std:    {per_model_mean.std():.2f}")

    # Per-task stats
    per_task_mean = matrix_df.mean(axis=1)
    print(f"\n  Per-task mean score (raw):")
    print(f"    Min:    {per_task_mean.min():.2f}")
    print(f"    Max:    {per_task_mean.max():.2f}")
    print(f"    Median: {per_task_mean.median():.2f}")
    print(f"    Std:    {per_task_mean.std():.2f}")

    # Task difficulty distribution (based on mean score)
    easy = (per_task_mean >= 8).sum()
    medium = ((per_task_mean >= 5) & (per_task_mean < 8)).sum()
    hard = (per_task_mean < 5).sum()
    print(f"\n  Task difficulty distribution (by mean score):")
    print(f"    Easy (>=8):    {easy}")
    print(f"    Medium (5-8):  {medium}")
    print(f"    Hard (<5):     {hard}")

    # Save raw score matrix
    output_name = "response_matrix.csv" if config["primary"] else f"response_matrix_{label}.csv"
    output_path = os.path.join(PROCESSED_DIR, output_name)
    # Transpose so rows=models, columns=tasks (matching BigCodeBench pattern)
    matrix_df_t = matrix_df.T
    matrix_df_t.index.name = "Model"
    matrix_df_t.to_csv(output_path)
    print(f"\n  Saved raw matrix: {output_path}")

    # Also save WB-rescaled matrix for primary config
    if config["primary"]:
        rescaled_df = (matrix_df - 5) * 2
        rescaled_df_t = rescaled_df.T
        rescaled_df_t.index.name = "Model"
        rescaled_path = os.path.join(PROCESSED_DIR, "response_matrix_rescaled.csv")
        rescaled_df_t.to_csv(rescaled_path)
        print(f"  Saved rescaled matrix: {rescaled_path}")

    return {
        "label": label,
        "n_models": n_models,
        "n_tasks": n_tasks,
        "fill_rate": fill_rate,
        "model_names": successful_models,
        "per_model_mean": per_model_mean,
        "matrix_df": matrix_df,
        "primary": config["primary"],
    }


def build_model_summary(all_stats):
    """Build a comprehensive model summary CSV."""
    rows = []

    # Collect model stats from all configs
    model_data = {}
    for stats in all_stats:
        for model in stats["model_names"]:
            if model not in model_data:
                model_data[model] = {}
            model_data[model][stats["label"]] = float(
                stats["per_model_mean"][model]
            )

    for model in sorted(model_data.keys()):
        row = {"model": model}

        for stats in all_stats:
            label = stats["label"]
            if label in model_data[model]:
                raw_mean = model_data[model][label]
                row[f"{label}_raw_mean"] = round(raw_mean, 3)
                row[f"{label}_wb_score"] = round((raw_mean - 5) * 2, 3)
            else:
                row[f"{label}_raw_mean"] = None
                row[f"{label}_wb_score"] = None

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Sort by primary config's WB score
    primary_label = [s["label"] for s in all_stats if s["primary"]][0]
    sort_col = f"{primary_label}_wb_score"
    if sort_col in summary_df.columns:
        summary_df = summary_df.sort_values(
            sort_col, ascending=False, na_position="last"
        )

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total unique models: {len(summary_df)}")
    for stats in all_stats:
        label = stats["label"]
        col = f"{label}_raw_mean"
        n = summary_df[col].notna().sum()
        print(f"  Models with {label} scores: {n}")

    print(f"\n  Top 15 models (by {primary_label} WB Score):")
    top = summary_df.dropna(subset=[sort_col]).head(15)
    for _, r in top.iterrows():
        wb = r[sort_col]
        raw = r[f"{primary_label}_raw_mean"]
        print(f"    {r['model']:45s}  raw={raw:.2f}  WB={wb:+.2f}")

    print(f"\n  Bottom 5 models (by {primary_label} WB Score):")
    bottom = summary_df.dropna(subset=[sort_col]).tail(5)
    for _, r in bottom.iterrows():
        wb = r[sort_col]
        raw = r[f"{primary_label}_raw_mean"]
        print(f"    {r['model']:45s}  raw={raw:.2f}  WB={wb:+.2f}")

    print(f"\n  Saved: {output_path}")
    return summary_df


def save_task_metadata_processed(task_meta_df, primary_stats):
    """Save processed task metadata with per-task statistics."""
    if task_meta_df is None:
        print("\n  No task metadata available, skipping.")
        return

    matrix_df = primary_stats["matrix_df"]

    # Compute per-task stats from primary matrix
    task_stats = pd.DataFrame({
        "session_id": matrix_df.index,
        "mean_score": matrix_df.mean(axis=1).values,
        "std_score": matrix_df.std(axis=1).values,
        "min_score": matrix_df.min(axis=1).values,
        "max_score": matrix_df.max(axis=1).values,
        "n_models_scored": matrix_df.notna().sum(axis=1).values,
    })

    # Merge with metadata
    merged = task_stats.merge(task_meta_df, on="session_id", how="left")

    output_path = os.path.join(PROCESSED_DIR, "task_metadata.csv")
    merged.to_csv(output_path, index=False)
    print(f"\n  Task metadata saved: {output_path}")

    # Category breakdown
    if "primary_tag" in merged.columns:
        print(f"\n  Task category distribution:")
        tag_counts = merged["primary_tag"].value_counts()
        for tag, count in tag_counts.items():
            mean = merged[merged["primary_tag"] == tag]["mean_score"].mean()
            print(f"    {tag:35s}  n={count:4d}  mean_score={mean:.2f}")


def main():
    print("WildBench Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download evaluation results
    print("STEP 1: Downloading evaluation results from GitHub")
    print("-" * 60)

    raw_subdirs = {}
    for config in EVAL_CONFIGS:
        raw_subdirs[config["label"]] = download_eval_results(config)

    # Step 2: Download task metadata
    print("\nSTEP 2: Downloading task metadata")
    print("-" * 60)
    task_meta_df = download_task_metadata()

    # Step 3: Build response matrices
    print("\nSTEP 3: Building response matrices")
    print("-" * 60)

    all_stats = []
    primary_stats = None
    for config in EVAL_CONFIGS:
        raw_subdir = raw_subdirs[config["label"]]
        stats = build_response_matrix(config, raw_subdir)
        all_stats.append(stats)
        if config["primary"]:
            primary_stats = stats

    # Step 4: Build model summary
    print("\nSTEP 4: Building model summary")
    print("-" * 60)
    build_model_summary(all_stats)

    # Step 5: Save enriched task metadata
    print("\nSTEP 5: Saving enriched task metadata")
    print("-" * 60)
    if primary_stats is not None:
        save_task_metadata_processed(task_meta_df, primary_stats)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    if primary_stats:
        print(f"\n  PRIMARY response matrix ({primary_stats['label']}):")
        print(
            f"    Dimensions: {primary_stats['n_models']} models x "
            f"{primary_stats['n_tasks']} tasks"
        )
        print(f"    Fill rate:  {primary_stats['fill_rate']*100:.1f}%")
        print(f"    Score type: Raw LLM-as-judge (1-10)")
        print(f"    Evaluator:  GPT-4o (2024-05-13)")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")

    print(f"\n  Score interpretation:")
    print(f"    Raw scores: 1 (worst) to 10 (best)")
    print(f"    WB Score:   (raw - 5) * 2, range [-8, +10]")
    print(f"    WB Score 0 means average quality (raw=5)")


if __name__ == "__main__":
    main()
