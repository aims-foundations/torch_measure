#!/usr/bin/env python3
"""
Build task-level response matrix from Open Ko-LLM Leaderboard results.

Source: https://huggingface.co/datasets/open-ko-llm-leaderboard/results

The Open Ko-LLM Leaderboard evaluates models on Korean NLP benchmarks:
  - ko_eqbench: Korean Emotional Intelligence Benchmark
  - ko_gpqa_diamond_zeroshot: Korean Graduate-level Q&A
  - ko_gsm8k: Korean grade school math
  - ko_ifeval: Korean instruction following evaluation
  - ko_winogrande: Korean Winograd schema challenge
  - kornat_common: Korean common knowledge
  - kornat_harmless: Korean harmlessness evaluation
  - kornat_helpful: Korean helpfulness evaluation
  - kornat_social: Korean social value alignment (A-SVA metric)

Outputs:
  - response_matrix.csv      : Models (rows) x Tasks (columns), values = scores
  - task_metadata.csv         : Task-level metadata
  - model_summary.csv         : Per-model overall info
  - collection_summary.txt    : Human-readable summary
"""

import json
import warnings
from pathlib import Path

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

CLONE_DIR = Path("/tmp/ko-leaderboard-results")

# ──────────────────────────────────────────────────────────────────────
# Task metadata
# ──────────────────────────────────────────────────────────────────────
TASK_INFO = {
    "ko_eqbench": {
        "description": "Korean Emotional Intelligence Benchmark",
        "category": "emotional_intelligence",
        "primary_metric": "eqbench,none",
    },
    "ko_gpqa_diamond_zeroshot": {
        "description": "Korean Graduate-level Q&A (GPQA Diamond)",
        "category": "knowledge_qa",
        "primary_metric": "acc_norm,none",
    },
    "ko_gsm8k": {
        "description": "Korean Grade School Math (GSM8K)",
        "category": "math",
        "primary_metric": "exact_match,flexible-extract",
    },
    "ko_ifeval": {
        "description": "Korean Instruction Following Evaluation",
        "category": "instruction_following",
        "primary_metric": "prompt_level_strict_acc,none",
    },
    "ko_winogrande": {
        "description": "Korean Winograd Schema Challenge",
        "category": "commonsense_reasoning",
        "primary_metric": "acc,none",
    },
    "kornat_common": {
        "description": "Korean Common Knowledge",
        "category": "knowledge",
        "primary_metric": "acc_norm,none",
    },
    "kornat_harmless": {
        "description": "Korean Harmlessness Evaluation",
        "category": "safety",
        "primary_metric": "acc_norm,none",
    },
    "kornat_helpful": {
        "description": "Korean Helpfulness Evaluation",
        "category": "helpfulness",
        "primary_metric": "acc_norm,none",
    },
    "kornat_social": {
        "description": "Korean Social Value Alignment",
        "category": "social_values",
        "primary_metric": "A-SVA,none",
    },
}


def get_primary_score(task_name, task_data):
    """Extract primary score for a given task."""
    if not isinstance(task_data, dict):
        return None, None

    info = TASK_INFO.get(task_name, {})
    primary = info.get("primary_metric")

    if primary and primary in task_data:
        val = task_data[primary]
        if isinstance(val, (int, float)):
            return val, primary

    # Fallback: try standard metrics in priority order
    metric_priority = [
        "acc,none", "acc_norm,none",
        "exact_match,flexible-extract", "exact_match,strict-match",
        "prompt_level_strict_acc,none", "prompt_level_loose_acc,none",
        "eqbench,none", "A-SVA,none",
    ]
    for metric in metric_priority:
        if metric in task_data:
            val = task_data[metric]
            if isinstance(val, (int, float)):
                return val, metric

    # Last resort: any numeric non-stderr value
    for key, val in task_data.items():
        if isinstance(val, (int, float)) and "stderr" not in key:
            return val, key

    return None, None


def load_all_results():
    """Load all model result files."""
    print("Loading Open Ko-LLM Leaderboard results...")

    model_task_scores = {}
    model_metadata = {}
    task_metrics_used = {}

    result_files = sorted(CLONE_DIR.glob("**/result_*.json"))
    print(f"  Found {len(result_files)} result files")

    skipped = 0
    for fpath in result_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue

        config = data.get("config_general", data.get("config", {}))
        model_name = config.get("model_name", "")
        if not model_name:
            parts = fpath.relative_to(CLONE_DIR).parts
            if len(parts) >= 2:
                model_name = f"{parts[0]}/{parts[1]}"

        results = data.get("results", {})
        if not results:
            skipped += 1
            continue

        short_name = model_name.split("/")[-1] if "/" in model_name else model_name

        scores = {}
        for task_name, task_data in results.items():
            score, metric = get_primary_score(task_name, task_data)
            if score is not None:
                scores[task_name] = score
                if task_name not in task_metrics_used:
                    task_metrics_used[task_name] = metric

        if not scores:
            skipped += 1
            continue

        # If duplicate, keep the one with more tasks or higher version
        if short_name in model_task_scores:
            if len(scores) <= len(model_task_scores[short_name]):
                continue

        model_task_scores[short_name] = scores
        model_metadata[short_name] = {
            "model": short_name,
            "full_name": model_name,
            "model_dtype": config.get("model_dtype", ""),
            "source": "ko-leaderboard",
        }

    print(f"  Loaded scores for {len(model_task_scores)} unique models")
    if skipped:
        print(f"  Skipped {skipped} files (parse errors or empty)")
    return model_task_scores, model_metadata, task_metrics_used


def build_matrices(model_task_scores, model_metadata, task_metrics_used):
    """Build response matrix, task metadata, and model summary."""
    all_tasks = sorted(set(t for scores in model_task_scores.values() for t in scores))
    all_models = sorted(model_task_scores.keys())

    print(f"\n  Matrix dimensions: {len(all_models)} models x {len(all_tasks)} tasks")

    # Response matrix
    matrix_data = []
    for model in all_models:
        row = {"model": model}
        for task in all_tasks:
            row[task] = model_task_scores[model].get(task, None)
        matrix_data.append(row)

    response_df = pd.DataFrame(matrix_data).set_index("model")

    # Task metadata
    task_meta_rows = []
    for task in all_tasks:
        info = TASK_INFO.get(task, {})
        n_models = response_df[task].notna().sum()
        mean_score = response_df[task].mean()
        task_meta_rows.append({
            "task_id": task,
            "language": "Korean",
            "category": info.get("category", "unknown"),
            "description": info.get("description", task),
            "primary_metric": task_metrics_used.get(task, "unknown"),
            "n_models_evaluated": n_models,
            "mean_score": round(mean_score, 4) if pd.notna(mean_score) else None,
        })
    task_meta_df = pd.DataFrame(task_meta_rows)

    # Model summary
    model_summary_rows = []
    for model in all_models:
        scores = model_task_scores[model]
        vals = [v for v in scores.values() if v is not None]
        meta = model_metadata[model]
        model_summary_rows.append({
            "model": model,
            "full_name": meta["full_name"],
            "source": meta["source"],
            "overall_mean": round(sum(vals) / len(vals), 4) if vals else None,
            "n_tasks_evaluated": len(vals),
            "model_dtype": meta["model_dtype"],
        })
    model_summary_df = pd.DataFrame(model_summary_rows)

    return response_df, task_meta_df, model_summary_df


def save_outputs(response_df, task_meta_df, model_summary_df):
    """Save all output files."""
    print("\nSaving outputs...")

    rmat_path = PROCESSED_DIR / "response_matrix.csv"
    response_df.to_csv(rmat_path)
    print(f"  Saved response_matrix.csv: {rmat_path} ({response_df.shape})")

    tmeta_path = PROCESSED_DIR / "task_metadata.csv"
    task_meta_df.to_csv(tmeta_path, index=False)
    print(f"  Saved task_metadata.csv: {tmeta_path}")

    msumm_path = PROCESSED_DIR / "model_summary.csv"
    model_summary_df.to_csv(msumm_path, index=False)
    print(f"  Saved model_summary.csv: {msumm_path}")


def write_collection_summary(response_df, task_meta_df, model_summary_df):
    """Write human-readable summary."""
    lines = []
    lines.append("Open Ko-LLM Leaderboard Results Collection")
    lines.append("=" * 60)
    lines.append("Source: https://huggingface.co/datasets/open-ko-llm-leaderboard/results")
    lines.append("Language: Korean")
    lines.append("")
    lines.append(f"Total models: {len(model_summary_df)}")
    lines.append(f"Total tasks: {len(task_meta_df)}")
    density = response_df.notna().sum().sum()
    total = response_df.shape[0] * response_df.shape[1]
    lines.append(f"Matrix density: {density} / {total} ({density/total*100:.1f}%)")
    lines.append("")

    lines.append("Tasks:")
    for _, row in task_meta_df.iterrows():
        lines.append(f"  {row['task_id']:35s}  category={row['category']:25s}  "
                     f"metric={row['primary_metric']:30s}  mean={row['mean_score']}  "
                     f"n_models={row['n_models_evaluated']}")

    lines.append("")
    lines.append("Top 20 models by overall mean score:")
    top_models = model_summary_df.sort_values("overall_mean", ascending=False).head(20)
    for _, row in top_models.iterrows():
        lines.append(f"  {row['model']:50s}  mean={row['overall_mean']:.4f}  "
                     f"n_tasks={row['n_tasks_evaluated']}")

    summary_text = "\n".join(lines)
    summary_path = PROCESSED_DIR / "collection_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved collection_summary.txt: {summary_path}")
    print("\n" + summary_text)


def main():
    print("Open Ko-LLM Leaderboard Response Matrix Builder")
    print("=" * 52)

    model_task_scores, model_metadata, task_metrics_used = load_all_results()
    response_df, task_meta_df, model_summary_df = build_matrices(
        model_task_scores, model_metadata, task_metrics_used
    )
    save_outputs(response_df, task_meta_df, model_summary_df)
    write_collection_summary(response_df, task_meta_df, model_summary_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
