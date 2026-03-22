#!/usr/bin/env python3
"""
Build task-level response matrix from Portuguese LLM Leaderboard results.

Source: https://huggingface.co/datasets/eduagarcia-temp/llm_pt_leaderboard_raw_results

This leaderboard evaluates models on Portuguese NLP tasks including:
  - enem_challenge: Brazilian university entrance exam
  - bluex: Brazilian university entrance exam (UNICAMP)
  - oab_exams: Brazilian bar exam
  - assin2_rte: Textual entailment (Portuguese)
  - assin2_sts: Semantic textual similarity (Portuguese)
  - faquad_nli: NLI from FAQ QA (Portuguese)
  - hatebr_offensive: Offensive language detection (Brazilian Portuguese)
  - portuguese_hate_speech: Hate speech detection (Portuguese)
  - tweetsentbr: Tweet sentiment (Brazilian Portuguese)

Outputs:
  - response_matrix.csv      : Models (rows) x Tasks (columns), values = accuracy/score
  - task_metadata.csv         : Task-level metadata
  - model_summary.csv         : Per-model overall averages
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

CLONE_DIR = Path("/tmp/pt-leaderboard-results")

# ──────────────────────────────────────────────────────────────────────
# Task metadata
# ──────────────────────────────────────────────────────────────────────
TASK_INFO = {
    "enem_challenge": {
        "description": "Brazilian university entrance exam (ENEM)",
        "category": "knowledge_exam",
        "language_variety": "Brazilian Portuguese",
    },
    "bluex": {
        "description": "UNICAMP entrance exam questions",
        "category": "knowledge_exam",
        "language_variety": "Brazilian Portuguese",
    },
    "oab_exams": {
        "description": "Brazilian bar exam (OAB)",
        "category": "knowledge_exam",
        "language_variety": "Brazilian Portuguese",
    },
    "assin2_rte": {
        "description": "Textual entailment (ASSIN 2)",
        "category": "nli",
        "language_variety": "Portuguese",
    },
    "assin2_sts": {
        "description": "Semantic textual similarity (ASSIN 2)",
        "category": "sts",
        "language_variety": "Portuguese",
    },
    "faquad_nli": {
        "description": "NLI from FAQ-based QA",
        "category": "nli",
        "language_variety": "Portuguese",
    },
    "hatebr_offensive": {
        "description": "Offensive language detection (HateBR)",
        "category": "hate_speech",
        "language_variety": "Brazilian Portuguese",
    },
    "portuguese_hate_speech": {
        "description": "Hate speech detection",
        "category": "hate_speech",
        "language_variety": "Portuguese",
    },
    "tweetsentbr": {
        "description": "Tweet sentiment analysis (Brazilian)",
        "category": "sentiment",
        "language_variety": "Brazilian Portuguese",
    },
}


def load_all_results():
    """Load all model result files."""
    print("Loading Portuguese LLM Leaderboard results...")

    model_task_scores = {}
    model_metadata = {}

    # Find all result JSON files (not raw/ subdirectories)
    result_files = sorted(CLONE_DIR.glob("**/results_*.json"))
    print(f"  Found {len(result_files)} result files")

    for fpath in result_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  WARNING: Could not parse {fpath}")
            continue

        config = data.get("config_general", data.get("config", {}))
        model_name = config.get("model_name", "")
        if not model_name:
            parts = fpath.relative_to(CLONE_DIR).parts
            if len(parts) >= 2:
                model_name = f"{parts[0]}/{parts[1]}"

        results = data.get("results", {})
        if not results:
            continue

        # Extract grouped results if available
        grouped = results.get("all_grouped", {})
        if grouped:
            scores = {k: v for k, v in grouped.items() if isinstance(v, (int, float))}
        else:
            # Try to extract from flat "all" dict or individual task dicts
            scores = {}
            all_results = results.get("all", results)
            for key, val in all_results.items():
                if isinstance(val, dict):
                    # Look for acc metric
                    for metric in ["acc,all", "acc,none", "acc_norm,none"]:
                        if metric in val:
                            task_name = key.split("|")[1] if "|" in key else key
                            scores[task_name] = val[metric]
                            break
                elif isinstance(val, (int, float)):
                    # Direct task -> score mapping
                    if key not in ("all_grouped_average", "all_grouped_npm"):
                        scores[key] = val

        if scores:
            short_name = model_name.split("/")[-1] if "/" in model_name else model_name

            # If we already have this model, keep the one with more tasks
            if short_name in model_task_scores:
                if len(scores) <= len(model_task_scores[short_name]):
                    continue

            model_task_scores[short_name] = scores
            model_metadata[short_name] = {
                "model": short_name,
                "full_name": model_name,
                "model_dtype": config.get("model_dtype", ""),
                "source": "pt-leaderboard",
                "avg_score": results.get("all_grouped_average", None),
            }

    print(f"  Loaded scores for {len(model_task_scores)} models")
    return model_task_scores, model_metadata


def build_matrices(model_task_scores, model_metadata):
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
            "language": "Portuguese",
            "language_variety": info.get("language_variety", "Portuguese"),
            "category": info.get("category", "unknown"),
            "description": info.get("description", task),
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
            "leaderboard_avg": meta.get("avg_score"),
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
    lines.append("Portuguese LLM Leaderboard Results Collection")
    lines.append("=" * 60)
    lines.append("Source: https://huggingface.co/datasets/eduagarcia-temp/llm_pt_leaderboard_raw_results")
    lines.append("Language: Portuguese (Brazilian and European)")
    lines.append("")
    lines.append(f"Total models: {len(model_summary_df)}")
    lines.append(f"Total tasks: {len(task_meta_df)}")
    density = response_df.notna().sum().sum()
    total = response_df.shape[0] * response_df.shape[1]
    lines.append(f"Matrix density: {density} / {total} ({density/total*100:.1f}%)")
    lines.append("")

    lines.append("Tasks:")
    for _, row in task_meta_df.iterrows():
        lines.append(f"  {row['task_id']:30s}  category={row['category']:15s}  "
                     f"variety={row['language_variety']:25s}  mean={row['mean_score']}  "
                     f"n_models={row['n_models_evaluated']}")

    lines.append("")
    lines.append("Top 10 models by overall mean score:")
    top_models = model_summary_df.sort_values("overall_mean", ascending=False).head(10)
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
    print("Portuguese LLM Leaderboard Response Matrix Builder")
    print("=" * 55)

    model_task_scores, model_metadata = load_all_results()
    response_df, task_meta_df, model_summary_df = build_matrices(
        model_task_scores, model_metadata
    )
    save_outputs(response_df, task_meta_df, model_summary_df)
    write_collection_summary(response_df, task_meta_df, model_summary_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
