#!/usr/bin/env python3
"""
Build task-level response matrix from ThaiLLM Leaderboard results.

Source: https://huggingface.co/datasets/ThaiLLM-Leaderboard/results

The Thai LLM Leaderboard evaluates models across four categories:
  - MC (Multiple Choice): m3exam, thaiexam
  - NLU (Natural Language Understanding): xnli, xcopa, wisesight sentiment, belebele
  - NLG (Natural Language Generation): xl_sum, flores200 translation, iapp_squad QA
  - LLM (Chat/Instruction): Math, Reasoning, Extraction, Roleplay, Writing,
    Social Science, STEM, Coding, Knowledge III

Outputs:
  - response_matrix.csv      : Models (rows) x Tasks (columns), values = scores
  - task_metadata.csv         : Task-level metadata (category, metric type)
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

CLONE_DIR = Path("/tmp/thai-leaderboard-results")

# Categories in the leaderboard
CATEGORIES = ["MC", "NLU", "NLG", "LLM"]


def extract_score(task_name, task_data):
    """Extract the primary score from a task result dict.
    Returns (score, metric_name)."""
    if not isinstance(task_data, dict):
        return None, None

    # For MC/NLU tasks: accuracy
    if "accuracy" in task_data:
        return task_data["accuracy"], "accuracy"
    # For LLM tasks: avg_rating
    if "avg_rating" in task_data:
        return task_data["avg_rating"], "avg_rating"
    # For NLG tasks, pick the most informative metric
    # BLEU for translation, ROUGE for summarization/QA
    metric_priority = ["SacreBLEU", "BLEU", "ROUGE1", "ROUGEL", "ROUGE2", "chrF++"]
    for metric in metric_priority:
        if metric in task_data:
            return task_data[metric], metric
    # Fallback: first numeric value
    for key, val in task_data.items():
        if isinstance(val, (int, float)):
            return val, key
    return None, None


def load_all_results():
    """Load all model results across all categories."""
    print("Loading ThaiLLM Leaderboard results...")

    # Structure: {model_name: {task_full_name: score}}
    model_task_scores = {}
    model_metadata = {}
    task_info = {}  # task_full_name -> {category, metric, ...}

    for category in CATEGORIES:
        cat_dir = CLONE_DIR / category
        if not cat_dir.exists():
            print(f"  WARNING: Category directory not found: {cat_dir}")
            continue

        model_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
        print(f"  Category {category}: {len(model_dirs)} models")

        for model_dir in model_dirs:
            result_file = model_dir / "results.json"
            if not result_file.exists():
                continue

            try:
                with open(result_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            config = data.get("config", {})
            model_name = config.get("model_name", model_dir.name)
            # Clean up model name
            short_name = model_name.replace("api/", "").split("/")[-1]

            if short_name not in model_task_scores:
                model_task_scores[short_name] = {}
                model_metadata[short_name] = {
                    "model": short_name,
                    "full_name": model_name,
                    "source": "thai-leaderboard",
                    "categories": set(),
                }

            model_metadata[short_name]["categories"].add(category)

            results = data.get("results", {})
            for task_name, task_data in results.items():
                score, metric = extract_score(task_name, task_data)
                if score is not None:
                    # Prefix with category to avoid name collisions
                    full_task = f"{category}/{task_name}"
                    model_task_scores[short_name][full_task] = score

                    if full_task not in task_info:
                        task_info[full_task] = {
                            "category": category,
                            "metric": metric,
                            "raw_task_name": task_name,
                        }

    print(f"  Loaded scores for {len(model_task_scores)} unique models")
    return model_task_scores, model_metadata, task_info


def build_matrices(model_task_scores, model_metadata, task_info):
    """Build response matrix, task metadata, and model summary."""
    all_tasks = sorted(task_info.keys())
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
        info = task_info[task]
        n_models = response_df[task].notna().sum()
        mean_score = response_df[task].mean()
        task_meta_rows.append({
            "task_id": task,
            "raw_task_name": info["raw_task_name"],
            "language": "Thai",
            "category": info["category"],
            "primary_metric": info["metric"],
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
            "categories_evaluated": ",".join(sorted(meta["categories"])),
            "n_tasks_evaluated": len(vals),
            "overall_mean": round(sum(vals) / len(vals), 4) if vals else None,
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
    lines.append("ThaiLLM Leaderboard Results Collection")
    lines.append("=" * 60)
    lines.append("Source: https://huggingface.co/datasets/ThaiLLM-Leaderboard/results")
    lines.append("Language: Thai")
    lines.append("")
    lines.append(f"Total models: {len(model_summary_df)}")
    lines.append(f"Total tasks: {len(task_meta_df)}")
    density = response_df.notna().sum().sum()
    total = response_df.shape[0] * response_df.shape[1]
    lines.append(f"Matrix density: {density} / {total} ({density/total*100:.1f}%)")
    lines.append("")

    lines.append("Tasks by category:")
    for cat in CATEGORIES:
        subset = task_meta_df[task_meta_df["category"] == cat]
        if len(subset) > 0:
            lines.append(f"\n  {cat} ({len(subset)} tasks):")
            for _, row in subset.iterrows():
                lines.append(f"    {row['raw_task_name']:45s}  metric={row['primary_metric']:12s}  "
                             f"mean={row['mean_score']}  n_models={row['n_models_evaluated']}")

    lines.append("")
    lines.append("Top 10 models (by tasks evaluated):")
    top_models = model_summary_df.sort_values("n_tasks_evaluated", ascending=False).head(10)
    for _, row in top_models.iterrows():
        lines.append(f"  {row['model']:50s}  categories={row['categories_evaluated']:15s}  "
                     f"n_tasks={row['n_tasks_evaluated']}")

    summary_text = "\n".join(lines)
    summary_path = PROCESSED_DIR / "collection_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved collection_summary.txt: {summary_path}")
    print("\n" + summary_text)


def main():
    print("ThaiLLM Leaderboard Response Matrix Builder")
    print("=" * 50)

    model_task_scores, model_metadata, task_info = load_all_results()
    response_df, task_meta_df, model_summary_df = build_matrices(
        model_task_scores, model_metadata, task_info
    )
    save_outputs(response_df, task_meta_df, model_summary_df)
    write_collection_summary(response_df, task_meta_df, model_summary_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
