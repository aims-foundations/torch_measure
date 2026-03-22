#!/usr/bin/env python3
"""
Build task-level response matrix from La Leaderboard results.

La Leaderboard evaluates models on Spanish, Catalan, Basque, and Galician tasks.
Source: https://huggingface.co/datasets/la-leaderboard/results

This script reads the aggregate per-task accuracy from each model's result JSON
and builds a (models x tasks) response matrix, plus task and model metadata.

Outputs:
  - response_matrix.csv      : Models (rows) x Tasks (columns), values = accuracy
  - task_metadata.csv         : Task-level metadata (language, category, metric)
  - model_summary.csv         : Per-model overall averages
  - collection_summary.txt    : Human-readable summary
"""

import json
import warnings
from collections import defaultdict
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

# Clone location
CLONE_DIR = Path("/tmp/la-leaderboard-results")

# ──────────────────────────────────────────────────────────────────────
# Language inference from task names
# ──────────────────────────────────────────────────────────────────────
TASK_LANGUAGE_MAP = {
    # Spanish tasks
    "copa_es": "es", "openbookqa_es": "es", "paws_es_spanish_bench": "es",
    "wnli_es": "es", "xnli_es_spanish_bench": "es", "xquad_es": "es",
    "xstorycloze_es": "es", "xlsum_es": "es", "mgsm_direct_es_spanish_bench": "es",
    "spalawex": "es", "fake_news_es": "es", "offendes": "es",
    "crows_pairs_spanish": "es",
    # Catalan tasks
    "piqa_ca": "ca", "copa_ca": "ca", "siqa_ca": "ca", "openbookqa_ca": "ca",
    "paws_ca": "ca", "wnli_ca": "ca", "xnli_ca": "ca", "xquad_ca": "ca",
    "xstorycloze_ca": "ca", "mgsm_direct_ca": "ca", "catcola": "ca",
    "catalanqa": "ca", "teca": "ca", "cabreu": "ca", "coqcat": "ca",
    "parafraseja": "ca", "escola": "ca",
    "arc_ca_catalan_bench": "ca",
    "belebele_cat_Latn": "ca",
    # Basque tasks
    "wnli_eu": "eu", "xcopa_eu": "eu", "xnli_eu_native": "eu",
    "xstorycloze_eu": "eu", "mgsm_native_cot_eu": "eu",
    "bec2016eu": "eu", "eus_exams_eu": "eu", "eus_proficiency": "eu",
    "eus_reading": "eu", "eus_trivia": "eu", "epec_koref_bin": "eu",
    "qnlieu": "eu", "wiceu": "eu", "belebele_eus_Latn": "eu",
    # Galician tasks
    "xnli_gl": "gl", "xstorycloze_gl": "gl", "mgsm_direct_gl": "gl",
    "openbookqa_gl": "gl", "paws_gl": "gl", "parafrases_gl": "gl",
    "galcola": "gl", "noticia": "gl", "summarization_gl": "gl",
    "belebele_glg_Latn": "gl",
    # Multi-language tasks
    "aquas": "ca",  # Catalan QA
    "ragquas": "ca",  # Catalan RAG QA
    "bhtc_v2": "eu",  # Basque topic classification
    "clindiagnoses": "es",  # Spanish clinical NLP
    "clintreates": "es",  # Spanish clinical NLP
    "humorqa": "ca",  # Catalan humor
    "teleia": "eu",  # Basque
    "vaxx_stance": "eu",  # Basque
    # Catalan subtasks
    "arc_ca_challenge": "ca", "arc_ca_easy": "ca",
    "cabreu_abstractive": "ca", "cabreu_extractive": "ca", "cabreu_extreme": "ca",
    # Basque exam subtasks (eus_exams_eu_*)
    "bertaqa_eu": "eu",
    "teleia_cervantes_ave": "eu", "teleia_pce": "eu", "teleia_siele": "eu",
    # Spanish belebele
    "belebele_spa_Latn": "es",
    # Aggregate leaderboard scores
    "laleaderboard": "multi", "laleaderboard_ca": "ca",
    "laleaderboard_es": "es", "laleaderboard_eu": "eu", "laleaderboard_gl": "gl",
}

# Dynamically add all eus_exams_eu_* subtasks as Basque
_EUS_EXAM_PREFIXES = [
    "eus_exams_eu_ejadministrari", "eus_exams_eu_ejlaguntza",
    "eus_exams_eu_ejlaguntzaile", "eus_exams_eu_ejteknikari",
    "eus_exams_eu_opebilbaoeu", "eus_exams_eu_opeehuadmineu",
    "eus_exams_eu_opeehuauxeu", "eus_exams_eu_opeehubiblioeu",
    "eus_exams_eu_opeehuderechoeu", "eus_exams_eu_opeehueconomicaseu",
    "eus_exams_eu_opeehuempresarialeseu", "eus_exams_eu_opeehusubalternoeu",
    "eus_exams_eu_opeehutecnicoeu", "eus_exams_eu_opeehuteknikarib",
    "eus_exams_eu_opegasteizkoudala", "eus_exams_eu_opeosakiadmineu",
    "eus_exams_eu_opeosakiauxenfeu", "eus_exams_eu_opeosakiauxeu",
    "eus_exams_eu_opeosakiceladoreu", "eus_exams_eu_opeosakienfeu",
    "eus_exams_eu_opeosakioperarioeu", "eus_exams_eu_opeosakitecnicoeu",
    "eus_exams_eu_opeosakivarioseu", "eus_exams_eu_osakidetza1e",
    "eus_exams_eu_osakidetza2e", "eus_exams_eu_osakidetza3e",
    "eus_exams_eu_osakidetza5e", "eus_exams_eu_osakidetza6e",
    "eus_exams_eu_osakidetza7e",
]
for _task in _EUS_EXAM_PREFIXES:
    TASK_LANGUAGE_MAP[_task] = "eu"

LANGUAGE_NAMES = {"es": "Spanish", "ca": "Catalan", "eu": "Basque", "gl": "Galician", "multi": "Multilingual"}


def get_primary_metric(task_results):
    """Extract the primary accuracy metric from a task result dict."""
    # Priority order: acc, acc_norm, exact_match (flexible), sas, etc.
    metric_priority = [
        "acc,none",
        "acc_norm,none",
        "exact_match,flexible-extract",
        "exact_match,remove_whitespace",
        "exact_match,strict-match",
        "sas_cross_encoder,none",
        "sas_encoder,none",
        "mcc,none",
        "rouge2,none",
    ]
    for metric in metric_priority:
        if metric in task_results:
            return task_results[metric], metric
    # Fall back to any numeric value
    for key, val in task_results.items():
        if isinstance(val, (int, float)) and "stderr" not in key and "shot" not in key:
            return val, key
    return None, None


def load_all_results():
    """Load all model result files from the clone directory."""
    print("Loading La Leaderboard results...")

    model_task_scores = {}  # model_name -> {task -> score}
    model_metadata = {}  # model_name -> metadata dict
    task_metrics = {}  # task_name -> metric used

    result_files = sorted(CLONE_DIR.glob("**/results_*.json"))
    print(f"  Found {len(result_files)} result files")

    for fpath in result_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  WARNING: Could not parse {fpath}")
            continue

        config = data.get("config", {})
        model_name = config.get("model_name", "")
        if not model_name:
            # Try to infer from path
            parts = fpath.relative_to(CLONE_DIR).parts
            if len(parts) >= 2:
                model_name = f"{parts[0]}/{parts[1]}"
            else:
                continue

        results = data.get("results", {})
        if not results:
            continue

        scores = {}
        for task_name, task_data in results.items():
            if not isinstance(task_data, dict):
                continue
            score, metric = get_primary_metric(task_data)
            if score is not None:
                scores[task_name] = score
                if task_name not in task_metrics:
                    task_metrics[task_name] = metric

        if scores:
            # Use short model name for readability
            short_name = model_name.split("/")[-1] if "/" in model_name else model_name
            model_task_scores[short_name] = scores
            model_metadata[short_name] = {
                "model": short_name,
                "full_name": model_name,
                "model_dtype": config.get("model_dtype", ""),
                "source": "la-leaderboard",
            }

    print(f"  Loaded scores for {len(model_task_scores)} models")
    return model_task_scores, model_metadata, task_metrics


def build_matrices(model_task_scores, model_metadata, task_metrics):
    """Build response matrix, task metadata, and model summary DataFrames."""

    # Get all tasks and models
    all_tasks = sorted(set(t for scores in model_task_scores.values() for t in scores))
    all_models = sorted(model_task_scores.keys())

    print(f"\n  Matrix dimensions: {len(all_models)} models x {len(all_tasks)} tasks")

    # Build response matrix (models as rows, tasks as columns)
    matrix_data = []
    for model in all_models:
        row = {"model": model}
        for task in all_tasks:
            row[task] = model_task_scores[model].get(task, None)
        matrix_data.append(row)

    response_df = pd.DataFrame(matrix_data)
    response_df = response_df.set_index("model")

    # Build task metadata
    task_meta_rows = []
    for task in all_tasks:
        lang = TASK_LANGUAGE_MAP.get(task, "unknown")
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        metric = task_metrics.get(task, "unknown")
        n_models_evaluated = response_df[task].notna().sum()
        mean_score = response_df[task].mean()

        task_meta_rows.append({
            "task_id": task,
            "language_code": lang,
            "language": lang_name,
            "primary_metric": metric,
            "n_models_evaluated": n_models_evaluated,
            "mean_score": round(mean_score, 4) if pd.notna(mean_score) else None,
        })

    task_meta_df = pd.DataFrame(task_meta_rows)

    # Build model summary
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

    # Response matrix
    rmat_path = PROCESSED_DIR / "response_matrix.csv"
    response_df.to_csv(rmat_path)
    print(f"  Saved response_matrix.csv: {rmat_path}")
    print(f"    Shape: {response_df.shape}")

    # Task metadata
    tmeta_path = PROCESSED_DIR / "task_metadata.csv"
    task_meta_df.to_csv(tmeta_path, index=False)
    print(f"  Saved task_metadata.csv: {tmeta_path}")

    # Model summary
    msumm_path = PROCESSED_DIR / "model_summary.csv"
    model_summary_df.to_csv(msumm_path, index=False)
    print(f"  Saved model_summary.csv: {msumm_path}")

    return rmat_path, tmeta_path, msumm_path


def write_collection_summary(response_df, task_meta_df, model_summary_df):
    """Write a human-readable summary."""
    lines = []
    lines.append("La Leaderboard Results Collection")
    lines.append("=" * 60)
    lines.append(f"Source: https://huggingface.co/datasets/la-leaderboard/results")
    lines.append(f"Languages: Spanish, Catalan, Basque, Galician")
    lines.append("")
    lines.append(f"Total models: {len(model_summary_df)}")
    lines.append(f"Total tasks: {len(task_meta_df)}")
    lines.append(f"Matrix density: {response_df.notna().sum().sum()} / "
                 f"{response_df.shape[0] * response_df.shape[1]} "
                 f"({response_df.notna().sum().sum() / (response_df.shape[0] * response_df.shape[1]) * 100:.1f}%)")
    lines.append("")

    lines.append("Tasks by language:")
    for lang in sorted(task_meta_df["language"].unique()):
        subset = task_meta_df[task_meta_df["language"] == lang]
        lines.append(f"  {lang}: {len(subset)} tasks")
        for _, row in subset.iterrows():
            lines.append(f"    - {row['task_id']} (metric: {row['primary_metric']}, "
                         f"mean: {row['mean_score']}, n_models: {row['n_models_evaluated']})")

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
    print("La Leaderboard Response Matrix Builder")
    print("=" * 50)

    model_task_scores, model_metadata, task_metrics = load_all_results()
    response_df, task_meta_df, model_summary_df = build_matrices(
        model_task_scores, model_metadata, task_metrics
    )
    save_outputs(response_df, task_meta_df, model_summary_df)
    write_collection_summary(response_df, task_meta_df, model_summary_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
