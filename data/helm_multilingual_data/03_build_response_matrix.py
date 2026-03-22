#!/usr/bin/env python3
"""Build response matrices from downloaded HELM multilingual per-instance data.

Reads per_instance_stats.json files from each downloaded run directory,
extracts item_id, model, and exact_match metric, and pivots into response
matrices following torch_measure conventions.

Outputs (to processed/):
  - response_matrix.csv          : items (rows) x models (cols), binary 0/1 (combined)
  - response_matrix_afr.csv      : African MMLU + Winogrande only
  - response_matrix_thaiexam.csv : ThaiExam only
  - response_matrix_cleva.csv    : CLEVA only
  - task_metadata.csv            : item_id, language, benchmark, subject/category
  - model_summary.csv            : per-model accuracy

Usage:
    python 03_build_response_matrix.py [--data-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_run_dir_name(run_dir_name: str) -> dict:
    """Parse run directory name to extract task, model, language, subject, etc.

    Examples:
      mmlu_clinical_afr:subject=clinical_knowledge,lang=af,method=multiple_choice_joint,model=anthropic_claude-3-5-haiku-20241022
      winogrande_afr:lang=am,method=multiple_choice_joint,model=anthropic_claude-3-7-sonnet-20250219
      thai_exam:exam=onet,method=multiple_choice_joint,model=openai_gpt-4o-2024-05-13
      cleva:task=classical_chinese_understanding,version=v1,prompt_id=3,model=openai_gpt-4-0613
    """
    result = {}

    # Split on first colon to get benchmark prefix and params
    if ":" not in run_dir_name:
        return {"raw_name": run_dir_name}

    benchmark_prefix, params_str = run_dir_name.split(":", 1)
    result["benchmark_prefix"] = benchmark_prefix

    # Parse key=value pairs (handling commas within model names that contain commas)
    # Use a regex that handles the specific HELM naming convention
    params = {}
    # Split on comma-separated key=value pairs
    # But model names can contain underscores and hyphens, not commas
    parts = params_str.split(",")
    i = 0
    while i < len(parts):
        if "=" in parts[i]:
            key, val = parts[i].split("=", 1)
            params[key] = val
        i += 1

    result["model"] = params.get("model", "unknown")
    result["method"] = params.get("method", "")

    # Determine benchmark and extract task-specific fields
    if benchmark_prefix.startswith("mmlu_clinical_afr"):
        result["benchmark"] = "mmlu_afr"
        result["subject"] = params.get("subject", "")
        result["language"] = params.get("lang", "")
    elif benchmark_prefix.startswith("winogrande_afr"):
        result["benchmark"] = "winogrande_afr"
        result["subject"] = "winogrande"
        result["language"] = params.get("lang", "")
    elif benchmark_prefix.startswith("thai_exam"):
        result["benchmark"] = "thai_exam"
        result["subject"] = params.get("exam", "")
        result["language"] = "th"
    elif benchmark_prefix.startswith("cleva"):
        result["benchmark"] = "cleva"
        result["subject"] = params.get("task", "")
        subtask = params.get("subtask", "")
        if subtask:
            result["subject"] = f"{result['subject']}_{subtask}"
        result["language"] = "zh"
    else:
        result["benchmark"] = benchmark_prefix
        result["subject"] = ""
        result["language"] = ""

    return result


def extract_exact_match(stats_list: list[dict]) -> float | None:
    """Extract the exact_match metric from a stats list.

    Prefers exact_match, falls back to quasi_exact_match.
    """
    for stat in stats_list:
        name_info = stat.get("name", {})
        if name_info.get("name") == "exact_match":
            return stat.get("mean")

    # Fallback to quasi_exact_match
    for stat in stats_list:
        name_info = stat.get("name", {})
        if name_info.get("name") == "quasi_exact_match":
            return stat.get("mean")

    return None


def process_project(project_dir: Path, project_name: str) -> tuple[list[dict], dict]:
    """Process all runs for a project.

    Returns:
        records: list of {item_id, model, score, benchmark, subject, language}
        item_metadata: dict of item_id -> {language, benchmark, subject}
    """
    records = []
    item_metadata = {}

    if not project_dir.exists():
        print(f"  WARNING: {project_dir} does not exist, skipping")
        return records, item_metadata

    # Find all version directories
    version_dirs = sorted([d for d in project_dir.iterdir() if d.is_dir()])
    if not version_dirs:
        print(f"  WARNING: No version directories found in {project_dir}")
        return records, item_metadata

    print(f"  Found {len(version_dirs)} version(s): {[v.name for v in version_dirs]}")

    for version_dir in version_dirs:
        run_dirs = sorted([d for d in version_dir.iterdir() if d.is_dir()])
        print(f"    {version_dir.name}: {len(run_dirs)} runs")

        for run_dir in run_dirs:
            stats_file = run_dir / "per_instance_stats.json"
            if not stats_file.exists():
                continue

            run_info = parse_run_dir_name(run_dir.name)
            model = run_info.get("model", "unknown")
            benchmark = run_info.get("benchmark", project_name)
            subject = run_info.get("subject", "")
            language = run_info.get("language", "")

            try:
                with open(stats_file) as f:
                    per_instance = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"      WARNING: Could not read {stats_file}: {e}")
                continue

            for entry in per_instance:
                instance_id = entry.get("instance_id", "")
                score = extract_exact_match(entry.get("stats", []))

                if score is None:
                    continue

                # Create a globally unique item_id
                # Format: {benchmark}_{subject}_{language}_{instance_id}
                if subject:
                    global_item_id = f"{benchmark}_{subject}_{language}_{instance_id}"
                else:
                    global_item_id = f"{benchmark}_{language}_{instance_id}"

                records.append({
                    "item_id": global_item_id,
                    "model": model,
                    "score": int(round(score)),  # Binary 0/1
                    "benchmark": benchmark,
                    "subject": subject,
                    "language": language,
                })

                if global_item_id not in item_metadata:
                    item_metadata[global_item_id] = {
                        "item_id": global_item_id,
                        "language": language,
                        "benchmark": benchmark,
                        "subject": subject,
                        "source_instance_id": instance_id,
                        "source_project": project_name,
                    }

    return records, item_metadata


def build_response_matrix(records: list[dict]) -> pd.DataFrame:
    """Pivot records into a response matrix: items x models."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # If there are duplicates (same item + model), take the max score
    pivot = df.pivot_table(
        index="item_id",
        columns="model",
        values="score",
        aggfunc="max",
    )

    # Sort columns (models) alphabetically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Sort rows by item_id
    pivot = pivot.sort_index()

    # Reset index so item_id is a column
    pivot = pivot.reset_index()

    # Rename the index column
    pivot = pivot.rename(columns={"item_id": "question_id"})

    return pivot


def build_model_summary(response_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build per-model accuracy summary from response matrix."""
    if response_matrix.empty:
        return pd.DataFrame()

    model_cols = [c for c in response_matrix.columns if c != "question_id"]

    summaries = []
    for model in model_cols:
        col = response_matrix[model]
        valid = col.dropna()
        n_questions = len(valid)
        n_correct = int(valid.sum())
        accuracy = n_correct / n_questions if n_questions > 0 else 0.0
        summaries.append({
            "model": model,
            "n_questions": n_questions,
            "n_correct": n_correct,
            "overall_accuracy": round(accuracy, 6),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("overall_accuracy", ascending=False)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Build response matrices from HELM multilingual data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing downloaded project data (afr/, thaiexam/, cleva/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "processed",
        help="Output directory for processed files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define projects to process
    projects = {
        "afr": {
            "dir": args.data_dir / "afr",
            "output_suffix": "afr",
            "description": "African MMLU + Winogrande (11 languages)",
        },
        "thaiexam": {
            "dir": args.data_dir / "thaiexam",
            "output_suffix": "thaiexam",
            "description": "Thai Examination Benchmarks",
        },
        "cleva": {
            "dir": args.data_dir / "cleva",
            "output_suffix": "cleva",
            "description": "Chinese Language Evaluation (CLEVA)",
        },
    }

    all_records = []
    all_metadata = {}

    for name, config in projects.items():
        print(f"\n{'='*60}")
        print(f"Processing: {config['description']}")
        print(f"Data dir: {config['dir']}")
        print(f"{'='*60}")

        records, metadata = process_project(config["dir"], name)
        print(f"  Extracted {len(records)} item-model records")
        print(f"  Unique items: {len(metadata)}")

        if records:
            # Build per-project response matrix
            rm = build_response_matrix(records)
            model_cols = [c for c in rm.columns if c != "question_id"]
            print(f"  Response matrix: {len(rm)} items x {len(model_cols)} models")

            # Save per-project response matrix
            rm_path = args.output_dir / f"response_matrix_{config['output_suffix']}.csv"
            rm.to_csv(rm_path, index=False)
            print(f"  Saved: {rm_path}")

            # Accumulate for combined matrix
            all_records.extend(records)
            all_metadata.update(metadata)

    # Build combined response matrix
    print(f"\n{'='*60}")
    print("Building combined response matrix")
    print(f"{'='*60}")

    if all_records:
        combined_rm = build_response_matrix(all_records)
        model_cols = [c for c in combined_rm.columns if c != "question_id"]
        print(f"Combined response matrix: {len(combined_rm)} items x {len(model_cols)} models")

        combined_path = args.output_dir / "response_matrix.csv"
        combined_rm.to_csv(combined_path, index=False)
        print(f"Saved: {combined_path}")

        # Build task metadata
        meta_df = pd.DataFrame(all_metadata.values())
        meta_df = meta_df.sort_values("item_id")
        meta_path = args.output_dir / "task_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"Saved: {meta_path} ({len(meta_df)} items)")

        # Build model summary
        model_summary = build_model_summary(combined_rm)
        summary_path = args.output_dir / "model_summary.csv"
        model_summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path} ({len(model_summary)} models)")

        # Print summary statistics
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        print(f"Total items: {len(combined_rm)}")
        print(f"Total models: {len(model_cols)}")
        print(f"Models: {sorted(model_cols)}")

        # Per-language breakdown
        lang_counts = meta_df.groupby("language").size()
        print(f"\nItems per language:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count}")

        # Per-benchmark breakdown
        bench_counts = meta_df.groupby("benchmark").size()
        print(f"\nItems per benchmark:")
        for bench, count in bench_counts.items():
            print(f"  {bench}: {count}")

        # Model accuracy summary
        print(f"\nModel accuracies (combined):")
        for _, row in model_summary.iterrows():
            print(f"  {row['model']}: {row['overall_accuracy']:.4f} "
                  f"({row['n_correct']}/{row['n_questions']})")
    else:
        print("No records found. Ensure data has been downloaded first.")


if __name__ == "__main__":
    main()
