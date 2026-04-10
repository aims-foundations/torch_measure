#!/usr/bin/env python3
"""Download and build response matrix for HELM Thai examination benchmarks.

Downloads per-instance data from the public gs://crfm-helm-public/ bucket via HTTPS,
parses per_instance_stats.json files, extracts item_id, model, and exact_match metric,
and pivots into a response matrix following torch_measure conventions.

Language: Thai (A-Level, IC, ONET, TGAT, TPAT1)

Outputs (to processed/):
  - response_matrix.csv   : models (rows) x items (cols), binary 0/1
  - item_content.csv      : item_id, content
  - task_metadata.csv     : item_id, language, benchmark, subject
  - model_summary.csv     : per-model accuracy

Usage:
    python 01_build_response_matrix.py [--skip-download] [--workers 8]
"""

import sys
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# GCS constants
# ---------------------------------------------------------------------------
GCS_API = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
GCS_DL = "https://storage.googleapis.com/crfm-helm-public"

PROJECT_PREFIX = "thaiexam/benchmark_output/runs/"

INSTANCE_FILES = [
    "instances.json",
    "per_instance_stats.json",
    "run_spec.json",
    "scenario.json",
    "stats.json",
    "display_predictions.json",
]

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
RAW_DIR = DATA_DIR / "raw" / "thaiexam"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def list_prefixes(prefix: str, delimiter: str = "/") -> list[str]:
    """List sub-prefixes (directories) under a GCS prefix."""
    params = urllib.parse.urlencode({
        "prefix": prefix,
        "delimiter": delimiter,
        "maxResults": 1000,
    })
    url = f"{GCS_API}?{params}"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    return data.get("prefixes", [])


def list_all_runs(project_prefix: str) -> list[str]:
    """List all run directories for a project (across all versions)."""
    versions = list_prefixes(project_prefix)
    all_runs = []
    for version_prefix in versions:
        runs = list_prefixes(version_prefix)
        all_runs.extend(runs)
    return all_runs


def download_file(url: str, dest: Path) -> bool:
    """Download a single file from GCS."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception:
        return False


def download_run(run_prefix: str, output_dir: Path, files: list[str]) -> dict:
    """Download per-instance files for a single run."""
    parts = run_prefix.strip("/").split("/")
    local_subdir = "/".join(parts[3:])  # version/run_name
    run_dir = output_dir / local_subdir

    results = {"run": run_prefix, "downloaded": [], "failed": []}
    for filename in files:
        url = f"{GCS_DL}/{run_prefix}{filename}"
        dest = run_dir / filename
        if dest.exists():
            results["downloaded"].append(filename)
            continue
        if download_file(url, dest):
            results["downloaded"].append(filename)
        else:
            results["failed"].append(filename)
    return results


def download(workers: int = 8, latest_only: bool = False) -> None:
    """Download all Thai exam data from GCS."""
    print(f"\n{'=' * 60}")
    print("Downloading: Thai examination benchmarks (A-Level, IC, ONET, TGAT, TPAT1)")
    print(f"{'=' * 60}")

    versions = list_prefixes(PROJECT_PREFIX)
    print(f"Available versions: {[v.split('/')[-2] for v in versions]}")

    if latest_only and versions:
        versions = [versions[-1]]
        print(f"Downloading latest only: {versions[0].split('/')[-2]}")

    all_runs = []
    for v in versions:
        runs = list_prefixes(v)
        all_runs.extend(runs)
        print(f"  {v.split('/')[-2]}: {len(runs)} runs")

    print(f"Total runs: {len(all_runs)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading to {RAW_DIR}...")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_run, run, RAW_DIR, INSTANCE_FILES): run
            for run in all_runs
        }
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            n_ok = len(result["downloaded"])
            run_name = result["run"].strip("/").split("/")[-1]
            if completed % 10 == 0 or completed == len(all_runs):
                print(f"  [{completed}/{len(all_runs)}] {run_name}: {n_ok} files")

    print(f"\nDone. Data saved to {RAW_DIR}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_run_dir_name(run_dir_name: str) -> dict:
    """Parse run directory name to extract task, model, subject.

    Examples:
      thai_exam:exam=onet,method=multiple_choice_joint,model=openai_gpt-4o-2024-05-13
    """
    result = {}

    if ":" not in run_dir_name:
        return {"raw_name": run_dir_name}

    benchmark_prefix, params_str = run_dir_name.split(":", 1)
    result["benchmark_prefix"] = benchmark_prefix

    params = {}
    parts = params_str.split(",")
    for part in parts:
        if "=" in part:
            key, val = part.split("=", 1)
            params[key] = val

    result["model"] = params.get("model", "unknown")
    result["method"] = params.get("method", "")

    result["benchmark"] = "thai_exam"
    result["subject"] = params.get("exam", "")
    result["language"] = "th"

    return result


def extract_exact_match(stats_list: list[dict]) -> float | None:
    """Extract the exact_match metric from a stats list.

    Prefers exact_match, falls back to quasi_exact_match.
    """
    for stat in stats_list:
        name_info = stat.get("name", {})
        if name_info.get("name") == "exact_match":
            return stat.get("mean")

    for stat in stats_list:
        name_info = stat.get("name", {})
        if name_info.get("name") == "quasi_exact_match":
            return stat.get("mean")

    return None


def process_project(project_dir: Path, project_name: str) -> tuple[list[dict], dict]:
    """Process all runs for the project.

    Returns:
        records: list of {item_id, model, score, benchmark, subject, language}
        item_metadata: dict of item_id -> {language, benchmark, subject}
    """
    records = []
    item_metadata = {}

    if not project_dir.exists():
        print(f"  WARNING: {project_dir} does not exist, skipping")
        return records, item_metadata

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

                if subject:
                    global_item_id = f"{benchmark}_{subject}_{language}_{instance_id}"
                else:
                    global_item_id = f"{benchmark}_{language}_{instance_id}"

                records.append({
                    "item_id": global_item_id,
                    "model": model,
                    "score": int(round(score)),
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
    """Pivot records into a response matrix: models (rows) x items (cols)."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # If there are duplicates (same item + model), take the max score
    pivot = df.pivot_table(
        index="model",
        columns="item_id",
        values="score",
        aggfunc="max",
    )

    # Sort columns (items) alphabetically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Sort rows (models) alphabetically
    pivot = pivot.sort_index()

    # Set index name
    pivot.index.name = "model"

    return pivot


def build_model_summary(response_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build per-model accuracy summary from response matrix."""
    if response_matrix.empty:
        return pd.DataFrame()

    summaries = []
    for model in response_matrix.index:
        row = response_matrix.loc[model]
        valid = row.dropna()
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download and build response matrix for HELM Thai exams")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument("--latest-only", action="store_true", help="Only download the latest version")
    args = parser.parse_args()

    # Step 1: Download
    if not args.skip_download:
        download(workers=args.workers, latest_only=args.latest_only)

    # Step 2: Process
    print(f"\n{'=' * 60}")
    print("Processing: Thai examination benchmarks")
    print(f"Data dir: {RAW_DIR}")
    print(f"{'=' * 60}")

    records, metadata = process_project(RAW_DIR, "thaiexam")
    print(f"  Extracted {len(records)} item-model records")
    print(f"  Unique items: {len(metadata)}")

    if not records:
        print("No records found. Ensure data has been downloaded first.")
        return

    # Step 3: Build response matrix
    rm = build_response_matrix(records)
    print(f"  Response matrix: {len(rm)} models x {len(rm.columns)} items")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save response matrix
    rm_path = PROCESSED_DIR / "response_matrix.csv"
    rm.to_csv(rm_path)
    print(f"  Saved: {rm_path}")

    # Save item content
    item_content = pd.DataFrame([
        {"item_id": item_id, "content": item_id}
        for item_id in sorted(metadata.keys())
    ])
    item_content_path = PROCESSED_DIR / "item_content.csv"
    item_content.to_csv(item_content_path, index=False)
    print(f"  Saved: {item_content_path} ({len(item_content)} items)")

    # Save task metadata
    meta_df = pd.DataFrame(metadata.values())
    meta_df = meta_df.sort_values("item_id")
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path} ({len(meta_df)} items)")

    # Save model summary
    model_summary = build_model_summary(rm)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    model_summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path} ({len(model_summary)} models)")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary Statistics")
    print(f"{'=' * 60}")
    print(f"Total items: {len(rm.columns)}")
    print(f"Total models: {len(rm)}")
    print(f"Models: {sorted(rm.index.tolist())}")

    lang_counts = meta_df.groupby("language").size()
    print(f"\nItems per language:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")

    bench_counts = meta_df.groupby("benchmark").size()
    print(f"\nItems per benchmark:")
    for bench, count in bench_counts.items():
        print(f"  {bench}: {count}")

    print(f"\nModel accuracies:")
    for _, row in model_summary.iterrows():
        print(f"  {row['model']}: {row['overall_accuracy']:.4f} "
              f"({row['n_correct']}/{row['n_questions']})")


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
