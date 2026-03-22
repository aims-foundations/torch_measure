#!/usr/bin/env python3
"""Download per-instance HELM evaluation data for non-English benchmarks.

Downloads from the public gs://crfm-helm-public/ bucket via HTTPS.
No authentication required.

Usage:
    python download_helm_multilingual.py [--project {thaiexam,cleva,afr,all}] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

GCS_API = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
GCS_DL = "https://storage.googleapis.com/crfm-helm-public"

# Non-English HELM projects with per-instance data
PROJECTS = {
    "thaiexam": {
        "prefix": "thaiexam/benchmark_output/runs/",
        "description": "Thai examination benchmarks (A-Level, IC, ONET, TGAT, TPAT1)",
        "language": "Thai",
    },
    "cleva": {
        "prefix": "cleva/benchmark_output/runs/",
        "description": "Chinese Language Evaluation (21 tasks)",
        "language": "Chinese",
    },
    "afr": {
        "prefix": "mmlu-winogrande-afr/benchmark_output/runs/",
        "description": "African languages MMLU + Winogrande (11 languages)",
        "language": "Afrikaans, Amharic, Bambara, Igbo, Sepedi, Shona, Sesotho, Setswana, Tsonga, Xhosa, Zulu",
    },
}

# Per-instance data files to download from each run
INSTANCE_FILES = [
    "instances.json",
    "per_instance_stats.json",
    "run_spec.json",
    "scenario.json",
    "stats.json",
    "display_predictions.json",
]


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
    except Exception as e:
        # File might not exist for all runs
        return False


def download_run(run_prefix: str, output_dir: Path, files: list[str]) -> dict:
    """Download per-instance files for a single run."""
    # Extract run name for local directory structure
    parts = run_prefix.strip("/").split("/")
    # e.g., thaiexam/benchmark_output/runs/v1.2.0/thai_exam:exam=onet,...
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


def main():
    parser = argparse.ArgumentParser(description="Download multilingual HELM per-instance data")
    parser.add_argument(
        "--project",
        choices=["thaiexam", "cleva", "afr", "all"],
        default="all",
        help="Which project to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only download the latest version of each project",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs without downloading",
    )
    args = parser.parse_args()

    projects = PROJECTS if args.project == "all" else {args.project: PROJECTS[args.project]}

    for name, config in projects.items():
        print(f"\n{'='*60}")
        print(f"Project: {name}")
        print(f"Description: {config['description']}")
        print(f"Language(s): {config['language']}")
        print(f"{'='*60}")

        prefix = config["prefix"]
        versions = list_prefixes(prefix)
        print(f"Available versions: {[v.split('/')[-2] for v in versions]}")

        if args.latest_only and versions:
            versions = [versions[-1]]
            print(f"Downloading latest only: {versions[0].split('/')[-2]}")

        all_runs = []
        for v in versions:
            runs = list_prefixes(v)
            all_runs.extend(runs)
            print(f"  {v.split('/')[-2]}: {len(runs)} runs")

        print(f"Total runs: {len(all_runs)}")

        if args.dry_run:
            for run in all_runs[:10]:
                print(f"  {run.strip('/').split('/')[-1]}")
            if len(all_runs) > 10:
                print(f"  ... and {len(all_runs) - 10} more")
            continue

        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading to {output_dir}...")
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_run, run, output_dir, INSTANCE_FILES): run
                for run in all_runs
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                n_ok = len(result["downloaded"])
                n_fail = len(result["failed"])
                run_name = result["run"].strip("/").split("/")[-1]
                if completed % 10 == 0 or completed == len(all_runs):
                    print(f"  [{completed}/{len(all_runs)}] {run_name}: {n_ok} files")

        print(f"\nDone. Data saved to {output_dir}")


if __name__ == "__main__":
    main()
