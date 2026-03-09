#!/usr/bin/env python3
"""
Build the SWE-bench Full response matrix from the experiments repo.

SWE-bench Full = the original 2,294-instance test set (NOT the 500-instance
Verified subset). Results come from:
  https://github.com/SWE-bench/experiments/tree/main/evaluation/test/

Each results.json has the structure:
  {
    "no_generation": [...],
    "generated": [...],
    "with_logs": [...],
    "install_fail": [...],
    "reset_failed": [...],
    "no_apply": [...],
    "applied": [...],
    "test_errored": [...],
    "test_timeout": [...],
    "resolved": [...]
  }

The canonical 2,294 instance IDs come from HuggingFace:
  princeton-nlp/SWE-bench (split="test")

Outputs:
  - response_matrix.csv: Binary matrix (models x instance_ids), 1=resolved, 0=not
  - model_summary.csv: Per-model summary statistics
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "raw" / "results_json"
CANONICAL_IDS_PATH = PROJECT_DIR / "raw" / "canonical_instance_ids.json"
PROCESSED_DIR = PROJECT_DIR / "processed"

# GitHub base URL for downloading results
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/SWE-bench/experiments"
    "/main/evaluation/test"
)
GITHUB_API_URL = (
    "https://api.github.com/repos/SWE-bench/experiments"
    "/contents/evaluation/test"
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def fetch_submission_list() -> list[str]:
    """Get the list of submission directories from the GitHub API."""
    print("Fetching submission list from GitHub API...")
    req = urllib.request.Request(
        GITHUB_API_URL, headers={"User-Agent": "swebench-full-matrix-builder"}
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    dirs = sorted(item["name"] for item in data if item["type"] == "dir")
    print(f"  Found {len(dirs)} submissions on GitHub")
    return dirs


def download_results_json(submissions: list[str]) -> None:
    """Download results.json for each submission if not already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded, skipped, failed = 0, 0, 0

    for sub in submissions:
        out_path = RAW_DIR / f"{sub}.json"
        if out_path.exists():
            skipped += 1
            continue
        url = f"{GITHUB_RAW_BASE}/{sub}/results/results.json"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "swebench-full-matrix-builder"}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            raw = resp.read()
            # Validate JSON
            json.loads(raw)
            out_path.write_bytes(raw)
            downloaded += 1
        except Exception as exc:
            print(f"  WARNING: Failed to download {sub}: {exc}", file=sys.stderr)
            failed += 1

    print(
        f"  Download summary: {downloaded} new, {skipped} cached, {failed} failed"
    )


def fetch_canonical_instance_ids() -> list[str]:
    """Load canonical 2,294 instance IDs from HuggingFace (cached locally)."""
    if CANONICAL_IDS_PATH.exists():
        with open(CANONICAL_IDS_PATH) as f:
            ids = json.load(f)
        print(f"  Loaded {len(ids)} canonical instance IDs from cache")
        return ids

    print("  Downloading canonical instance IDs from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench", split="test")
        ids = sorted(row["instance_id"] for row in ds)
    except ImportError:
        print(
            "  WARNING: 'datasets' library not available. "
            "Falling back to union of all results.json IDs.",
            file=sys.stderr,
        )
        return []

    CANONICAL_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANONICAL_IDS_PATH, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"  Saved {len(ids)} canonical instance IDs")
    return ids


# ---------------------------------------------------------------------------
# Core matrix building
# ---------------------------------------------------------------------------
def load_results(results_dir: Path) -> tuple[dict, set]:
    """Load all results.json files.

    Returns:
        model_results: {model_name: set_of_resolved_ids}
        all_instance_ids: set of all instance IDs seen across all files
    """
    model_results = {}
    all_instance_ids = set()

    for fpath in sorted(results_dir.glob("*.json")):
        model_name = fpath.stem
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            print(f"  WARNING: Skipping {fpath.name}: {exc}", file=sys.stderr)
            continue

        if "resolved" not in data:
            print(
                f"  WARNING: Skipping {fpath.name}: no 'resolved' key",
                file=sys.stderr,
            )
            continue

        resolved = set(data["resolved"])
        model_results[model_name] = resolved

        # Collect all instance IDs from all categories
        for key in data:
            if isinstance(data[key], list):
                all_instance_ids.update(data[key])

    return model_results, all_instance_ids


def build_response_matrix(
    model_results: dict, instance_ids: list
) -> pd.DataFrame:
    """Build binary response matrix: models (rows) x instances (columns)."""
    rows = {}
    for model_name, resolved_set in sorted(model_results.items()):
        rows[model_name] = [
            1 if iid in resolved_set else 0 for iid in instance_ids
        ]

    df = pd.DataFrame.from_dict(rows, orient="index", columns=instance_ids)
    df.index.name = "model"
    return df


def build_model_summary(response_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build per-model summary statistics."""
    n_instances = response_matrix.shape[1]
    summary = pd.DataFrame(
        {
            "model": response_matrix.index,
            "resolved_count": response_matrix.sum(axis=1).values,
            "total_instances": n_instances,
            "resolve_rate": (
                response_matrix.sum(axis=1).values / n_instances * 100
            ).round(2),
        }
    )
    summary = summary.sort_values(
        "resolved_count", ascending=False
    ).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Statistics printing
# ---------------------------------------------------------------------------
def print_statistics(
    response_matrix: pd.DataFrame,
    model_results: dict,
    summary: pd.DataFrame,
) -> None:
    """Print comprehensive summary statistics."""
    n_models = response_matrix.shape[0]
    n_instances = response_matrix.shape[1]
    matrix = response_matrix.values

    print("\n" + "=" * 80)
    print("SWE-BENCH FULL RESPONSE MATRIX STATISTICS")
    print("=" * 80)

    # Overall matrix stats
    total_cells = n_models * n_instances
    total_resolved = int(matrix.sum())
    print(f"\n  Matrix shape: {n_models} models x {n_instances} instances")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total resolved cells: {total_resolved:,}")
    print(f"  Overall density (resolve rate): {matrix.mean():.4f}")

    # Per-model stats
    per_model = matrix.sum(axis=1)
    per_model_rate = per_model / n_instances
    print(f"\n  Per-model resolve counts:")
    print(f"    Min:    {int(per_model.min()):>5d}  "
          f"({per_model_rate.min()*100:.1f}%)")
    print(f"    Max:    {int(per_model.max()):>5d}  "
          f"({per_model_rate.max()*100:.1f}%)")
    print(f"    Mean:   {per_model.mean():>8.1f}  "
          f"({per_model_rate.mean()*100:.1f}%)")
    print(f"    Median: {np.median(per_model):>8.1f}  "
          f"({np.median(per_model_rate)*100:.1f}%)")
    print(f"    Std:    {per_model.std():>8.1f}  "
          f"({per_model_rate.std()*100:.1f}%)")

    # Per-instance stats
    per_instance = matrix.sum(axis=0)
    per_instance_rate = per_instance / n_models
    print(f"\n  Per-instance statistics:")
    print(f"    Resolved by ALL {n_models} models: "
          f"{(per_instance == n_models).sum()}")
    print(f"    Resolved by NO model:  {(per_instance == 0).sum()}")
    print(f"    Mean models resolving:  {per_instance.mean():.2f} "
          f"(of {n_models})")
    print(f"    Median models resolving: {np.median(per_instance):.1f}")
    print(f"    Std:    {per_instance.std():.2f}")

    # Difficulty distribution
    print(f"\n  Instance difficulty distribution "
          f"(by fraction of {n_models} models resolving):")
    thresholds = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == len(thresholds) - 2:
            count = ((per_instance_rate >= lo) & (per_instance_rate <= hi)).sum()
            label = f"[{lo:.0%}, {hi:.0%}]"
        else:
            count = ((per_instance_rate >= lo) & (per_instance_rate < hi)).sum()
            label = f"[{lo:.0%}, {hi:.0%})"
        print(f"    {label:>12s}: {count:>5d} instances")

    # Top models
    print("\n" + "=" * 80)
    print("TOP MODELS (by resolve rate):")
    print("=" * 80)
    print(summary.head(24).to_string(index=False))

    # Bottom models
    print("\n" + "=" * 80)
    print("BOTTOM 10 MODELS (by resolve rate):")
    print("=" * 80)
    print(summary.tail(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("SWE-bench Full Response Matrix Builder")
    print("  Benchmark: SWE-bench Full (2,294 instances)")
    print("  Source: github.com/SWE-bench/experiments/evaluation/test/")
    print("=" * 80)

    # Step 1: Discover and download results
    print("\n[Step 1] Discovering submissions...")
    try:
        submissions = fetch_submission_list()
    except Exception as exc:
        print(f"  Could not fetch from GitHub API: {exc}")
        print("  Falling back to local files...")
        submissions = []

    if submissions:
        print("\n[Step 2] Downloading results.json files...")
        download_results_json(submissions)
    else:
        print("\n[Step 2] Skipping download (using local files only)")

    # Step 3: Get canonical instance IDs
    print("\n[Step 3] Loading canonical instance IDs...")
    canonical_ids = fetch_canonical_instance_ids()

    # Step 4: Load results
    print(f"\n[Step 4] Loading results from: {RAW_DIR}")
    model_results, observed_ids = load_results(RAW_DIR)
    print(f"  Found {len(model_results)} models with results")
    print(f"  Found {len(observed_ids)} unique instance IDs in result files")

    # Use canonical IDs if available, else fall back to observed
    if canonical_ids:
        instance_ids = canonical_ids
        # Check for any IDs in results not in canonical set
        extra_ids = observed_ids - set(canonical_ids)
        if extra_ids:
            print(
                f"  WARNING: {len(extra_ids)} instance IDs in results "
                f"but not in canonical set (will be excluded)"
            )
        missing_ids = set(canonical_ids) - observed_ids
        if missing_ids:
            print(
                f"  NOTE: {len(missing_ids)} canonical IDs never appear "
                f"in any results file (likely never attempted)"
            )
    else:
        instance_ids = sorted(observed_ids)

    print(f"  Using {len(instance_ids)} instance IDs for matrix columns")

    # Step 5: Build matrix
    print("\n[Step 5] Building response matrix...")
    response_matrix = build_response_matrix(model_results, instance_ids)
    print(f"  Matrix shape: {response_matrix.shape}")

    # Step 6: Build summary
    print("\n[Step 6] Building model summary...")
    summary = build_model_summary(response_matrix)

    # Step 7: Save outputs
    print("\n[Step 7] Saving outputs...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matrix_path = PROCESSED_DIR / "response_matrix.csv"
    response_matrix.to_csv(matrix_path)
    matrix_size_mb = matrix_path.stat().st_size / (1024 * 1024)
    print(f"  Saved response matrix: {matrix_path} ({matrix_size_mb:.2f} MB)")

    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved model summary: {summary_path}")

    # Step 8: Print statistics
    print_statistics(response_matrix, model_results, summary)

    # Final file listing
    print("\n" + "=" * 80)
    print("OUTPUT FILES:")
    print("=" * 80)
    for f in sorted(PROCESSED_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s}  {size_kb:>8.1f} KB")

    print("\nDone.")


if __name__ == "__main__":
    main()
