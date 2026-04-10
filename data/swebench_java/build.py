#!/usr/bin/env python3
"""
Build the SWE-bench Java response matrix from multiple data sources.

SWE-bench Java evaluates LLM agents on their ability to resolve real GitHub
issues in Java repositories. Paper: "SWE-bench-java: A GitHub Issue Resolving
Benchmark for Java" (arXiv:2408.14354).

Data sources:
  1. Multi-SWE-bench (ByteDance) — Java verified split (128 instances):
     - https://github.com/multi-swe-bench/experiments/tree/main/evaluation/java/verified
     - results.json per submission with resolved/unresolved instance ID lists
     - This is the primary data source from the SWE-bench-java paper authors

  2. SWE-bench Multilingual (official, from SWE-bench/experiments) — Java subset:
     - https://github.com/SWE-bench/experiments/tree/main/evaluation/multilingual
     - per_instance_details.json with {resolved: true/false} per instance
     - 43 Java instances out of 300 total multilingual instances
     - Different set of Java repos (druid, lucene, gson, javaparser, lombok, rxjava)

  3. HuggingFace dataset: Daoguang/Multi-SWE-bench (split="java_verified")
     - 91 canonical instance IDs (original paper size, subset of the 128)

Outputs:
  - response_matrix.csv: Combined binary matrix (models x instance_ids), 1=resolved, 0=not
  - response_matrix_multi_swebench_java.csv: Matrix from Multi-SWE-bench Java only
  - response_matrix_swebench_multilingual_java.csv: Matrix from SWE-bench Multilingual Java
  - model_summary.csv: Per-model summary statistics
"""

import json
import os
import re
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
RAW_DIR = PROJECT_DIR / "raw"
PROCESSED_DIR = PROJECT_DIR / "processed"

MULTI_SWE_DIR = RAW_DIR / "multi_swe_bench_java"
SWEBENCH_ML_DIR = RAW_DIR / "swebench_multilingual_java"
HF_CANONICAL_IDS_PATH = RAW_DIR / "hf_canonical_instance_ids.json"

# GitHub URLs for Multi-SWE-bench Java verified
MULTI_SWE_API_URL = (
    "https://api.github.com/repos/multi-swe-bench/experiments"
    "/contents/evaluation/java/verified"
)
MULTI_SWE_RAW_BASE = (
    "https://raw.githubusercontent.com/multi-swe-bench/experiments"
    "/main/evaluation/java/verified"
)

# GitHub URLs for SWE-bench Multilingual
SWEBENCH_ML_API_URL = (
    "https://api.github.com/repos/SWE-bench/experiments"
    "/contents/evaluation/multilingual"
)
SWEBENCH_ML_RAW_BASE = (
    "https://raw.githubusercontent.com/SWE-bench/experiments"
    "/main/evaluation/multilingual"
)

# Java repo mappings for SWE-bench Multilingual (to filter Java instances)
SWEBENCH_ML_JAVA_REPOS = {
    "apache__druid",
    "apache__lucene",
    "google__gson",
    "javaparser__javaparser",
    "projectlombok__lombok",
    "reactivex__rxjava",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_multi_swe_bench_results():
    """Download results.json for all Multi-SWE-bench Java verified submissions."""
    print("\n  Downloading Multi-SWE-bench Java verified results...")
    MULTI_SWE_DIR.mkdir(parents=True, exist_ok=True)

    # Also download index.json
    index_path = MULTI_SWE_DIR / "index.json"
    if not index_path.exists():
        url = f"{MULTI_SWE_RAW_BASE}/index.json"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "swebench-java-matrix-builder"}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            index_path.write_bytes(resp.read())
            print("    Downloaded index.json")
        except Exception as exc:
            print(f"    WARNING: Failed to download index.json: {exc}")

    # Get submission directories
    try:
        req = urllib.request.Request(
            MULTI_SWE_API_URL,
            headers={"User-Agent": "swebench-java-matrix-builder"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        items = json.loads(resp.read())
        dirs = [item["name"] for item in items if item["type"] == "dir"]
        print(f"    Found {len(dirs)} submissions on GitHub")
    except Exception as exc:
        print(f"    Could not fetch from GitHub API: {exc}")
        print("    Using local files only")
        return

    downloaded, skipped, failed = 0, 0, 0
    for sub_dir in sorted(dirs):
        out_path = MULTI_SWE_DIR / f"{sub_dir}.json"
        if out_path.exists():
            skipped += 1
            continue
        url = f"{MULTI_SWE_RAW_BASE}/{sub_dir}/results/results.json"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "swebench-java-matrix-builder"}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            raw = resp.read()
            json.loads(raw)  # validate
            out_path.write_bytes(raw)
            downloaded += 1
        except Exception as exc:
            print(f"    WARNING: Failed {sub_dir}: {exc}", file=sys.stderr)
            failed += 1

    print(f"    Download: {downloaded} new, {skipped} cached, {failed} failed")


def download_swebench_multilingual_results():
    """Download per_instance_details.json for SWE-bench Multilingual submissions."""
    print("\n  Downloading SWE-bench Multilingual results...")
    SWEBENCH_ML_DIR.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            SWEBENCH_ML_API_URL,
            headers={"User-Agent": "swebench-java-matrix-builder"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        items = json.loads(resp.read())
        dirs = [item["name"] for item in items if item["type"] == "dir"]
        print(f"    Found {len(dirs)} submissions on GitHub")
    except Exception as exc:
        print(f"    Could not fetch from GitHub API: {exc}")
        print("    Using local files only")
        return

    downloaded, skipped, failed = 0, 0, 0
    for sub_dir in sorted(dirs):
        out_path = SWEBENCH_ML_DIR / f"{sub_dir}.json"
        if out_path.exists():
            skipped += 1
            continue
        url = f"{SWEBENCH_ML_RAW_BASE}/{sub_dir}/per_instance_details.json"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "swebench-java-matrix-builder"}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            raw = resp.read()
            json.loads(raw)
            out_path.write_bytes(raw)
            downloaded += 1
        except Exception as exc:
            print(f"    WARNING: Failed {sub_dir}: {exc}", file=sys.stderr)
            failed += 1

    print(f"    Download: {downloaded} new, {skipped} cached, {failed} failed")


def fetch_hf_canonical_ids():
    """Load canonical Java instance IDs from HuggingFace (cached locally)."""
    if HF_CANONICAL_IDS_PATH.exists():
        with open(HF_CANONICAL_IDS_PATH) as f:
            ids = json.load(f)
        print(f"    Loaded {len(ids)} HF canonical instance IDs from cache")
        return ids

    print("    Downloading canonical instance IDs from HuggingFace...")
    try:
        os.environ.setdefault(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        from datasets import load_dataset

        ds = load_dataset("Daoguang/Multi-SWE-bench", split="java_verified")
        ids = sorted(row["instance_id"] for row in ds)
    except ImportError:
        print("    WARNING: 'datasets' library not available", file=sys.stderr)
        return []

    HF_CANONICAL_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HF_CANONICAL_IDS_PATH, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"    Saved {len(ids)} canonical instance IDs")
    return ids


# ---------------------------------------------------------------------------
# Model name parsing
# ---------------------------------------------------------------------------
def parse_multi_swe_model_name(filename: str) -> str:
    """Parse model name from Multi-SWE-bench submission filename.

    Format: 20250329_MSWE-agent_Claude-3.5-Sonnet(Oct).json
    Returns: MSWE-agent_Claude-3.5-Sonnet(Oct)
    """
    stem = filename.replace(".json", "")
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return stem
    return parts[1]


def parse_swebench_ml_model_name(filename: str) -> str:
    """Parse model name from SWE-bench Multilingual submission filename.

    Format: 20260213_mini-v2.0.0a0_claude-4-5-sonnet.json
    Returns: claude-4-5-sonnet
    """
    stem = filename.replace(".json", "")
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return stem
    rest = parts[1]
    # Remove mini-vX.Y.Z version prefix
    rest = re.sub(r"^mini-v[\d.]+[a-z0-9]*_", "", rest)
    return "SWEbenchML_" + rest


def _get_repo_prefix(instance_id: str) -> str:
    """Extract repo prefix from instance_id (e.g., 'apache__dubbo' from 'apache__dubbo-10638')."""
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0]
    return instance_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_multi_swe_bench_results():
    """Load all Multi-SWE-bench Java verified results.

    Returns:
        model_results: {model_name: set_of_resolved_ids}
        all_instance_ids: set of all instance IDs seen
        canonical_ids: list of canonical IDs from index.json
    """
    model_results = {}
    all_instance_ids = set()

    # Load canonical IDs from index.json
    canonical_ids = []
    index_path = MULTI_SWE_DIR / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
        canonical_ids = index_data.get("all_ids", [])

    for fpath in sorted(MULTI_SWE_DIR.glob("*.json")):
        if fpath.name == "index.json":
            continue

        model_name = parse_multi_swe_model_name(fpath.name)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            print(f"    WARNING: Skipping {fpath.name}: {exc}", file=sys.stderr)
            continue

        if "resolved" not in data:
            print(
                f"    WARNING: Skipping {fpath.name}: no 'resolved' key",
                file=sys.stderr,
            )
            continue

        resolved = set(data["resolved"])
        model_results[model_name] = resolved

        # Collect all instance IDs from submitted_ids
        submitted = data.get("submitted_ids", [])
        all_instance_ids.update(submitted)
        all_instance_ids.update(resolved)

    return model_results, all_instance_ids, canonical_ids


def load_swebench_multilingual_java_results():
    """Load Java-only results from SWE-bench Multilingual per_instance_details.

    Returns:
        model_results: {model_name: set_of_resolved_java_ids}
        all_java_ids: set of all Java instance IDs seen
    """
    model_results = {}
    all_java_ids = set()

    for fpath in sorted(SWEBENCH_ML_DIR.glob("*.json")):
        model_name = parse_swebench_ml_model_name(fpath.name)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            print(f"    WARNING: Skipping {fpath.name}: {exc}", file=sys.stderr)
            continue

        resolved = set()
        java_ids_this_model = set()
        for instance_id, info in data.items():
            repo_prefix = _get_repo_prefix(instance_id)
            if repo_prefix not in SWEBENCH_ML_JAVA_REPOS:
                continue
            java_ids_this_model.add(instance_id)
            all_java_ids.add(instance_id)
            if info.get("resolved", False):
                resolved.add(instance_id)

        if java_ids_this_model:
            model_results[model_name] = resolved

    return model_results, all_java_ids


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------
def build_response_matrix(
    model_results: dict,
    instance_ids: list,
    submitted_universe: dict = None,
) -> pd.DataFrame:
    """Build binary response matrix: models (rows) x instances (columns).

    Args:
        model_results: {model_name: set_of_resolved_ids}
        instance_ids: list of all instance IDs (columns)
        submitted_universe: optional {model_name: set_of_submitted_ids}
            If provided, cells where the instance was not submitted are NaN.
            If None, all cells are 0 or 1.
    """
    rows = {}
    for model_name in sorted(model_results.keys()):
        resolved_set = model_results[model_name]
        row = []
        for iid in instance_ids:
            if submitted_universe and model_name in submitted_universe:
                if iid not in submitted_universe[model_name]:
                    row.append(np.nan)
                else:
                    row.append(1 if iid in resolved_set else 0)
            else:
                row.append(1 if iid in resolved_set else 0)
        rows[model_name] = row

    df = pd.DataFrame.from_dict(rows, orient="index", columns=instance_ids)
    df.index.name = "model"
    return df


def build_model_summary(response_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build per-model summary statistics."""
    n_instances = response_matrix.shape[1]
    resolved_counts = response_matrix.sum(axis=1).astype(int)
    total_attempted = response_matrix.notna().sum(axis=1).astype(int)

    summary = pd.DataFrame(
        {
            "model": response_matrix.index,
            "resolved_count": resolved_counts.values,
            "total_attempted": total_attempted.values,
            "total_instances": n_instances,
            "resolve_rate": (resolved_counts.values / n_instances * 100).round(2),
        }
    )
    summary = summary.sort_values(
        "resolved_count", ascending=False
    ).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Statistics printing
# ---------------------------------------------------------------------------
def print_matrix_statistics(
    df: pd.DataFrame,
    label: str,
) -> None:
    """Print comprehensive statistics for a response matrix."""
    n_models = df.shape[0]
    n_instances = df.shape[1]
    total_cells = n_models * n_instances
    n_filled = int(df.notna().sum().sum())
    fill_rate = n_filled / total_cells if total_cells > 0 else 0
    matrix_vals = df.values

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Matrix dimensions: {n_models} models x {n_instances} instances")
    print(f"  Total cells:       {total_cells:,}")
    print(f"  Filled cells:      {n_filled:,} ({fill_rate*100:.1f}%)")

    if n_filled > 0:
        vals = matrix_vals[~np.isnan(matrix_vals)]
        n_resolved = int(vals.sum())
        n_unresolved = n_filled - n_resolved
        print(f"  Resolved (1):      {n_resolved:,} ({n_resolved/n_filled*100:.1f}%)")
        print(f"  Unresolved (0):    {n_unresolved:,} ({n_unresolved/n_filled*100:.1f}%)")

        # Per-model stats
        per_model_resolved = df.sum(axis=1)
        per_model_rate = per_model_resolved / n_instances
        print(f"\n  Per-model resolve counts:")
        print(f"    Min:    {int(per_model_resolved.min()):>5d}  "
              f"({per_model_rate.min()*100:.1f}%)")
        print(f"    Max:    {int(per_model_resolved.max()):>5d}  "
              f"({per_model_rate.max()*100:.1f}%)")
        print(f"    Mean:   {per_model_resolved.mean():>8.1f}  "
              f"({per_model_rate.mean()*100:.1f}%)")
        print(f"    Median: {np.median(per_model_resolved):>8.1f}  "
              f"({np.median(per_model_rate)*100:.1f}%)")
        print(f"    Std:    {per_model_resolved.std():>8.1f}  "
              f"({per_model_rate.std()*100:.1f}%)")

        # Per-instance stats
        per_instance = df.sum(axis=0)
        per_instance_rate = per_instance / n_models
        print(f"\n  Per-instance statistics:")
        print(f"    Resolved by ALL {n_models} models: "
              f"{int((per_instance == n_models).sum())}")
        print(f"    Resolved by NO model:  {int((per_instance == 0).sum())}")
        print(f"    Mean models resolving:  {per_instance.mean():.2f} "
              f"(of {n_models})")
        print(f"    Median: {np.median(per_instance):.1f}")

        # Difficulty distribution
        print(f"\n  Instance difficulty (fraction of {n_models} models resolving):")
        thresholds = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        for i in range(len(thresholds) - 1):
            lo, hi = thresholds[i], thresholds[i + 1]
            if i == len(thresholds) - 2:
                count = int(
                    ((per_instance_rate >= lo) & (per_instance_rate <= hi)).sum()
                )
                label_range = f"[{lo:.0%}, {hi:.0%}]"
            else:
                count = int(
                    ((per_instance_rate >= lo) & (per_instance_rate < hi)).sum()
                )
                label_range = f"[{lo:.0%}, {hi:.0%})"
            print(f"    {label_range:>12s}: {count:>5d} instances")

        # Repository breakdown
        repos = {}
        for iid in df.columns:
            repo = _get_repo_prefix(iid)
            repos.setdefault(repo, []).append(iid)
        if repos:
            print(f"\n  Repository breakdown:")
            for repo in sorted(repos.keys()):
                ids = repos[repo]
                avg_resolve = df[ids].mean().mean() * 100
                print(f"    {repo:40s}: {len(ids):>4d} instances, "
                      f"avg resolve {avg_resolve:.1f}%")


def print_leaderboard(summary: pd.DataFrame, label: str) -> None:
    """Print a leaderboard table."""
    print(f"\n{'=' * 70}")
    print(f"  LEADERBOARD: {label}")
    print(f"{'=' * 70}")
    print(f"  {'Rank':>4s}  {'Model':<50s}  {'Resolved':>8s}  {'Rate':>6s}")
    print(f"  {'----':>4s}  {'-'*50}  {'--------':>8s}  {'------':>6s}")
    for i, (_, row) in enumerate(summary.iterrows()):
        print(
            f"  {i+1:4d}  {row['model']:<50s}  "
            f"{row['resolved_count']:>8d}  {row['resolve_rate']:>5.1f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  SWE-bench Java Response Matrix Builder")
    print("  Benchmark: SWE-bench Java (Java GitHub issue resolution)")
    print("  Paper: arXiv:2408.14354")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Download data
    # ------------------------------------------------------------------
    print("\n[Step 1] Downloading data...")
    download_multi_swe_bench_results()
    download_swebench_multilingual_results()

    # ------------------------------------------------------------------
    # Step 2: Load HuggingFace canonical IDs
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading canonical instance IDs...")
    hf_ids = fetch_hf_canonical_ids()

    # ------------------------------------------------------------------
    # Step 3: Load Multi-SWE-bench Java verified results
    # ------------------------------------------------------------------
    print("\n[Step 3] Loading Multi-SWE-bench Java verified results...")
    mswe_results, mswe_all_ids, mswe_canonical_ids = (
        load_multi_swe_bench_results()
    )
    print(f"    Found {len(mswe_results)} models")
    print(f"    Found {len(mswe_all_ids)} unique instance IDs in results")
    print(f"    Index.json canonical IDs: {len(mswe_canonical_ids)}")

    # Use index.json canonical IDs as the ground truth for Multi-SWE-bench
    if mswe_canonical_ids:
        mswe_instance_ids = sorted(mswe_canonical_ids)
    else:
        mswe_instance_ids = sorted(mswe_all_ids)
    print(f"    Using {len(mswe_instance_ids)} instance IDs for matrix")

    # Build submitted universe for NaN handling
    mswe_submitted = {}
    for fpath in sorted(MULTI_SWE_DIR.glob("*.json")):
        if fpath.name == "index.json":
            continue
        model_name = parse_multi_swe_model_name(fpath.name)
        try:
            with open(fpath) as f:
                data = json.load(f)
            mswe_submitted[model_name] = set(data.get("submitted_ids", []))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 4: Build Multi-SWE-bench Java matrix
    # ------------------------------------------------------------------
    print("\n[Step 4] Building Multi-SWE-bench Java response matrix...")
    df_mswe = build_response_matrix(
        mswe_results, mswe_instance_ids, mswe_submitted
    )

    mswe_summary = build_model_summary(df_mswe)
    print_matrix_statistics(df_mswe, "Multi-SWE-bench Java Verified (128 instances)")
    print_leaderboard(mswe_summary, "Multi-SWE-bench Java Verified")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mswe_path = PROCESSED_DIR / "response_matrix_multi_swebench_java.csv"
    df_mswe.to_csv(mswe_path)
    print(f"\n  Saved: {mswe_path}")

    # ------------------------------------------------------------------
    # Step 5: Load SWE-bench Multilingual Java results
    # ------------------------------------------------------------------
    print("\n[Step 5] Loading SWE-bench Multilingual Java results...")
    ml_results, ml_java_ids = load_swebench_multilingual_java_results()
    print(f"    Found {len(ml_results)} models")
    print(f"    Found {len(ml_java_ids)} Java instance IDs")

    ml_java_ids_sorted = sorted(ml_java_ids)

    # ------------------------------------------------------------------
    # Step 6: Build SWE-bench Multilingual Java matrix
    # ------------------------------------------------------------------
    if ml_results:
        print("\n[Step 6] Building SWE-bench Multilingual Java response matrix...")
        df_ml = build_response_matrix(ml_results, ml_java_ids_sorted)

        ml_summary = build_model_summary(df_ml)
        print_matrix_statistics(
            df_ml,
            "SWE-bench Multilingual Java Subset (43 instances)"
        )
        print_leaderboard(ml_summary, "SWE-bench Multilingual Java Subset")

        ml_path = PROCESSED_DIR / "response_matrix_swebench_multilingual_java.csv"
        df_ml.to_csv(ml_path)
        print(f"\n  Saved: {ml_path}")
    else:
        print("\n[Step 6] No SWE-bench Multilingual Java data found, skipping.")
        df_ml = None

    # ------------------------------------------------------------------
    # Step 7: Build combined response matrix
    # ------------------------------------------------------------------
    print("\n[Step 7] Building combined response matrix...")

    # Combine all Java instance IDs from both sources
    all_java_ids = sorted(set(mswe_instance_ids) | set(ml_java_ids_sorted))
    print(f"    Total unique Java instance IDs: {len(all_java_ids)}")
    print(f"    From Multi-SWE-bench: {len(mswe_instance_ids)}")
    print(f"    From SWE-bench Multilingual: {len(ml_java_ids_sorted)}")
    overlap = set(mswe_instance_ids) & set(ml_java_ids_sorted)
    print(f"    Overlap: {len(overlap)}")

    # Merge model results
    combined_results = {}
    combined_submitted = {}

    # Add Multi-SWE-bench results
    for model, resolved in mswe_results.items():
        combined_results[model] = set(resolved)
        if model in mswe_submitted:
            combined_submitted[model] = set(mswe_submitted[model])

    # Add SWE-bench Multilingual results
    for model, resolved in ml_results.items():
        combined_results[model] = set(resolved)
        combined_submitted[model] = set(ml_java_ids)  # all Java IDs were evaluated

    df_combined = build_response_matrix(
        combined_results, all_java_ids, combined_submitted
    )

    combined_summary = build_model_summary(df_combined)
    print_matrix_statistics(
        df_combined,
        f"Combined SWE-bench Java (all sources, {len(all_java_ids)} instances)"
    )
    print_leaderboard(combined_summary, "Combined SWE-bench Java")

    combined_path = PROCESSED_DIR / "response_matrix.csv"
    df_combined.to_csv(combined_path)
    print(f"\n  Saved: {combined_path}")

    summary_path = PROCESSED_DIR / "model_summary.csv"
    combined_summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Data sources:")
    print(f"    1. Multi-SWE-bench Java Verified: "
          f"{df_mswe.shape[0]} models x {df_mswe.shape[1]} instances")
    if df_ml is not None:
        print(f"    2. SWE-bench Multilingual Java:   "
              f"{df_ml.shape[0]} models x {df_ml.shape[1]} instances")
    print(f"    3. HuggingFace canonical IDs:      {len(hf_ids)} "
          f"(original SWE-bench-java-verified)")

    print(f"\n  Combined matrix:")
    print(f"    Dimensions: {df_combined.shape[0]} models x "
          f"{df_combined.shape[1]} instances")
    n_filled = int(df_combined.notna().sum().sum())
    total = df_combined.shape[0] * df_combined.shape[1]
    fill_rate = n_filled / total * 100 if total > 0 else 0
    vals = df_combined.values[~np.isnan(df_combined.values)]
    mean_resolve = vals.mean() * 100 if len(vals) > 0 else 0
    print(f"    Fill rate:  {fill_rate:.1f}%")
    print(f"    Value distribution: "
          f"1 (resolved) = {vals.sum():.0f}, "
          f"0 (unresolved) = {len(vals) - vals.sum():.0f}")
    print(f"    Mean resolve rate: {mean_resolve:.1f}%")

    print(f"\n  Output files:")
    for f in sorted(PROCESSED_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:55s}  {size_kb:>8.1f} KB")

    print("\nDone.")


if __name__ == "__main__":
    main()
