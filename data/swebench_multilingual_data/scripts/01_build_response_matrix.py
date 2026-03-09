#!/usr/bin/env python3
"""
Build response matrices for SWE-bench Multilingual benchmarks.

Data sources:
  1. SWE-bench Multilingual (official, from SWE-bench/experiments repo):
     - 300 curated instances across 9 languages (C, C++, Go, Java, JS, TS, PHP, Ruby, Rust)
     - 42 repositories
     - Per-instance results in per_instance_details.json with {resolved: true/false}
     - Source: https://github.com/SWE-bench/experiments/tree/main/evaluation/multilingual

  2. Multi-SWE-bench (ByteDance, from multi-swe-bench/experiments repo):
     - ~1,632 verified instances across 8 languages (C, C++, Go, Java, JS, TS, Python, Rust)
     - Per-language results.json with resolved/unresolved instance ID lists
     - Source: https://github.com/multi-swe-bench/experiments

Outputs:
  - response_matrix.csv:
      Primary matrix from SWE-bench Multilingual (models x instance_ids), 1=resolved
  - response_matrix_multi_swebench.csv:
      Matrix from Multi-SWE-bench (models x instance_ids across all languages), 1=resolved
  - response_matrix_multi_swebench_{lang}.csv:
      Per-language matrices from Multi-SWE-bench
  - model_summary.csv:
      Per-model summary statistics for SWE-bench Multilingual
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Data directories
SWEBENCH_ML_DIR = RAW_DIR / "swebench_multilingual"
MULTI_SWE_DIR = RAW_DIR / "multi_swebench"


# ===========================================================================
# Part 1: SWE-bench Multilingual (official)
# ===========================================================================

def parse_model_name_swebench_ml(dirname):
    """Extract a clean model name from the submission directory name.

    Format: 20260213_mini-v2.0.0a0_claude-4-5-sonnet
    """
    # Strip date prefix and version string
    parts = dirname.split("_", 1)
    if len(parts) < 2:
        return dirname
    rest = parts[1]
    # Remove mini-vX.Y.Z or mini-vX.Y.Za0 prefix
    rest = re.sub(r"^mini-v[\d.]+[a-z0-9]*_", "", rest)
    return rest


def load_swebench_multilingual():
    """Load all SWE-bench Multilingual per-instance results.

    Returns:
        model_results: {model_name: {instance_id: resolved_bool}}
        all_instance_ids: sorted list of all instance IDs
    """
    model_results = {}
    all_instance_ids = set()

    if not SWEBENCH_ML_DIR.exists():
        print(f"WARNING: {SWEBENCH_ML_DIR} not found", file=sys.stderr)
        return {}, []

    for subdir in sorted(SWEBENCH_ML_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        details_path = subdir / "per_instance_details.json"
        if not details_path.exists():
            continue

        model_name = parse_model_name_swebench_ml(subdir.name)
        try:
            with open(details_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARNING: Skipping {subdir.name}: {e}", file=sys.stderr)
            continue

        instance_results = {}
        for instance_id, info in data.items():
            all_instance_ids.add(instance_id)
            instance_results[instance_id] = 1 if info.get("resolved", False) else 0

        model_results[model_name] = instance_results

    return model_results, sorted(all_instance_ids)


def build_swebench_multilingual_matrix():
    """Build and save the SWE-bench Multilingual response matrix."""
    print("=" * 70)
    print("  SWE-bench Multilingual (official)")
    print("=" * 70)

    model_results, instance_ids = load_swebench_multilingual()

    if not model_results:
        print("  No data found. Skipping.")
        return None

    # Build response matrix: models (rows) x instances (columns)
    rows = {}
    for model_name in sorted(model_results.keys()):
        results = model_results[model_name]
        rows[model_name] = [results.get(iid, np.nan) for iid in instance_ids]

    df = pd.DataFrame.from_dict(rows, orient="index", columns=instance_ids)
    df.index.name = "model"

    n_models = df.shape[0]
    n_instances = df.shape[1]
    total_cells = n_models * n_instances
    n_filled = int(df.notna().sum().sum())
    n_pass = int(df.sum().sum())
    n_fail = n_filled - n_pass
    fill_rate = n_filled / total_cells if total_cells > 0 else 0
    mean_pass = df.values[~np.isnan(df.values)].mean() if n_filled > 0 else 0

    print(f"\n  Models:          {n_models}")
    print(f"  Instances:       {n_instances}")
    print(f"  Matrix dims:     {n_models} x {n_instances}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  Filled cells:    {n_filled:,} ({fill_rate*100:.1f}%)")
    print(f"  Pass cells:      {n_pass:,} ({n_pass/n_filled*100:.1f}%)" if n_filled else "")
    print(f"  Fail cells:      {n_fail:,} ({n_fail/n_filled*100:.1f}%)" if n_filled else "")
    print(f"  Mean pass rate:  {mean_pass*100:.1f}%")

    # Per-model stats
    per_model_pass = df.mean(axis=1)
    print(f"\n  Per-model resolve rate:")
    print(f"    Min:    {per_model_pass.min()*100:.1f}% ({per_model_pass.idxmin()})")
    print(f"    Max:    {per_model_pass.max()*100:.1f}% ({per_model_pass.idxmax()})")
    print(f"    Median: {per_model_pass.median()*100:.1f}%")
    print(f"    Std:    {per_model_pass.std()*100:.1f}%")

    # Per-instance stats
    per_instance_solve = df.mean(axis=0)
    print(f"\n  Per-instance solve rate:")
    print(f"    Min:    {per_instance_solve.min()*100:.1f}%")
    print(f"    Max:    {per_instance_solve.max()*100:.1f}%")
    print(f"    Median: {np.nanmedian(per_instance_solve)*100:.1f}%")
    print(f"    Std:    {per_instance_solve.std()*100:.1f}%")

    # Difficulty distribution
    unsolved = (per_instance_solve == 0).sum()
    solved_all = (per_instance_solve == 1.0).sum()
    hard = (per_instance_solve < 0.2).sum()
    easy = (per_instance_solve > 0.8).sum()
    print(f"\n  Instance difficulty distribution:")
    print(f"    Unsolved by all (0%):     {unsolved}")
    print(f"    Hard (<20%):              {hard}")
    print(f"    Easy (>80%):              {easy}")
    print(f"    Solved by all (100%):     {solved_all}")

    # Language breakdown (infer language from instance_id repo names)
    lang_map = _infer_language_from_instance_ids(instance_ids)
    if lang_map:
        print(f"\n  Language breakdown (inferred from repos):")
        lang_counts = {}
        lang_resolve = {}
        for iid in instance_ids:
            lang = lang_map.get(iid, "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            lang_resolve.setdefault(lang, []).append(per_instance_solve.get(iid, 0))
        for lang in sorted(lang_counts.keys()):
            count = lang_counts[lang]
            avg_resolve = np.nanmean(lang_resolve[lang]) * 100
            print(f"    {lang:12s}: {count:4d} instances, "
                  f"avg resolve rate {avg_resolve:.1f}%")

    # Save
    output_path = PROCESSED_DIR / "response_matrix.csv"
    df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    # Build and save model summary
    summary = pd.DataFrame({
        "model": df.index,
        "resolved_count": df.sum(axis=1).astype(int).values,
        "total_instances": n_instances,
        "resolve_rate": (df.mean(axis=1) * 100).round(2).values,
    })
    summary = summary.sort_values("resolved_count", ascending=False).reset_index(drop=True)

    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Print leaderboard
    print(f"\n  Leaderboard (SWE-bench Multilingual):")
    print(f"  {'Rank':>4s}  {'Model':<35s}  {'Resolved':>8s}  {'Rate':>6s}")
    print(f"  {'----':>4s}  {'-'*35}  {'--------':>8s}  {'------':>6s}")
    for i, (_, row) in enumerate(summary.iterrows()):
        print(f"  {i+1:4d}  {row['model']:<35s}  "
              f"{row['resolved_count']:>8d}  {row['resolve_rate']:>5.1f}%")

    return df


def _infer_language_from_instance_ids(instance_ids):
    """Try to infer programming language from repository names in instance IDs.

    Uses known repo-to-language mappings from SWE-bench Multilingual.
    """
    # Complete repo-language mappings for SWE-bench Multilingual (41 repos, 9 languages)
    repo_lang = {
        # Java (6 repos)
        "apache__druid": "Java",
        "apache__lucene": "Java",
        "google__gson": "Java",
        "javaparser__javaparser": "Java",
        "projectlombok__lombok": "Java",
        "reactivex__rxjava": "Java",
        # JavaScript (5 repos)
        "axios__axios": "JavaScript",
        "facebook__docusaurus": "JavaScript",
        "immutable-js__immutable-js": "JavaScript",
        "mrdoob__three.js": "JavaScript",
        "preactjs__preact": "JavaScript",
        # TypeScript (2 repos)
        "babel__babel": "TypeScript",
        "vuejs__core": "TypeScript",
        # Go (5 repos)
        "caddyserver__caddy": "Go",
        "gin-gonic__gin": "Go",
        "gohugoio__hugo": "Go",
        "hashicorp__terraform": "Go",
        "prometheus__prometheus": "Go",
        # Rust (7 repos)
        "astral-sh__ruff": "Rust",
        "burntsushi__ripgrep": "Rust",
        "nushell__nushell": "Rust",
        "sharkdp__bat": "Rust",
        "tokio-rs__axum": "Rust",
        "tokio-rs__tokio": "Rust",
        "uutils__coreutils": "Rust",
        # C (4 repos)
        "jqlang__jq": "C",
        "micropython__micropython": "C",
        "redis__redis": "C",
        "valkey-io__valkey": "C",
        # C++ (2 repos)
        "fmtlib__fmt": "C++",
        "nlohmann__json": "C++",
        # PHP (4 repos)
        "briannesbitt__carbon": "PHP",
        "laravel__framework": "PHP",
        "php-cs-fixer__php-cs-fixer": "PHP",
        "phpoffice__phpspreadsheet": "PHP",
        # Ruby (6 repos)
        "faker-ruby__faker": "Ruby",
        "fastlane__fastlane": "Ruby",
        "fluent__fluentd": "Ruby",
        "jekyll__jekyll": "Ruby",
        "jordansissel__fpm": "Ruby",
        "rubocop__rubocop": "Ruby",
    }

    lang_map = {}
    for iid in instance_ids:
        # instance_id format: owner__repo-number
        parts = iid.rsplit("-", 1)
        if len(parts) == 2:
            repo_key = parts[0]
            lang_map[iid] = repo_lang.get(repo_key, "unknown")
        else:
            # Handle truncated IDs (e.g., "mrdoob__three" from data issues)
            for repo_prefix, lang in repo_lang.items():
                if iid.startswith(repo_prefix.split("__")[0] + "__"):
                    lang_map[iid] = lang
                    break
            else:
                lang_map[iid] = "unknown"

    return lang_map


# ===========================================================================
# Part 2: Multi-SWE-bench (ByteDance)
# ===========================================================================

def parse_model_name_multi_swe(dirname):
    """Extract a clean model name from Multi-SWE-bench submission directory.

    Format: 20250329_MagentLess_Claude-3.7-Sonnet
    Returns: MagentLess_Claude-3.7-Sonnet
    """
    parts = dirname.split("_", 1)
    if len(parts) < 2:
        return dirname
    return parts[1]


def load_multi_swebench():
    """Load all Multi-SWE-bench results across languages.

    Returns:
        per_lang: {lang: {model: {instance_id: 0/1}}}
        all_instance_ids: sorted list of all instance IDs (across all langs)
    """
    per_lang = {}
    all_instance_ids = set()

    if not MULTI_SWE_DIR.exists():
        print(f"WARNING: {MULTI_SWE_DIR} not found", file=sys.stderr)
        return {}, []

    languages = sorted([
        d.name for d in MULTI_SWE_DIR.iterdir()
        if d.is_dir() and d.name not in ("flash", "mini", "visual_evaluation")
    ])

    for lang in languages:
        lang_dir = MULTI_SWE_DIR / lang
        per_lang[lang] = {}

        for subdir in sorted(lang_dir.iterdir()):
            if not subdir.is_dir():
                continue
            results_path = subdir / "results.json"
            if not results_path.exists():
                continue

            model_name = parse_model_name_multi_swe(subdir.name)
            try:
                with open(results_path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING: Skipping {subdir.name}: {e}", file=sys.stderr)
                continue

            resolved_ids = set(data.get("resolved", []))
            # All submitted IDs form the universe for this submission
            submitted_ids = set(data.get("submitted_ids", []))
            # unresolved = submitted but not resolved
            unresolved_ids = set(data.get("unresolved_ids", []))
            # Instances that were not submitted (empty_error_patch) count as 0
            empty_error_ids = set(data.get("empty_error_patch_ids", []))

            # Build per-instance results
            instance_results = {}
            # All submitted IDs get a value
            for iid in submitted_ids:
                all_instance_ids.add(iid)
                instance_results[iid] = 1 if iid in resolved_ids else 0

            per_lang[lang][model_name] = instance_results

    return per_lang, sorted(all_instance_ids)


def build_multi_swebench_matrices():
    """Build Multi-SWE-bench response matrices (combined + per-language)."""
    print("\n" + "=" * 70)
    print("  Multi-SWE-bench (ByteDance)")
    print("=" * 70)

    per_lang, all_instance_ids = load_multi_swebench()

    if not per_lang:
        print("  No data found. Skipping.")
        return None

    # Collect all unique model names across languages
    all_models = set()
    for lang, model_data in per_lang.items():
        all_models.update(model_data.keys())
    all_models = sorted(all_models)

    print(f"\n  Languages: {len(per_lang)}")
    print(f"  Unique models: {len(all_models)}")
    print(f"  Total unique instance IDs: {len(all_instance_ids)}")

    # --- Per-language matrices ---
    lang_matrices = {}
    for lang in sorted(per_lang.keys()):
        model_data = per_lang[lang]
        # Get all instance IDs for this language
        lang_ids = set()
        for results in model_data.values():
            lang_ids.update(results.keys())
        lang_ids = sorted(lang_ids)

        if not lang_ids:
            continue

        lang_models = sorted(model_data.keys())
        rows = {}
        for model in lang_models:
            results = model_data[model]
            rows[model] = [results.get(iid, np.nan) for iid in lang_ids]

        df_lang = pd.DataFrame.from_dict(rows, orient="index", columns=lang_ids)
        df_lang.index.name = "model"
        lang_matrices[lang] = df_lang

        n_filled = int(df_lang.notna().sum().sum())
        total = df_lang.shape[0] * df_lang.shape[1]
        fill_rate = n_filled / total * 100 if total > 0 else 0
        n_pass = int(np.nansum(df_lang.values))
        mean_resolve = np.nanmean(df_lang.values) * 100 if n_filled > 0 else 0

        print(f"\n  {lang}:")
        print(f"    Models: {df_lang.shape[0]}, Instances: {df_lang.shape[1]}")
        print(f"    Fill rate: {fill_rate:.1f}%, Mean resolve: {mean_resolve:.1f}%")

        # Save per-language matrix
        lang_path = PROCESSED_DIR / f"response_matrix_multi_swebench_{lang}.csv"
        df_lang.to_csv(lang_path)
        print(f"    Saved: {lang_path}")

    # --- Combined matrix (all languages) ---
    print(f"\n  Building combined Multi-SWE-bench matrix...")
    rows_combined = {}
    for model in all_models:
        row = []
        for iid in all_instance_ids:
            val = np.nan
            for lang, model_data in per_lang.items():
                if model in model_data and iid in model_data[model]:
                    val = model_data[model][iid]
                    break
            row.append(val)
        rows_combined[model] = row

    df_combined = pd.DataFrame.from_dict(
        rows_combined, orient="index", columns=all_instance_ids
    )
    df_combined.index.name = "model"

    n_models = df_combined.shape[0]
    n_instances = df_combined.shape[1]
    total_cells = n_models * n_instances
    n_filled = int(df_combined.notna().sum().sum())
    fill_rate = n_filled / total_cells * 100 if total_cells > 0 else 0
    n_pass = int(np.nansum(df_combined.values))
    mean_resolve = np.nanmean(
        df_combined.values[~np.isnan(df_combined.values)]
    ) * 100 if n_filled > 0 else 0

    print(f"\n  Combined matrix:")
    print(f"    Models:        {n_models}")
    print(f"    Instances:     {n_instances}")
    print(f"    Matrix dims:   {n_models} x {n_instances}")
    print(f"    Total cells:   {total_cells:,}")
    print(f"    Filled cells:  {n_filled:,} ({fill_rate:.1f}%)")
    print(f"    Pass cells:    {n_pass:,}")
    print(f"    Mean resolve:  {mean_resolve:.1f}%")

    combined_path = PROCESSED_DIR / "response_matrix_multi_swebench.csv"
    df_combined.to_csv(combined_path)
    print(f"    Saved: {combined_path}")

    return df_combined


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("SWE-bench Multilingual Response Matrix Builder")
    print("=" * 70)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")

    # Part 1: SWE-bench Multilingual (official)
    df_ml = build_swebench_multilingual_matrix()

    # Part 2: Multi-SWE-bench (ByteDance)
    df_multi = build_multi_swebench_matrices()

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    if df_ml is not None:
        print(f"\n  SWE-bench Multilingual (primary):")
        print(f"    Dimensions: {df_ml.shape[0]} models x {df_ml.shape[1]} instances")
        n_filled = int(df_ml.notna().sum().sum())
        total = df_ml.shape[0] * df_ml.shape[1]
        print(f"    Fill rate:  {n_filled/total*100:.1f}%")
        print(f"    Mean pass:  {np.nanmean(df_ml.values)*100:.1f}%")

    if df_multi is not None:
        n_filled = int(df_multi.notna().sum().sum())
        total = df_multi.shape[0] * df_multi.shape[1]
        print(f"\n  Multi-SWE-bench (combined):")
        print(f"    Dimensions: {df_multi.shape[0]} models x {df_multi.shape[1]} instances")
        print(f"    Fill rate:  {n_filled/total*100:.1f}%")
        print(f"    Mean pass:  {np.nanmean(df_multi.values[~np.isnan(df_multi.values)])*100:.1f}%")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = PROCESSED_DIR / f
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:55s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
