#!/usr/bin/env python3
"""
Build the Aider Leaderboard response matrix from published YAML data.

Data source: https://github.com/Aider-AI/aider/tree/main/aider/website/_data/
The Aider benchmark evaluates LLMs on code editing tasks using Exercism exercises.

Two benchmarks exist:
  1. "edit" benchmark: 133 Python-only Exercism exercises (legacy, ~98 models)
  2. "polyglot" benchmark: 225 exercises in C++, Go, Java, JavaScript, Python, Rust (~69 models)

NOTE: Per-exercise pass/fail data is NOT publicly available. The benchmark generates
per-exercise .aider.results.json files locally, but only aggregate results are published
to the leaderboard YAML files. This script works with the published aggregate data.

Output:
  - response_matrix.csv: Models x metrics matrix for both benchmarks
  - model_summary.csv: Condensed summary with key performance metrics
  - data_availability_report.txt: Documentation of what data is/isn't available
"""

import os
import sys
from pathlib import Path

import yaml
import pandas as pd
import json

# Paths
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(filepath):
    """Load a YAML file and return the data."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def normalize_model_name(name):
    """Normalize model name for consistent identification."""
    if name is None:
        return "unknown"
    return str(name).strip()


def build_leaderboard_df(yaml_data, benchmark_name):
    """
    Build a DataFrame from leaderboard YAML data.

    Each row is a model/configuration run. Columns include all available metrics.
    """
    rows = []
    for entry in yaml_data:
        row = {
            "benchmark": benchmark_name,
            "model": normalize_model_name(entry.get("model")),
            "dirname": entry.get("dirname", ""),
            "date": entry.get("date", ""),
            "test_cases": entry.get("test_cases"),
            "edit_format": entry.get("edit_format", ""),
            "pass_rate_1": entry.get("pass_rate_1"),
            "pass_rate_2": entry.get("pass_rate_2"),
            "pass_num_1": entry.get("pass_num_1"),
            "pass_num_2": entry.get("pass_num_2"),
            "percent_cases_well_formed": entry.get("percent_cases_well_formed"),
            "error_outputs": entry.get("error_outputs", 0),
            "num_malformed_responses": entry.get("num_malformed_responses", 0),
            "num_with_malformed_responses": entry.get("num_with_malformed_responses", 0),
            "user_asks": entry.get("user_asks", 0),
            "lazy_comments": entry.get("lazy_comments", 0),
            "syntax_errors": entry.get("syntax_errors", 0),
            "indentation_errors": entry.get("indentation_errors", 0),
            "exhausted_context_windows": entry.get("exhausted_context_windows", 0),
            "test_timeouts": entry.get("test_timeouts", 0),
            "total_tests": entry.get("total_tests"),
            "command": entry.get("command", ""),
            "versions": entry.get("versions", ""),
            "seconds_per_case": entry.get("seconds_per_case"),
            "total_cost": entry.get("total_cost"),
            "commit_hash": entry.get("commit_hash", ""),
            "prompt_tokens": entry.get("prompt_tokens"),
            "completion_tokens": entry.get("completion_tokens"),
            "reasoning_effort": entry.get("reasoning_effort"),
            "thinking_tokens": entry.get("thinking_tokens"),
        }

        # Handle released date (two possible keys)
        released = entry.get("released") or entry.get("_released")
        row["released"] = released

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_response_matrix():
    """
    Build the full response matrix from all available leaderboard YAML files.

    Since per-exercise pass/fail data is not publicly available, this constructs
    a model-level matrix where each row is a unique model evaluation run and
    columns contain all available performance metrics.
    """
    all_dfs = []

    # 1. Edit leaderboard (133 Python exercises)
    edit_file = RAW_DIR / "edit_leaderboard.yml"
    if edit_file.exists():
        edit_data = load_yaml(edit_file)
        edit_df = build_leaderboard_df(edit_data, "edit_133_python")
        all_dfs.append(edit_df)
        print(f"Loaded edit leaderboard: {len(edit_data)} entries")

    # 2. Polyglot leaderboard (225 multi-language exercises)
    polyglot_file = RAW_DIR / "polyglot_leaderboard.yml"
    if polyglot_file.exists():
        polyglot_data = load_yaml(polyglot_file)
        polyglot_df = build_leaderboard_df(polyglot_data, "polyglot_225")
        all_dfs.append(polyglot_df)
        print(f"Loaded polyglot leaderboard: {len(polyglot_data)} entries")

    # 3. O1 polyglot leaderboard
    o1_poly_file = RAW_DIR / "o1_polyglot_leaderboard.yml"
    if o1_poly_file.exists():
        o1_poly_data = load_yaml(o1_poly_file)
        o1_poly_df = build_leaderboard_df(o1_poly_data, "o1_polyglot")
        all_dfs.append(o1_poly_df)
        print(f"Loaded o1 polyglot leaderboard: {len(o1_poly_data)} entries")

    # 4. Refactor leaderboard
    refactor_file = RAW_DIR / "refactor_leaderboard.yml"
    if refactor_file.exists():
        refactor_data = load_yaml(refactor_file)
        refactor_df = build_leaderboard_df(refactor_data, "refactor")
        all_dfs.append(refactor_df)
        print(f"Loaded refactor leaderboard: {len(refactor_data)} entries")

    # 5. Qwen3 leaderboard
    qwen3_file = RAW_DIR / "qwen3_leaderboard.yml"
    if qwen3_file.exists():
        qwen3_data = load_yaml(qwen3_file)
        qwen3_df = build_leaderboard_df(qwen3_data, "qwen3_polyglot")
        all_dfs.append(qwen3_df)
        print(f"Loaded Qwen3 leaderboard: {len(qwen3_data)} entries")

    # 6. O1 results
    o1_file = RAW_DIR / "o1_results.yml"
    if o1_file.exists():
        o1_data = load_yaml(o1_file)
        o1_df = build_leaderboard_df(o1_data, "o1_results")
        all_dfs.append(o1_df)
        print(f"Loaded o1 results: {len(o1_data)} entries")

    if not all_dfs:
        print("ERROR: No data files found!")
        sys.exit(1)

    # Combine all leaderboards
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal entries across all leaderboards: {len(combined_df)}")
    print(f"Unique models: {combined_df['model'].nunique()}")
    print(f"Benchmarks: {combined_df['benchmark'].unique().tolist()}")

    return combined_df


def build_model_summary(combined_df):
    """
    Build a condensed model summary with one row per unique model,
    taking the best result for each model across configurations.

    For models that appear in both edit and polyglot benchmarks,
    we keep separate entries with benchmark prefix.
    """
    summary_rows = []

    for benchmark in combined_df["benchmark"].unique():
        bench_df = combined_df[combined_df["benchmark"] == benchmark]

        # Group by model and take the best pass_rate_2 for each
        for model, group in bench_df.groupby("model"):
            valid = group.dropna(subset=["pass_rate_2"])
            if valid.empty:
                continue
            best_idx = valid["pass_rate_2"].idxmax()
            best = group.loc[best_idx]

            summary_rows.append({
                "model": model,
                "benchmark": benchmark,
                "pass_rate_1": best["pass_rate_1"],
                "pass_rate_2": best["pass_rate_2"],
                "pass_num_1": best.get("pass_num_1"),
                "pass_num_2": best.get("pass_num_2"),
                "test_cases": best["test_cases"],
                "edit_format": best["edit_format"],
                "percent_well_formed": best["percent_cases_well_formed"],
                "total_cost": best["total_cost"],
                "seconds_per_case": best["seconds_per_case"],
                "date": best["date"],
                "error_outputs": best["error_outputs"],
                "syntax_errors": best["syntax_errors"],
                "exhausted_context_windows": best["exhausted_context_windows"],
                "test_timeouts": best["test_timeouts"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["benchmark", "pass_rate_2"], ascending=[True, False]
    ).reset_index(drop=True)

    return summary_df


def build_pivot_response_matrix(combined_df):
    """
    Build a pivot-style response matrix: models (rows) x metrics (columns).

    For the polyglot benchmark (the main one), create a matrix where:
    - Rows = unique models
    - Columns = key metrics
    This is the closest to a "response matrix" we can build without per-exercise data.
    """
    # Focus on polyglot as the primary benchmark
    polyglot_benchmarks = ["polyglot_225", "qwen3_polyglot"]
    polyglot_df = combined_df[combined_df["benchmark"].isin(polyglot_benchmarks)].copy()

    if polyglot_df.empty:
        print("WARNING: No polyglot benchmark data found")
        return None

    # For duplicate models, keep the one with highest pass_rate_2
    polyglot_df = polyglot_df.sort_values("pass_rate_2", ascending=False)
    polyglot_df = polyglot_df.drop_duplicates(subset=["model"], keep="first")

    # Set model as index
    metrics = [
        "pass_rate_1", "pass_rate_2", "pass_num_1", "pass_num_2",
        "test_cases", "edit_format", "percent_cases_well_formed",
        "error_outputs", "num_malformed_responses", "syntax_errors",
        "indentation_errors", "exhausted_context_windows", "test_timeouts",
        "total_cost", "seconds_per_case", "date"
    ]

    pivot_df = polyglot_df.set_index("model")[
        [c for c in metrics if c in polyglot_df.columns]
    ]
    pivot_df = pivot_df.sort_values("pass_rate_2", ascending=False)

    return pivot_df


def write_data_availability_report(combined_df):
    """Write a report documenting what data is and isn't available."""
    report_path = PROCESSED_DIR / "data_availability_report.txt"

    # Load exercise list if available
    exercises_file = RAW_DIR / "polyglot_exercises.json"
    exercises = []
    if exercises_file.exists():
        with open(exercises_file) as f:
            exercises = json.load(f)

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AIDER LEADERBOARD DATA AVAILABILITY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATA SOURCE\n")
        f.write("-" * 40 + "\n")
        f.write("Repository: https://github.com/Aider-AI/aider\n")
        f.write("Data files: aider/website/_data/*.yml\n")
        f.write("Leaderboard: https://aider.chat/docs/leaderboards/\n")
        f.write("Polyglot exercises: https://github.com/Aider-AI/polyglot-benchmark\n\n")

        f.write("WHAT IS AVAILABLE (published)\n")
        f.write("-" * 40 + "\n")
        f.write("- Model-level aggregate results (pass rates, error counts, costs)\n")
        f.write("- Per-benchmark YAML files with detailed run configurations\n")
        f.write("- Exercise list (225 polyglot exercises across 6 languages)\n")
        f.write("- Epoch AI mirror with training compute metadata\n\n")

        f.write("WHAT IS NOT AVAILABLE (not published)\n")
        f.write("-" * 40 + "\n")
        f.write("- Per-exercise pass/fail results for each model\n")
        f.write("- Individual .aider.results.json files (generated locally)\n")
        f.write("- Chat history / conversation logs\n")
        f.write("- Per-language pass rates (except in some third-party repos)\n\n")

        f.write("NOTE: The benchmark runner (benchmark/benchmark.py) stores per-exercise\n")
        f.write("results in tmp.benchmarks/<dirname>/<lang>/exercises/practice/<exercise>/\n")
        f.write(".aider.results.json, but these are local-only and not committed to the repo.\n")
        f.write("The published data is only the aggregated YAML files.\n\n")

        f.write("BENCHMARKS COVERED\n")
        f.write("-" * 40 + "\n")
        for benchmark in combined_df["benchmark"].unique():
            bench_df = combined_df[combined_df["benchmark"] == benchmark]
            n_models = bench_df["model"].nunique()
            test_cases = bench_df["test_cases"].mode().iloc[0] if not bench_df["test_cases"].mode().empty else "N/A"
            f.write(f"  {benchmark}: {n_models} unique models, {test_cases} test cases\n")

        f.write(f"\nTotal unique model entries: {len(combined_df)}\n")
        f.write(f"Total unique model names: {combined_df['model'].nunique()}\n")

        if exercises:
            f.write(f"\nPOLYGLOT EXERCISES ({len(exercises)} total)\n")
            f.write("-" * 40 + "\n")
            # Group by language
            lang_counts = {}
            for ex in exercises:
                lang = ex.split("/")[0]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            for lang, count in sorted(lang_counts.items()):
                f.write(f"  {lang}: {count} exercises\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RESPONSE MATRIX INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("Since per-exercise binary (pass/fail) data is not publicly available,\n")
        f.write("the 'response_matrix.csv' contains:\n")
        f.write("  - One row per model evaluation run\n")
        f.write("  - Columns for all available metrics (pass rates, error counts, etc.)\n")
        f.write("  - Multiple runs of the same model may appear (different configs)\n\n")
        f.write("The 'model_summary.csv' provides:\n")
        f.write("  - One row per unique (model, benchmark) pair\n")
        f.write("  - Best pass_rate_2 for models with multiple runs\n\n")
        f.write("To obtain per-exercise data, one would need to:\n")
        f.write("  1. Clone https://github.com/Aider-AI/polyglot-benchmark\n")
        f.write("  2. Run the benchmark locally for each model of interest\n")
        f.write("  3. Collect the .aider.results.json files from each exercise\n")

    print(f"Data availability report written to: {report_path}")


def main():
    print("=" * 70)
    print("Building Aider Leaderboard Response Matrix")
    print("=" * 70)

    # Build the combined dataframe from all YAML files
    combined_df = build_response_matrix()

    # Save the full response matrix (all entries, all metrics)
    response_matrix_path = PROCESSED_DIR / "response_matrix.csv"
    combined_df.to_csv(response_matrix_path, index=False)
    print(f"\nFull response matrix saved to: {response_matrix_path}")
    print(f"  Shape: {combined_df.shape}")

    # Build and save model summary
    summary_df = build_model_summary(combined_df)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nModel summary saved to: {summary_path}")
    print(f"  Shape: {summary_df.shape}")

    # Build and save the polyglot pivot matrix
    pivot_df = build_pivot_response_matrix(combined_df)
    if pivot_df is not None:
        pivot_path = PROCESSED_DIR / "polyglot_response_matrix.csv"
        pivot_df.to_csv(pivot_path)
        print(f"\nPolyglot response matrix saved to: {pivot_path}")
        print(f"  Shape: {pivot_df.shape}")

    # Write data availability report
    write_data_availability_report(combined_df)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for benchmark in sorted(combined_df["benchmark"].unique()):
        bench_df = combined_df[combined_df["benchmark"] == benchmark]
        print(f"\n--- {benchmark} ---")
        print(f"  Entries: {len(bench_df)}")
        print(f"  Unique models: {bench_df['model'].nunique()}")
        if bench_df["pass_rate_2"].notna().any():
            print(f"  Pass rate (try 2) range: "
                  f"{bench_df['pass_rate_2'].min():.1f}% - "
                  f"{bench_df['pass_rate_2'].max():.1f}%")
            print(f"  Pass rate (try 2) mean: "
                  f"{bench_df['pass_rate_2'].mean():.1f}%")
            print(f"  Pass rate (try 2) median: "
                  f"{bench_df['pass_rate_2'].median():.1f}%")

        # Top 5 models
        top5 = bench_df.nlargest(5, "pass_rate_2")[["model", "pass_rate_2", "edit_format"]]
        print("  Top 5 models:")
        for _, row in top5.iterrows():
            print(f"    {row['model']}: {row['pass_rate_2']}% ({row['edit_format']})")

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    for fpath in sorted(PROCESSED_DIR.glob("*")):
        size = fpath.stat().st_size
        print(f"  {fpath.name}: {size:,} bytes")


if __name__ == "__main__":
    main()
