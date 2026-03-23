"""
02_build_response_matrix.py — ODCV-Bench dataset exploration and processing.

Loads from raw/ODCV-Bench (git clone of the ODCV-Bench repo).
Expected structure:
  - existing_results/scores_<model>.csv — per-model scores
  - existing_results/reasons_<model>.csv — per-model reasons
  - existing_results/results/<model>-<type>/ — detailed results
  - incentivized_scenarios/ and mandated_scenarios/ — scenario definitions
Builds model x scenario misalignment matrix. Saves results.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def list_raw_contents():
    """Recursively list files in raw/ (excluding .git), print summary."""
    print("=" * 60)
    print("FILES IN raw/")
    print("=" * 60)
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), RAW_DIR)
            all_files.append(rel)
    for f in sorted(all_files)[:80]:
        print(f"  {f}")
    if len(all_files) > 80:
        print(f"  ... and {len(all_files) - 80} more files")
    print(f"\nTotal files: {len(all_files)}")
    return all_files


def main():
    print("ODCV-Bench Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "ODCV-Bench"
    if not repo_dir.exists():
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "odcv" in candidate.name.lower():
                repo_dir = candidate
                break

    print(f"\nRepo directory: {repo_dir}")
    print(f"  Contents: {os.listdir(repo_dir) if repo_dir.exists() else 'NOT FOUND'}")

    # Discover scenario directories
    print("\n" + "=" * 60)
    print("SCENARIOS")
    print("=" * 60)

    scenario_types = {}
    for scenario_dir_name in ["incentivized_scenarios", "mandated_scenarios"]:
        scenario_dir = repo_dir / scenario_dir_name
        if scenario_dir.exists():
            scenarios = sorted([d.name for d in scenario_dir.iterdir() if d.is_dir()])
            scenario_types[scenario_dir_name] = scenarios
            print(f"\n{scenario_dir_name}: {len(scenarios)} scenarios")
            for s in scenarios:
                print(f"  - {s}")
                # List files in scenario dir
                sdir = scenario_dir / s
                sfiles = [f.name for f in sdir.iterdir() if not f.name.startswith(".")]
                if sfiles:
                    print(f"    Files: {sfiles[:5]}")
        else:
            print(f"\n{scenario_dir_name}: directory not found")

    # Load existing results (scores CSVs)
    print("\n" + "=" * 60)
    print("EXISTING RESULTS — SCORES")
    print("=" * 60)

    results_dir = repo_dir / "existing_results"
    score_dfs = {}
    reason_dfs = {}

    if results_dir.exists():
        score_files = sorted(results_dir.glob("scores_*.csv"))
        reason_files = sorted(results_dir.glob("reasons_*.csv"))

        print(f"\nScore files: {len(score_files)}")
        for sf in score_files:
            model_name = sf.stem.replace("scores_", "")
            try:
                df = pd.read_csv(sf)
                score_dfs[model_name] = df
                print(f"\n  {model_name}: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                print(f"    Dtypes:\n{df.dtypes.to_string()}")
                print(f"    Sample:\n{df.head(2).to_string()}")
            except Exception as e:
                print(f"  Error loading {sf.name}: {e}")

        print(f"\nReason files: {len(reason_files)}")
        for rf in reason_files:
            model_name = rf.stem.replace("reasons_", "")
            try:
                df = pd.read_csv(rf)
                reason_dfs[model_name] = df
                print(f"\n  {model_name}: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
            except Exception as e:
                print(f"  Error loading {rf.name}: {e}")

        # Check detailed results directory
        detailed_dir = results_dir / "results"
        if detailed_dir.exists():
            print(f"\nDetailed results directories:")
            for d in sorted(detailed_dir.iterdir()):
                if d.is_dir():
                    n_files = len(list(d.iterdir()))
                    print(f"  {d.name}: {n_files} files")
                    # Sample files
                    for f in list(d.iterdir())[:3]:
                        print(f"    {f.name}")
    else:
        print("  existing_results/ not found")

    if not score_dfs:
        print("\nNo score data loaded. Exiting.")
        return

    # Build the model x scenario matrix
    print("\n" + "=" * 60)
    print("MODEL x SCENARIO MATRIX")
    print("=" * 60)

    # Discover the scenario column in score DataFrames
    sample_df = list(score_dfs.values())[0]
    print(f"\nSample score DataFrame columns: {list(sample_df.columns)}")

    # Try to identify scenario and score columns
    scenario_col = None
    score_col_candidates = []
    for col in sample_df.columns:
        if any(kw in col.lower() for kw in ["scenario", "task", "name", "test"]):
            scenario_col = col
        if sample_df[col].dtype in ("float64", "int64", "float32", "int32"):
            score_col_candidates.append(col)

    # If no scenario column found, check if scenarios are rows
    if scenario_col is None:
        # The CSV might have scenario names as the index or first column
        first_col = sample_df.columns[0]
        if sample_df[first_col].dtype == "object":
            scenario_col = first_col
            print(f"  Using first column as scenario: {scenario_col}")
        else:
            print(f"  Could not identify scenario column. Columns: {list(sample_df.columns)}")

    print(f"  Scenario column: {scenario_col}")
    print(f"  Numeric columns: {score_col_candidates}")

    # Build cross-model comparison
    all_model_scores = []
    for model_name, sdf in score_dfs.items():
        sdf_copy = sdf.copy()
        sdf_copy["model"] = model_name
        all_model_scores.append(sdf_copy)

    if all_model_scores:
        combined = pd.concat(all_model_scores, ignore_index=True)
        print(f"\nCombined scores: {combined.shape}")
        print(f"  Models: {combined['model'].unique().tolist()}")
        print(f"  Columns: {list(combined.columns)}")

        # Save combined scores
        combined.to_csv(PROCESSED_DIR / "all_model_scores.csv", index=False)
        print(f"  -> Saved to processed/all_model_scores.csv")

        # Try to build a pivot table (model x scenario)
        if scenario_col and score_col_candidates:
            for score_col in score_col_candidates[:3]:
                try:
                    pivot = combined.pivot_table(
                        index=scenario_col,
                        columns="model",
                        values=score_col,
                        aggfunc="mean",
                    )
                    print(f"\nPivot table ({scenario_col} x model, values={score_col}):")
                    print(pivot.to_string())
                    pivot.to_csv(PROCESSED_DIR / f"matrix_scenario_x_model_{score_col}.csv")
                    print(f"  -> Saved to processed/matrix_scenario_x_model_{score_col}.csv")
                except Exception as e:
                    print(f"  Error building pivot for {score_col}: {e}")

        # Per-model summary statistics
        print("\n" + "=" * 60)
        print("PER-MODEL SUMMARY")
        print("=" * 60)
        model_summary_rows = []
        for model_name, sdf in score_dfs.items():
            row = {"model": model_name, "n_rows": len(sdf)}
            for col in score_col_candidates:
                if col in sdf.columns:
                    row[f"mean_{col}"] = sdf[col].mean()
                    row[f"std_{col}"] = sdf[col].std()
            model_summary_rows.append(row)

        model_summary = pd.DataFrame(model_summary_rows)
        print(model_summary.to_string(index=False))
        model_summary.to_csv(PROCESSED_DIR / "model_summary.csv", index=False)
        print(f"\n  -> Saved to processed/model_summary.csv")

    # Save scenario catalog
    all_scenarios = set()
    for stype, scenarios in scenario_types.items():
        for s in scenarios:
            all_scenarios.add(s)

    scenario_catalog = []
    for s in sorted(all_scenarios):
        entry = {"scenario": s}
        for stype in scenario_types:
            entry[stype] = s in scenario_types.get(stype, [])
        scenario_catalog.append(entry)

    if scenario_catalog:
        pd.DataFrame(scenario_catalog).to_csv(PROCESSED_DIR / "scenario_catalog.csv", index=False)
        print(f"\n  -> Saved scenario catalog ({len(scenario_catalog)} scenarios) to processed/scenario_catalog.csv")

    # Overall summary
    summary_rows = [
        {"metric": "n_models", "value": len(score_dfs)},
        {"metric": "n_scenarios_incentivized", "value": len(scenario_types.get("incentivized_scenarios", []))},
        {"metric": "n_scenarios_mandated", "value": len(scenario_types.get("mandated_scenarios", []))},
        {"metric": "n_score_files", "value": len(score_dfs)},
        {"metric": "n_reason_files", "value": len(reason_dfs)},
    ]
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
