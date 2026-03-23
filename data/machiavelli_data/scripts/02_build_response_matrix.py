"""
02_build_response_matrix.py — Machiavelli Benchmark dataset exploration and processing.

Loads from raw/machiavelli (git clone of aypan17/machiavelli).
Expected structure:
  - experiments/results/ — main_results.csv, main_results_expanded.csv, count_achievement_change.csv
  - machiavelli/annotate/ — ethical annotation code (labels, metrics)
  - machiavelli/game/ — game environment code
  - machiavelli/agent/ — agent configs including game2beta.json
Explores the game/annotation structure. Summarizes ethical annotations.
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


def find_data_files(base_dir, extensions=(".json", ".csv", ".jsonl", ".pkl", ".parquet")):
    """Find data files recursively, excluding .git."""
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                data_files.append(os.path.join(root, f))
    return sorted(data_files)


def main():
    print("Machiavelli Benchmark Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "machiavelli"
    if not repo_dir.exists():
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "machiavelli" in candidate.name.lower():
                repo_dir = candidate
                break

    if not repo_dir.exists():
        print(f"\nERROR: Repo not found at {repo_dir}")
        return

    print(f"\nRepo directory: {repo_dir}")
    print(f"  Contents: {sorted(os.listdir(repo_dir))}")

    # List all data files
    data_files = find_data_files(repo_dir)
    print(f"\nData files: {len(data_files)}")
    for f in data_files:
        rel = os.path.relpath(f, repo_dir)
        size_kb = os.path.getsize(f) / 1024
        print(f"  {rel} ({size_kb:.1f} KB)")

    # Load experiment results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    results_dir = repo_dir / "experiments" / "results"
    results_dfs = {}

    if results_dir.exists():
        csv_files = sorted(results_dir.glob("*.csv"))
        print(f"Result CSV files: {[f.name for f in csv_files]}")

        for cf in csv_files:
            print(f"\n--- {cf.name} ---")
            try:
                df = pd.read_csv(cf)
                results_dfs[cf.stem] = df
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Dtypes:\n{df.dtypes.to_string()}")
                print(f"  Sample:\n{df.head(5).to_string()}")

                # For numeric columns, show summary stats
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if len(numeric_cols) > 0:
                    print(f"\n  Numeric summary:")
                    print(df[numeric_cols].describe().to_string())

                # For categorical columns, show value counts
                cat_cols = df.select_dtypes(include=["object"]).columns
                for col in cat_cols:
                    if df[col].nunique() <= 30:
                        print(f"\n  {col} value counts:")
                        print(df[col].value_counts().to_string())

            except Exception as e:
                print(f"  Error loading: {e}")
    else:
        print("  experiments/results/ not found")

    # Load game2beta.json (maps games to difficulty parameters)
    print("\n" + "=" * 60)
    print("GAME CONFIGURATIONS")
    print("=" * 60)

    game2beta_file = repo_dir / "machiavelli" / "agent" / "game2beta.json"
    game_data = None
    if game2beta_file.exists():
        try:
            with open(game2beta_file) as f:
                game_data = json.load(f)
            print(f"game2beta.json: {len(game_data)} games")
            if isinstance(game_data, dict):
                sample_items = list(game_data.items())[:10]
                for game_name, beta in sample_items:
                    print(f"  {game_name}: {beta}")
                if len(game_data) > 10:
                    print(f"  ... and {len(game_data) - 10} more games")

                # Distribution of beta values
                betas = list(game_data.values())
                if all(isinstance(b, (int, float)) for b in betas):
                    beta_series = pd.Series(betas)
                    print(f"\n  Beta distribution:")
                    print(f"    Mean: {beta_series.mean():.4f}")
                    print(f"    Std: {beta_series.std():.4f}")
                    print(f"    Min: {beta_series.min():.4f}")
                    print(f"    Max: {beta_series.max():.4f}")
                    print(f"    Median: {beta_series.median():.4f}")

                # Save game catalog
                game_catalog = pd.DataFrame([
                    {"game": k, "beta": v} for k, v in game_data.items()
                ])
                game_catalog.to_csv(PROCESSED_DIR / "game_catalog.csv", index=False)
                print(f"\n  -> Saved to processed/game_catalog.csv")
        except Exception as e:
            print(f"Error loading game2beta.json: {e}")
    else:
        print("game2beta.json not found")

    # Explore the games list
    print("\n" + "=" * 60)
    print("GAMES LIST")
    print("=" * 60)

    games_file = repo_dir / "experiments" / "games.txt"
    if games_file.exists():
        try:
            with open(games_file) as f:
                games = [line.strip() for line in f if line.strip()]
            print(f"Games listed: {len(games)}")
            for g in games[:20]:
                print(f"  - {g}")
            if len(games) > 20:
                print(f"  ... and {len(games) - 20} more")
        except Exception as e:
            print(f"Error: {e}")

    # Explore annotation structure
    print("\n" + "=" * 60)
    print("ANNOTATION STRUCTURE")
    print("=" * 60)

    annotate_dir = repo_dir / "machiavelli" / "annotate"
    if annotate_dir.exists():
        py_files = sorted(annotate_dir.glob("*.py"))
        print(f"Annotation files: {[f.name for f in py_files]}")

        # Look at metrics.py to understand ethical dimensions
        metrics_file = annotate_dir / "metrics.py"
        if metrics_file.exists():
            print(f"\n--- metrics.py (ethical dimensions) ---")
            try:
                with open(metrics_file) as f:
                    content = f.read()
                lines = content.split("\n")
                # Find class definitions, function definitions, and key constants
                for i, line in enumerate(lines):
                    if any(kw in line for kw in ["class ", "def ", "METRIC", "LABEL", "DIMENSION",
                                                  "power", "harm", "deception", "utility",
                                                  "morality", "ethics"]):
                        print(f"  L{i+1}: {line.rstrip()[:120]}")
            except Exception as e:
                print(f"  Error: {e}")

        # Look at parse_labels.py for label structure
        labels_file = annotate_dir / "parse_labels.py"
        if labels_file.exists():
            print(f"\n--- parse_labels.py ---")
            try:
                with open(labels_file) as f:
                    content = f.read()
                lines = content.split("\n")
                for i, line in enumerate(lines[:60]):
                    if line.strip() and not line.strip().startswith("#"):
                        if any(kw in line.lower() for kw in ["label", "category", "class ", "def ",
                                                              "dict", "list", "map", "enum"]):
                            print(f"  L{i+1}: {line.rstrip()[:120]}")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print("  annotate/ directory not found")

    # Save results to processed/
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    for name, df in results_dfs.items():
        out_path = PROCESSED_DIR / f"machiavelli_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> Saved {name} to processed/machiavelli_{name}.csv ({len(df)} rows)")

    # Try to build an ethical annotations summary from main_results
    if "main_results" in results_dfs:
        df = results_dfs["main_results"]
        print("\n" + "=" * 60)
        print("ETHICAL ANNOTATIONS SUMMARY (from main_results)")
        print("=" * 60)

        # Identify agent/model column and metric columns
        model_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["agent", "model", "method"])]
        metric_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64")]

        if model_cols:
            print(f"  Agent/model column: {model_cols}")
        print(f"  Metric columns: {metric_cols}")

        if model_cols and metric_cols:
            model_col = model_cols[0]
            for mcol in metric_cols:
                print(f"\n  {mcol} by {model_col}:")
                summary = df.groupby(model_col)[mcol].agg(["mean", "std", "count"])
                print(summary.to_string())

    if "main_results_expanded" in results_dfs:
        df = results_dfs["main_results_expanded"]
        print(f"\nExpanded results: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # If there's a game column and ethics columns, build game x metric matrix
        game_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["game", "scenario"])]
        if game_cols:
            game_col = game_cols[0]
            metric_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64")]
            print(f"\n  Games: {df[game_col].nunique()}")
            print(f"  Metrics: {metric_cols}")

    # Overall summary
    summary_rows = [
        {"metric": "n_result_files", "value": len(results_dfs)},
        {"metric": "n_games", "value": len(game_data) if game_data else 0},
        {"metric": "n_annotation_files", "value": len(list(annotate_dir.glob("*.py"))) if annotate_dir.exists() else 0},
    ]
    for name, df in results_dfs.items():
        summary_rows.append({"metric": f"rows_{name}", "value": len(df)})
        summary_rows.append({"metric": f"cols_{name}", "value": len(df.columns)})

    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"\n  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
