#!/usr/bin/env python3
"""
Build the SWE-bench Verified response matrix from the experiments repo.

Reads results.json files downloaded from:
  https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/

Each results.json has the structure:
  {
    "no_generation": [...],   # instance IDs with no generated patch
    "no_logs": [...],         # instance IDs with no execution logs
    "resolved": [...]         # instance IDs that were successfully resolved
  }

Outputs:
  - response_matrix.csv: Binary matrix (models x instance_ids), 1=resolved, 0=not resolved
  - model_summary.csv: Per-model summary statistics
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Paths
RAW_DIR = Path(__file__).resolve().parent.parent / "raw" / "results_json"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"


def load_results(results_dir: Path) -> dict:
    """Load all results.json files and return {model_name: set_of_resolved_ids}."""
    model_results = {}
    all_instance_ids = set()

    for fpath in sorted(results_dir.glob("*.json")):
        model_name = fpath.stem  # filename without .json
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARNING: Skipping {fpath.name}: {e}", file=sys.stderr)
            continue

        if "resolved" not in data:
            print(f"WARNING: Skipping {fpath.name}: no 'resolved' key", file=sys.stderr)
            continue

        resolved = set(data["resolved"])
        model_results[model_name] = resolved

        # Collect all instance IDs from all categories
        for key in data:
            if isinstance(data[key], list):
                all_instance_ids.update(data[key])

    return model_results, sorted(all_instance_ids)


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
    summary = pd.DataFrame({
        "model": response_matrix.index,
        "resolved_count": response_matrix.sum(axis=1).values,
        "total_instances": n_instances,
        "resolve_rate": (response_matrix.sum(axis=1).values / n_instances * 100).round(2),
    })
    summary = summary.sort_values("resolved_count", ascending=False).reset_index(drop=True)
    return summary


def main():
    print(f"Loading results from: {RAW_DIR}")
    model_results, instance_ids = load_results(RAW_DIR)

    print(f"Found {len(model_results)} models")
    print(f"Found {len(instance_ids)} unique instance IDs")

    print("Building response matrix...")
    response_matrix = build_response_matrix(model_results, instance_ids)
    print(f"Response matrix shape: {response_matrix.shape}")

    print("Building model summary...")
    summary = build_model_summary(response_matrix)

    # Save outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matrix_path = PROCESSED_DIR / "response_matrix.csv"
    response_matrix.to_csv(matrix_path)
    print(f"Saved response matrix to: {matrix_path}")

    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved model summary to: {summary_path}")

    # Print top/bottom models
    print("\n" + "=" * 80)
    print("TOP 20 MODELS (by resolve rate):")
    print("=" * 80)
    print(summary.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("BOTTOM 10 MODELS (by resolve rate):")
    print("=" * 80)
    print(summary.tail(10).to_string(index=False))

    # Basic statistics
    print("\n" + "=" * 80)
    print("MATRIX STATISTICS:")
    print("=" * 80)
    print(f"  Models (rows): {response_matrix.shape[0]}")
    print(f"  Instances (columns): {response_matrix.shape[1]}")
    print(f"  Total resolved cells: {response_matrix.values.sum()}")
    print(f"  Overall density: {response_matrix.values.mean():.4f}")

    # Per-instance statistics
    instance_resolve_counts = response_matrix.sum(axis=0)
    print(f"\n  Instances resolved by ALL models: "
          f"{(instance_resolve_counts == len(model_results)).sum()}")
    print(f"  Instances resolved by NO model: "
          f"{(instance_resolve_counts == 0).sum()}")
    print(f"  Mean models resolving each instance: "
          f"{instance_resolve_counts.mean():.1f}")
    print(f"  Median models resolving each instance: "
          f"{instance_resolve_counts.median():.1f}")


if __name__ == "__main__":
    main()
