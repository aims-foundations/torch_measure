#!/usr/bin/env python3
"""
Build response matrices from AgentDojo benchmark results.

Parses per-task JSON result files from the ethz-spylab/agentdojo repository
and produces:
  1. response_matrix.csv          -- utility scores (tasks x models)
  2. response_matrix_security.csv -- security scores under attack (tasks x models)
  3. task_metadata.csv            -- per-task metadata (suite, task_id, type, etc.)
  4. full_results.csv             -- full long-form results with all fields

Data source: https://github.com/ethz-spylab/agentdojo  (cloned to RUNS_DIR)

Each JSON file contains:
  - suite_name, pipeline_name (model), user_task_id, injection_task_id
  - attack_type (None for no-attack, else attack name)
  - utility (bool), security (bool)
  - error, duration, messages, injections

The response matrices focus on the primary "important_instructions" attack
(the default attack used across all models in the benchmark) and the
"none" condition (no attack) for utility baseline.

Author: auto-generated
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def parse_all_runs(runs_dir: str) -> pd.DataFrame:
    """Walk the runs/ directory tree and parse all JSON result files.

    Directory structure:
      runs/{model}/{suite}/{user_task}/{attack}/{injection_task}.json
      runs/{model}/{suite}/{user_task}/none/none.json  (no-attack baseline)

    Returns a DataFrame with one row per result file.
    """
    rows = []
    runs_path = Path(runs_dir)

    for model_dir in sorted(runs_path.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for suite_dir in sorted(model_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            suite_name = suite_dir.name

            for task_dir in sorted(suite_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_id = task_dir.name  # e.g. user_task_0 or injection_task_0

                for attack_dir in sorted(task_dir.iterdir()):
                    if not attack_dir.is_dir():
                        continue
                    attack_name = attack_dir.name

                    for json_file in sorted(attack_dir.glob("*.json")):
                        try:
                            with open(json_file, "r") as f:
                                data = json.load(f)
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Warning: could not parse {json_file}: {e}",
                                  file=sys.stderr)
                            continue

                        rows.append({
                            "model": model_name,
                            "suite": data.get("suite_name", suite_name),
                            "user_task_id": data.get("user_task_id", task_id),
                            "injection_task_id": data.get("injection_task_id"),
                            "attack_type": data.get("attack_type", attack_name),
                            "utility": data.get("utility"),
                            "security": data.get("security"),
                            "error": data.get("error"),
                            "duration": data.get("duration"),
                            "source_file": str(json_file.relative_to(runs_path)),
                        })

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} result records from {runs_dir}")
    return df


def identify_model_and_defense(model_dir_name: str) -> tuple:
    """Split model directory name into (base_model, defense).

    Models with defenses are stored as:
      gpt-4o-2024-05-13-repeat_user_prompt
      gpt-4o-2024-05-13-tool_filter
      Meta-SecAlign-70B-repeat_user_prompt
      etc.
    """
    defense_suffixes = [
        "repeat_user_prompt",
        "spotlighting_with_delimiting",
        "tool_filter",
        "transformers_pi_detector",
    ]
    for suffix in defense_suffixes:
        if model_dir_name.endswith(f"-{suffix}"):
            base = model_dir_name[: -(len(suffix) + 1)]
            return base, suffix
    return model_dir_name, None


def build_utility_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build tasks x models matrix for UTILITY scores (no-attack condition).

    Under the 'none' attack, each user task has a single result file (none.json).
    utility = True/False -> 1/0 in the matrix.
    """
    # Filter to no-attack results
    no_attack = df[df["attack_type"].isna()].copy()

    # Create a composite task ID
    no_attack["task_key"] = no_attack["suite"] + "/" + no_attack["user_task_id"]

    # Build model label (include defense if present)
    no_attack["model_label"] = no_attack["model"].apply(
        lambda m: (
            f"{identify_model_and_defense(m)[0]} ({identify_model_and_defense(m)[1]})"
            if identify_model_and_defense(m)[1]
            else identify_model_and_defense(m)[0]
        )
    )

    # Convert utility to int
    no_attack["utility_int"] = no_attack["utility"].apply(
        lambda x: int(x) if isinstance(x, bool) else None
    )

    # Pivot
    matrix = no_attack.pivot_table(
        index="task_key",
        columns="model_label",
        values="utility_int",
        aggfunc="first",
    )

    # Sort index naturally
    matrix = matrix.reindex(
        sorted(matrix.index, key=lambda x: (x.split("/")[0],
                                             int(x.split("_")[-1]))),
    )

    return matrix


def build_security_matrix(df: pd.DataFrame,
                          attack: str = "important_instructions") -> pd.DataFrame:
    """Build (user_task, injection_task) x models matrix for SECURITY scores.

    Under attack, each (user_task, injection_task) pair has a result.
    security = True means the model did NOT execute the injection -> secure.
    security = False means injection succeeded -> NOT secure.

    We report security as 1 (secure) / 0 (compromised).
    """
    attack_df = df[df["attack_type"] == attack].copy()

    if attack_df.empty:
        print(f"Warning: no results found for attack '{attack}'", file=sys.stderr)
        return pd.DataFrame()

    # Create composite task key
    attack_df["task_key"] = (
        attack_df["suite"] + "/" +
        attack_df["user_task_id"] + "/" +
        attack_df["injection_task_id"].fillna("none")
    )

    # Build model label
    attack_df["model_label"] = attack_df["model"].apply(
        lambda m: (
            f"{identify_model_and_defense(m)[0]} ({identify_model_and_defense(m)[1]})"
            if identify_model_and_defense(m)[1]
            else identify_model_and_defense(m)[0]
        )
    )

    # Convert security to int
    attack_df["security_int"] = attack_df["security"].apply(
        lambda x: int(x) if isinstance(x, bool) else None
    )

    # Pivot
    matrix = attack_df.pivot_table(
        index="task_key",
        columns="model_label",
        values="security_int",
        aggfunc="first",
    )

    # Sort index
    def sort_key(x):
        parts = x.split("/")
        suite = parts[0]
        ut_num = int(parts[1].split("_")[-1])
        it_num = int(parts[2].split("_")[-1]) if len(parts) > 2 and parts[2] != "none" else -1
        return (suite, ut_num, it_num)

    matrix = matrix.reindex(sorted(matrix.index, key=sort_key))

    return matrix


def build_task_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build task metadata table with task_type classification."""
    # Get unique tasks from the no-attack condition
    no_attack = df[df["attack_type"].isna()].copy()
    if no_attack.empty:
        # Fall back to all data
        no_attack = df.copy()

    no_attack["task_key"] = no_attack["suite"] + "/" + no_attack["user_task_id"]

    meta = no_attack.groupby("task_key").agg(
        suite=("suite", "first"),
        user_task_id=("user_task_id", "first"),
        num_models_evaluated=("model", "nunique"),
    ).reset_index()

    # Classify task type (user_task vs injection_task standalone)
    meta["task_type"] = meta["user_task_id"].apply(
        lambda x: "injection_task" if x.startswith("injection_task") else "user_task"
    )

    # Extract numeric task index
    meta["task_index"] = meta["user_task_id"].apply(
        lambda x: int(x.split("_")[-1])
    )

    # Also get injection task info from the attack data
    attack_df = df[df["attack_type"].notna()].copy()
    if not attack_df.empty:
        inj_counts = (
            attack_df.groupby(["suite", "user_task_id"])["injection_task_id"]
            .nunique()
            .reset_index()
            .rename(columns={"injection_task_id": "num_injection_tasks"})
        )
        inj_counts["task_key"] = inj_counts["suite"] + "/" + inj_counts["user_task_id"]
        meta = meta.merge(
            inj_counts[["task_key", "num_injection_tasks"]],
            on="task_key", how="left"
        )
    else:
        meta["num_injection_tasks"] = 0

    meta["num_injection_tasks"] = meta["num_injection_tasks"].fillna(0).astype(int)

    # Reorder columns
    meta = meta[["task_key", "suite", "task_type", "user_task_id", "task_index",
                  "num_models_evaluated", "num_injection_tasks"]]

    # Sort
    meta = meta.sort_values("task_key").reset_index(drop=True)
    return meta


def build_utility_under_attack_matrix(df: pd.DataFrame,
                                      attack: str = "important_instructions") -> pd.DataFrame:
    """Build tasks x models matrix for UTILITY scores UNDER attack.

    This shows whether models can still complete the user task while
    being subjected to injection attacks.
    """
    attack_df = df[df["attack_type"] == attack].copy()

    if attack_df.empty:
        return pd.DataFrame()

    # For utility under attack, aggregate per user_task (across injection tasks)
    attack_df["task_key"] = attack_df["suite"] + "/" + attack_df["user_task_id"]

    attack_df["model_label"] = attack_df["model"].apply(
        lambda m: (
            f"{identify_model_and_defense(m)[0]} ({identify_model_and_defense(m)[1]})"
            if identify_model_and_defense(m)[1]
            else identify_model_and_defense(m)[0]
        )
    )

    attack_df["utility_int"] = attack_df["utility"].apply(
        lambda x: float(x) if isinstance(x, bool) else None
    )

    # Average utility across injection tasks for each (task, model)
    matrix = attack_df.pivot_table(
        index="task_key",
        columns="model_label",
        values="utility_int",
        aggfunc="mean",
    )

    # Sort
    matrix = matrix.reindex(
        sorted(matrix.index, key=lambda x: (x.split("/")[0],
                                             int(x.split("_")[-1]))),
    )

    return matrix


def print_summary(utility_matrix, security_matrix, utility_attack_matrix,
                  task_meta, full_df):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("AGENTDOJO RESPONSE MATRIX -- SUMMARY REPORT")
    print("=" * 80)

    # Models
    all_models = sorted(set(
        list(utility_matrix.columns) +
        list(security_matrix.columns if not security_matrix.empty else [])
    ))
    print(f"\n--- Models / Configurations ({len(all_models)}) ---")
    for m in all_models:
        print(f"  - {m}")

    # Domains
    suites = sorted(task_meta["suite"].unique())
    print(f"\n--- Domains ({len(suites)}) ---")
    for s in suites:
        suite_meta = task_meta[task_meta["suite"] == s]
        n_user = len(suite_meta[suite_meta["task_type"] == "user_task"])
        n_inj = len(suite_meta[suite_meta["task_type"] == "injection_task"])
        print(f"  - {s}: {n_user} user tasks, {n_inj} injection tasks "
              f"({n_user + n_inj} total)")

    # Utility matrix
    print(f"\n--- Utility Matrix (no attack) ---")
    print(f"  Dimensions: {utility_matrix.shape[0]} tasks x {utility_matrix.shape[1]} models")
    total_cells = utility_matrix.shape[0] * utility_matrix.shape[1]
    filled_cells = utility_matrix.notna().sum().sum()
    print(f"  Fill rate: {filled_cells}/{total_cells} ({100*filled_cells/total_cells:.1f}%)")
    print(f"  Score type: binary (1=pass, 0=fail)")
    print(f"  Overall utility rate: {utility_matrix.mean().mean():.3f} "
          f"({100*utility_matrix.mean().mean():.1f}%)")

    print(f"\n  Per-model utility (no attack):")
    model_util = utility_matrix.mean().sort_values(ascending=False)
    for m, v in model_util.items():
        print(f"    {m}: {100*v:.1f}%")

    # Security matrix
    if not security_matrix.empty:
        print(f"\n--- Security Matrix (important_instructions attack) ---")
        print(f"  Dimensions: {security_matrix.shape[0]} task pairs x "
              f"{security_matrix.shape[1]} models")
        total_cells_s = security_matrix.shape[0] * security_matrix.shape[1]
        filled_cells_s = security_matrix.notna().sum().sum()
        print(f"  Fill rate: {filled_cells_s}/{total_cells_s} "
              f"({100*filled_cells_s/total_cells_s:.1f}%)")
        print(f"  Score type: binary (1=secure/defended, 0=compromised)")
        print(f"  Overall security rate: {security_matrix.mean().mean():.3f} "
              f"({100*security_matrix.mean().mean():.1f}%)")

        print(f"\n  Per-model security (important_instructions attack):")
        model_sec = security_matrix.mean().sort_values(ascending=False)
        for m, v in model_sec.items():
            print(f"    {m}: {100*v:.1f}%")

    # Utility under attack
    if not utility_attack_matrix.empty:
        print(f"\n--- Utility Under Attack Matrix ---")
        print(f"  Dimensions: {utility_attack_matrix.shape[0]} tasks x "
              f"{utility_attack_matrix.shape[1]} models")
        print(f"  Score type: float [0,1] (mean utility across injection tasks)")
        print(f"  Overall utility under attack: "
              f"{utility_attack_matrix.mean().mean():.3f} "
              f"({100*utility_attack_matrix.mean().mean():.1f}%)")

    # Per-suite breakdown
    print(f"\n--- Per-Suite Breakdown ---")
    for s in suites:
        suite_tasks = [t for t in utility_matrix.index if t.startswith(f"{s}/")]
        if suite_tasks:
            suite_util = utility_matrix.loc[suite_tasks].mean().mean()
            print(f"  {s}: {len(suite_tasks)} tasks, "
                  f"avg utility = {100*suite_util:.1f}%")

            if not security_matrix.empty:
                suite_sec_tasks = [t for t in security_matrix.index
                                   if t.startswith(f"{s}/")]
                if suite_sec_tasks:
                    suite_sec = security_matrix.loc[suite_sec_tasks].mean().mean()
                    print(f"    security = {100*suite_sec:.1f}% "
                          f"({len(suite_sec_tasks)} task pairs)")

    # Full dataset stats
    print(f"\n--- Full Dataset Statistics ---")
    print(f"  Total result records parsed: {len(full_df)}")
    n_no_attack = len(full_df[full_df["attack_type"].isna()])
    n_attack = len(full_df[full_df["attack_type"].notna()])
    print(f"  No-attack records: {n_no_attack}")
    print(f"  Attack records: {n_attack}")
    unique_attacks = sorted(full_df["attack_type"].dropna().unique())
    print(f"  Unique attack types: {len(unique_attacks)}")
    for a in unique_attacks:
        n = len(full_df[full_df["attack_type"] == a])
        print(f"    - {a}: {n} records")
    print(f"  Records with errors: {full_df['error'].notna().sum()}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Build response matrices from AgentDojo benchmark results."
    )
    parser.add_argument(
        "--runs_dir", type=str,
        default="/tmp/agentdojo_repo/runs",
        help="Path to the runs/ directory from the agentdojo repo."
    )
    parser.add_argument(
        "--output_dir", type=str,
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
        default=str(_BENCHMARK_DIR / "processed"),
        help="Directory to write output CSV files."
    )
    parser.add_argument(
        "--attack", type=str,
        default="important_instructions",
        help="Attack type for security matrix (default: important_instructions)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Parse all JSON results
    print("Step 1: Parsing all JSON result files...")
    full_df = parse_all_runs(args.runs_dir)

    # Step 2: Build utility matrix (no attack)
    print("\nStep 2: Building utility matrix (no attack)...")
    utility_matrix = build_utility_matrix(full_df)

    # Step 3: Build security matrix (under attack)
    print(f"\nStep 3: Building security matrix (attack={args.attack})...")
    security_matrix = build_security_matrix(full_df, attack=args.attack)

    # Step 4: Build utility-under-attack matrix
    print(f"\nStep 4: Building utility-under-attack matrix...")
    utility_attack_matrix = build_utility_under_attack_matrix(full_df, attack=args.attack)

    # Step 5: Build task metadata
    print("\nStep 5: Building task metadata...")
    task_meta = build_task_metadata(full_df)

    # Step 6: Save outputs
    print("\nStep 6: Saving outputs...")

    utility_path = os.path.join(args.output_dir, "response_matrix.csv")
    utility_matrix.to_csv(utility_path)
    print(f"  Utility matrix saved to: {utility_path}")

    if not security_matrix.empty:
        security_path = os.path.join(args.output_dir, "response_matrix_security.csv")
        security_matrix.to_csv(security_path)
        print(f"  Security matrix saved to: {security_path}")

    if not utility_attack_matrix.empty:
        utility_attack_path = os.path.join(
            args.output_dir, "response_matrix_utility_under_attack.csv"
        )
        utility_attack_matrix.to_csv(utility_attack_path)
        print(f"  Utility-under-attack matrix saved to: {utility_attack_path}")

    task_meta_path = os.path.join(args.output_dir, "task_metadata.csv")
    task_meta.to_csv(task_meta_path, index=False)
    print(f"  Task metadata saved to: {task_meta_path}")

    full_path = os.path.join(args.output_dir, "full_results.csv")
    full_df.to_csv(full_path, index=False)
    print(f"  Full results saved to: {full_path}")

    # Step 7: Summary
    print_summary(utility_matrix, security_matrix, utility_attack_matrix,
                  task_meta, full_df)


if __name__ == "__main__":
    main()
