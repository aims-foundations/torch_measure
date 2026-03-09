"""
Build the (agent-model × task) response matrix from individual trial data.

Creates:
1. Binary response matrix (pass/fail per trial)
2. Resolution rate matrix (avg pass rate per agent-model × task)
3. Summary statistics and quality checks
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")
OUT_DIR = str(_BENCHMARK_DIR / "processed")


def main():
    # Load individual trial data
    print("Loading trial data...")
    trials = pd.read_csv(f"{RAW_DIR}/db_individual_trials.csv")
    print(f"Total trials: {len(trials)}")

    # Create agent-model identifier
    trials["agent_model"] = trials["agent_name"] + " | " + trials["model_name"]

    # Load per-task results
    per_task = pd.read_csv(f"{RAW_DIR}/db_per_task_results.csv")
    per_task["agent_model"] = per_task["agent_name"] + " | " + per_task["model_name"]

    # ===== 1. Resolution Rate Matrix =====
    print("\n=== Building Resolution Rate Matrix ===")
    rate_matrix = per_task.pivot_table(
        index="agent_model",
        columns="task_name",
        values="p_hat",
        aggfunc="mean"
    )

    # Filter to agent-model combos with results on all 89 tasks
    full_coverage = rate_matrix.dropna(thresh=89)
    partial_coverage = rate_matrix[~rate_matrix.index.isin(full_coverage.index)]

    print(f"Agent-model combos with full coverage (89 tasks): {len(full_coverage)}")
    print(f"Agent-model combos with partial coverage: {len(partial_coverage)}")

    # Save full resolution rate matrix
    rate_matrix.to_csv(f"{OUT_DIR}/resolution_rate_matrix.csv")
    print(f"Saved resolution_rate_matrix.csv ({rate_matrix.shape})")

    # ===== 2. Binary Pass/Fail Matrix (per trial) =====
    print("\n=== Building Binary Response Matrix ===")

    # For each agent-model × task, create binary pass/fail for each trial
    # First, number the trials within each (agent-model, task) group
    trials_sorted = trials.sort_values(["agent_model", "task_name", "created_at"])
    trials_sorted["trial_num"] = trials_sorted.groupby(
        ["agent_model", "task_name"]
    ).cumcount()

    # Create a multi-index response matrix: rows = agent-model, cols = task
    # Value = list of binary outcomes
    response_lists = trials_sorted.groupby(
        ["agent_model", "task_name"]
    )["reward"].apply(list).reset_index()
    response_lists.columns = ["agent_model", "task_name", "outcomes"]

    # Pivot to wide format
    response_wide = response_lists.pivot(
        index="agent_model", columns="task_name", values="outcomes"
    )
    response_wide.to_json(f"{OUT_DIR}/binary_response_matrix.json", orient="index")
    print(f"Saved binary_response_matrix.json ({response_wide.shape})")

    # ===== 3. Simplified binary matrix (majority vote per agent-model × task) =====
    print("\n=== Building Majority-Vote Binary Matrix ===")
    binary_matrix = rate_matrix.copy()
    binary_matrix = (binary_matrix >= 0.5).astype(int)
    binary_matrix.to_csv(f"{OUT_DIR}/binary_majority_matrix.csv")
    print(f"Saved binary_majority_matrix.csv ({binary_matrix.shape})")

    # ===== 4. Trials count matrix =====
    print("\n=== Building Trials Count Matrix ===")
    count_matrix = per_task.pivot_table(
        index="agent_model",
        columns="task_name",
        values="n_trials",
        aggfunc="sum"
    )
    count_matrix.to_csv(f"{OUT_DIR}/trials_count_matrix.csv")
    print(f"Saved trials_count_matrix.csv")

    # ===== 5. Per-model summary (best agent for each model) =====
    print("\n=== Per-Model Summary ===")
    # Compute overall accuracy per agent-model
    agg = per_task.groupby(["agent_name", "model_name", "agent_model"]).agg(
        avg_resolution_rate=("p_hat", "mean"),
        n_tasks=("task_name", "nunique"),
        total_trials=("n_trials", "sum"),
    ).reset_index()
    agg = agg.sort_values("avg_resolution_rate", ascending=False)
    agg.to_csv(f"{OUT_DIR}/agent_model_summary.csv", index=False)
    print(f"Saved agent_model_summary.csv ({len(agg)} combos)")

    # Best per model
    print(f"\n--- Best agent per model (full 89-task coverage) ---")
    full_agg = agg[agg["n_tasks"] == 89].copy()
    best_per_model = full_agg.loc[
        full_agg.groupby("model_name")["avg_resolution_rate"].idxmax()
    ].sort_values("avg_resolution_rate", ascending=False)
    for _, row in best_per_model.iterrows():
        print(f"  {row['model_name']:40s} {row['avg_resolution_rate']*100:5.1f}% "
              f"(agent: {row['agent_name']}, "
              f"trials: {int(row['total_trials'])})")

    # ===== 6. Task difficulty from model performance =====
    print(f"\n=== Task Difficulty (empirical) ===")
    # Full-coverage combos only
    full_rate = rate_matrix.loc[full_coverage.index]
    task_difficulty = full_rate.mean(axis=0).sort_values()
    print(f"Easiest tasks:")
    for name, rate in task_difficulty.tail(10).items():
        print(f"  {name:40s} {rate*100:5.1f}% avg resolution")
    print(f"Hardest tasks:")
    for name, rate in task_difficulty.head(10).items():
        print(f"  {name:40s} {rate*100:5.1f}% avg resolution")

    task_difficulty.to_csv(f"{OUT_DIR}/task_empirical_difficulty.csv",
                           header=["avg_resolution_rate"])
    print(f"Saved task_empirical_difficulty.csv")

    # ===== Summary stats =====
    print(f"\n=== Overall Statistics ===")
    print(f"Total tasks: {rate_matrix.shape[1]}")
    print(f"Total agent-model combos: {rate_matrix.shape[0]}")
    print(f"  Full coverage (89 tasks): {len(full_coverage)}")
    print(f"  Partial coverage: {len(partial_coverage)}")
    print(f"Total individual trials: {len(trials)}")
    print(f"Matrix density: {rate_matrix.notna().mean().mean():.1%}")

    # Trials per combo stats
    trials_per = count_matrix.sum(axis=1)
    print(f"\nTrials per agent-model combo: "
          f"min={trials_per.min():.0f}, "
          f"max={trials_per.max():.0f}, "
          f"median={trials_per.median():.0f}")

    trials_per_task = count_matrix.values[count_matrix.notna().values]
    print(f"Trials per (agent-model, task): "
          f"min={np.nanmin(trials_per_task):.0f}, "
          f"max={np.nanmax(trials_per_task):.0f}, "
          f"median={np.nanmedian(trials_per_task):.0f}")


if __name__ == "__main__":
    main()
