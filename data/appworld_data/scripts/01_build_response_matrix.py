#!/usr/bin/env python3
"""
Build response matrix for AppWorld benchmark (Stony Brook NLP).

AppWorld is an interactive coding agent benchmark with 732 tasks across 4 splits
(train, dev, test_normal, test_challenge), evaluated on Task Goal Completion (TGC)
and Scenario Goal Completion (SGC) metrics.

Data sources:
  1. Leaderboard JSON: https://appworld.dev/leaderboard.json
     - 18 model-agent submissions with aggregate TGC/SGC/interactions per level
  2. Task metadata: HuggingFace dataset LukaszTP/AppWorld-Tasks (732 tasks)
  3. Encrypted bundles: StonyBrookNLP/appworld-leaderboard repo
     - Per-task results exist but are encrypted in .bundle files

KEY FINDING: Per-task binary (pass/fail) results are NOT publicly available.
The leaderboard only reports aggregate percentages by difficulty level.
Per-task results are locked inside encrypted leaderboard.bundle files that
require the `appworld` CLI tool to decrypt and evaluate.

This script:
  - Downloads the leaderboard JSON and task metadata
  - Builds an aggregate response matrix (models x level-split combinations)
  - Builds a task metadata CSV
  - Documents what would be needed for a full per-task matrix

Output files:
  - processed/response_matrix.csv         (aggregate: models x metric-level-split)
  - processed/response_matrix_wide.csv    (wide format for easy reading)
  - processed/task_metadata.csv           (732 tasks with difficulty, split, etc.)
  - processed/leaderboard_raw.json        (raw leaderboard data)
  - processed/ACCESS_NOTES.txt            (how to get per-task data)
"""

import json
import os
import sys
import urllib.request
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")

LEADERBOARD_URL = "https://appworld.dev/leaderboard.json"
TASKS_PARQUET_URL = (
    "https://huggingface.co/api/datasets/LukaszTP/AppWorld-Tasks/parquet/default/train/0.parquet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Download leaderboard JSON
# ---------------------------------------------------------------------------
def download_leaderboard():
    """Download the live leaderboard JSON from appworld.dev."""
    print("[1/4] Downloading leaderboard JSON ...")
    leaderboard_path = os.path.join(RAW_DIR, "leaderboard.json")
    urllib.request.urlretrieve(LEADERBOARD_URL, leaderboard_path)
    with open(leaderboard_path, "r") as f:
        data = json.load(f)
    print(f"       Loaded {len(data)} leaderboard entries.")
    # Also save a copy in processed/
    with open(os.path.join(OUTPUT_DIR, "leaderboard_raw.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data


# ---------------------------------------------------------------------------
# Step 2: Download task metadata from HuggingFace
# ---------------------------------------------------------------------------
def download_task_metadata():
    """Download the AppWorld-Tasks dataset from HuggingFace (LukaszTP)."""
    print("[2/4] Downloading task metadata from HuggingFace ...")
    parquet_path = os.path.join(RAW_DIR, "appworld_tasks.parquet")

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. Install with: pip install pandas pyarrow")
        sys.exit(1)

    urllib.request.urlretrieve(TASKS_PARQUET_URL, parquet_path)
    df = pd.read_parquet(parquet_path)
    print(f"       Loaded {len(df)} tasks with columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Build task metadata CSV
# ---------------------------------------------------------------------------
def build_task_metadata(df_tasks):
    """Build and save task_metadata.csv."""
    import pandas as pd

    print("[3/4] Building task metadata CSV ...")

    rows = []
    for _, row in df_tasks.iterrows():
        # Each task may belong to one split (the datasets column is a list)
        splits = row["datasets"]
        split_name = splits[0] if len(splits) == 1 else ";".join(splits)

        # Extract scenario_id from task_id (format: {generator_id}_{task_number})
        task_id = row["task_id"]
        parts = task_id.rsplit("_", 1)
        scenario_id = parts[0] if len(parts) == 2 else task_id
        task_number = int(parts[1]) if len(parts) == 2 else 1

        rows.append({
            "task_id": task_id,
            "scenario_id": scenario_id,
            "task_number": task_number,
            "split": split_name,
            "difficulty": row["difficulty"],
            "num_apps": row["num_apps"],
            "num_apis": row["num_apis"],
            "num_api_calls": row["num_api_calls"],
            "num_solution_code_lines": row["num_solution_code_lines"],
            "seconds_to_solve": row.get("seconds_to_solve"),
            "seconds_to_evaluate": row.get("seconds_to_evaluate"),
            "instruction_length": len(str(row.get("instruction", ""))),
        })

    df_meta = pd.DataFrame(rows)
    meta_path = os.path.join(OUTPUT_DIR, "task_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"       Saved {len(df_meta)} tasks to {meta_path}")

    # Print summary statistics
    print("\n       === Task Metadata Summary ===")
    print(f"       Total tasks: {len(df_meta)}")
    print(f"       Splits: {dict(df_meta['split'].value_counts())}")
    print(f"       Difficulty levels: {dict(df_meta['difficulty'].value_counts().sort_index())}")
    print(f"       Unique scenarios: {df_meta['scenario_id'].nunique()}")
    print()

    # Cross-tabulation
    ct = pd.crosstab(df_meta["split"], df_meta["difficulty"], margins=True)
    ct.columns = [f"level_{c}" if c != "All" else "total" for c in ct.columns]
    print("       Difficulty x Split cross-tabulation:")
    print(ct.to_string().replace("\n", "\n       "))
    print()

    return df_meta


# ---------------------------------------------------------------------------
# Step 4: Build aggregate response matrix
# ---------------------------------------------------------------------------
def build_response_matrix(leaderboard_data):
    """
    Build the response matrix from leaderboard data.

    Since per-task results are encrypted, we build an AGGREGATE matrix:
      - Rows = model-agent combinations (18 entries)
      - Columns = metric × level × split combinations

    Metrics: task_goal_completion (TGC), scenario_goal_completion (SGC), interactions
    Levels: all, level_1, level_2, level_3
    Splits: test_normal, test_challenge
    """
    import pandas as pd

    print("[4/4] Building aggregate response matrix ...")

    rows = []
    for entry in leaderboard_data:
        row = {
            "id": entry["id"],
            "method": entry["method"]["name"],
            "method_full": entry["method"]["tooltip"],
            "llm": entry["llm"]["name"],
            "llm_full": entry["llm"]["tooltip"],
            "model_agent": f"{entry['method']['name']}_{entry['llm']['name']}",
            "url": entry["url"],
            "date": entry["date"],
        }

        # Extract metrics for each split and level
        for split in ["test_normal", "test_challenge"]:
            if split in entry:
                for level_key, metrics in entry[split].items():
                    level_label = level_key.replace(" ", "_")
                    for metric_name, value in metrics.items():
                        col_name = f"{split}__{level_label}__{metric_name}"
                        row[col_name] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by best test_normal TGC (all levels)
    sort_col = "test_normal__all__task_goal_completion"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Save full matrix
    matrix_path = os.path.join(OUTPUT_DIR, "response_matrix.csv")
    df.to_csv(matrix_path, index=False)
    print(f"       Saved {df.shape[0]} x {df.shape[1]} matrix to {matrix_path}")

    # Build a more readable wide-format summary
    summary_rows = []
    for _, row in df.iterrows():
        summary_rows.append({
            "method": row["method"],
            "llm": row["llm"],
            "llm_full": row["llm_full"],
            "date": row["date"],
            # test_normal
            "TN_TGC_all": row.get("test_normal__all__task_goal_completion"),
            "TN_SGC_all": row.get("test_normal__all__scenario_goal_completion"),
            "TN_interactions": row.get("test_normal__all__interactions"),
            "TN_TGC_L1": row.get("test_normal__level_1__task_goal_completion"),
            "TN_TGC_L2": row.get("test_normal__level_2__task_goal_completion"),
            "TN_TGC_L3": row.get("test_normal__level_3__task_goal_completion"),
            "TN_SGC_L1": row.get("test_normal__level_1__scenario_goal_completion"),
            "TN_SGC_L2": row.get("test_normal__level_2__scenario_goal_completion"),
            "TN_SGC_L3": row.get("test_normal__level_3__scenario_goal_completion"),
            # test_challenge
            "TC_TGC_all": row.get("test_challenge__all__task_goal_completion"),
            "TC_SGC_all": row.get("test_challenge__all__scenario_goal_completion"),
            "TC_interactions": row.get("test_challenge__all__interactions"),
            "TC_TGC_L1": row.get("test_challenge__level_1__task_goal_completion"),
            "TC_TGC_L2": row.get("test_challenge__level_2__task_goal_completion"),
            "TC_TGC_L3": row.get("test_challenge__level_3__task_goal_completion"),
            "TC_SGC_L1": row.get("test_challenge__level_1__scenario_goal_completion"),
            "TC_SGC_L2": row.get("test_challenge__level_2__scenario_goal_completion"),
            "TC_SGC_L3": row.get("test_challenge__level_3__scenario_goal_completion"),
        })

    df_wide = pd.DataFrame(summary_rows)
    wide_path = os.path.join(OUTPUT_DIR, "response_matrix_wide.csv")
    df_wide.to_csv(wide_path, index=False)
    print(f"       Saved wide-format {df_wide.shape[0]} x {df_wide.shape[1]} to {wide_path}")

    # Print summary
    print("\n       === Response Matrix Summary ===")
    print(f"       Entries (model-agent combos): {len(df)}")
    print(f"       Unique methods: {df['method'].nunique()} -> {sorted(df['method'].unique())}")
    print(f"       Unique LLMs: {df['llm'].nunique()} -> {sorted(df['llm'].unique())}")
    print(f"       Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Score ranges
    tn_col = "test_normal__all__task_goal_completion"
    tc_col = "test_challenge__all__task_goal_completion"
    if tn_col in df.columns:
        print(f"       test_normal TGC (all): "
              f"min={df[tn_col].min():.1f}%, max={df[tn_col].max():.1f}%, "
              f"mean={df[tn_col].mean():.1f}%")
    if tc_col in df.columns:
        print(f"       test_challenge TGC (all): "
              f"min={df[tc_col].min():.1f}%, max={df[tc_col].max():.1f}%, "
              f"mean={df[tc_col].mean():.1f}%")
    print()

    # Top 5
    print("       Top 5 by test_normal TGC (all):")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        print(f"         {i+1}. {row['method']} + {row['llm']}: "
              f"TN={row.get(tn_col, 'N/A')}% / TC={row.get(tc_col, 'N/A')}%")
    print()

    return df


# ---------------------------------------------------------------------------
# Step 5: Write access notes
# ---------------------------------------------------------------------------
def write_access_notes():
    """Document how to access per-task results."""
    notes_path = os.path.join(OUTPUT_DIR, "ACCESS_NOTES.txt")
    notes = f"""AppWorld Response Matrix -- Access Notes
========================================
Generated: {datetime.now().isoformat()}

SUMMARY
-------
AppWorld (Stony Brook NLP, ACL 2024 Best Resource Paper) is an interactive
coding agent benchmark with 732 tasks across 9 simulated apps (Spotify,
Amazon, Gmail, Venmo, Splitwise, Todoist, SimpleNote, Phone, FileSystem).

The benchmark's official number is "750 tasks" (from marketing materials),
but the downloadable task metadata contains 732 unique tasks. The discrepancy
may be due to the special BASE_TASK_ID or versioning differences.

WHAT IS PUBLICLY AVAILABLE
--------------------------
1. AGGREGATE scores from the leaderboard (18 model-agent entries):
   - Task Goal Completion (TGC): average % of tests passed per task
   - Scenario Goal Completion (SGC): % of scenarios fully completed
   - Interactions: average number of agent-environment interactions
   - Broken down by: split (test_normal, test_challenge) x level (1, 2, 3)

2. Task metadata (732 tasks):
   - task_id, difficulty (1-3), split, instruction, num_apps, num_apis,
     num_api_calls, num_solution_code_lines, timing metadata

WHAT IS NOT PUBLICLY AVAILABLE
------------------------------
Per-task, per-model binary (pass/fail) results are NOT directly available.

The per-task results are stored inside encrypted `.bundle` files in the
StonyBrookNLP/appworld-leaderboard GitHub repo. Each experiment directory
(e.g., react_gpt4o_test_normal/) contains a single `leaderboard.bundle`
file (132 bytes pointer) that references encrypted data.

HOW TO ACCESS PER-TASK RESULTS
------------------------------
To obtain per-task results, you would need to:

1. Install the appworld package:
   pip install appworld

2. Clone the leaderboard repo:
   git clone https://github.com/StonyBrookNLP/appworld-leaderboard.git

3. Unpack experiment bundles:
   appworld unpack <experiment_name>
   (e.g., appworld unpack react_gpt4o_test_normal)

4. After unpacking, per-task results appear in:
   experiments/outputs/<experiment_name>/tasks/<task_id>/evaluation/report.md

5. Or run evaluation to get structured JSON:
   appworld evaluate <experiment_name> <dataset_name>

   Output format:
   {{
     "aggregate": {{"task_goal_completion": X, "scenario_goal_completion": Y}},
     "individual": {{
       "<task_id>": {{
         "success": true/false,
         "difficulty": 1/2/3,
         "num_tests": N,
         "passes": [...],
         "failures": [...]
       }}
     }}
   }}

NOTE: The encryption uses a password/salt mechanism. The `appworld` package
contains the decryption keys, so standard pip install should suffice.
However, the authors explicitly warn:
  "Do NOT put your experiment outputs in an unencrypted format publicly
   accessible on the internet."

LEADERBOARD ENTRIES (18 total)
------------------------------
Baselines (2024-07-26, from paper):
  - ReAct + GPT-4o, GPT-4 Turbo, LLaMA3-70B, DeepSeekCoder
  - PlanExec + GPT-4o, GPT-4 Turbo, LLaMA3-70B, DeepSeekCoder
  - IPFunCall + GPT-4o, GPT-4 Turbo
  - FullCodeRefl + GPT-4o, GPT-4 Turbo, LLaMA3-70B, DeepSeekCoder

External submissions:
  - LOOP + Qwen2.5-32B (2025-04-09)
  - IBM CUGA + GPT-4.1 (2025-07-12)
  - ReAct + 2 SetBSR Demos + GPT-4o (2025-07-13)
  - Alibaba Cloud ApsaraLab AgentRL + Qwen3-14B (2026-02-15)

TASK SPLIT DETAILS
------------------
  Split            | Level 1 | Level 2 | Level 3 | Total
  -----------------+---------+---------+---------+------
  train            |      36 |      36 |      18 |    90
  dev              |      30 |      24 |       3 |    57
  test_normal      |      57 |      48 |      63 |   168
  test_challenge   |      72 |     150 |     195 |   417
  -----------------+---------+---------+---------+------
  Total            |     195 |     258 |     279 |   732

  test_normal scenarios:   56 (3 tasks each)
  test_challenge scenarios: 139 (3 tasks each)

SCORING DETAILS
---------------
- TGC (Task Goal Completion): Average percentage of state-based unit tests
  passed across all tasks. Not binary -- a task can partially pass (e.g.,
  if 7/10 database checks pass, score = 70%).

- SGC (Scenario Goal Completion): Percentage of scenarios where ALL tasks
  in the scenario are fully completed (100% TGC). This is a stricter metric.

- Interactions: Average number of agent-environment interaction turns.

REFERENCES
----------
- Paper: https://arxiv.org/abs/2407.18901
- Main repo: https://github.com/StonyBrookNLP/appworld
- Leaderboard repo: https://github.com/StonyBrookNLP/appworld-leaderboard
- Leaderboard website: https://appworld.dev/leaderboard
- Task explorer: https://appworld.dev/task-explorer
- HuggingFace tasks: https://huggingface.co/datasets/LukaszTP/AppWorld-Tasks
"""
    with open(notes_path, "w") as f:
        f.write(notes)
    print(f"       Saved access notes to {notes_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("AppWorld Response Matrix Builder")
    print("=" * 70)
    print()

    leaderboard_data = download_leaderboard()
    df_tasks = download_task_metadata()
    df_meta = build_task_metadata(df_tasks)
    df_matrix = build_response_matrix(leaderboard_data)
    write_access_notes()

    print("=" * 70)
    print("DONE. All files saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
