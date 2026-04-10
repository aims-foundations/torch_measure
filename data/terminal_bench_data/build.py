"""
Build the Terminal-Bench response matrix end-to-end.

Pipeline:
  1. download()           -- pull task metadata from HuggingFace (ia03/terminal-bench)
                             and query Supabase for per-task per-model trial results
  2. build_response_matrix() -- pivot trial data into (agent-model x task) matrices
  3. merge_metadata()     -- reconcile HF + DB metadata, add empirical difficulty
  4. save item_content.csv -- task descriptions for downstream analysis

Outputs (all under processed/):
  - response_matrix.csv           (resolution-rate matrix, models x tasks)
  - resolution_rate_matrix.csv    (same, kept for back-compat)
  - binary_majority_matrix.csv    (>= 0.5 threshold)
  - binary_response_matrix.json   (per-trial binary outcomes)
  - trials_count_matrix.csv
  - agent_model_summary.csv
  - task_empirical_difficulty.csv
  - tasks_complete_metadata.csv / .json
  - all_tasks_metadata.csv / .json
  - item_content.csv              (task_id, description)
"""

import ast
import json

import numpy as np
import pandas as pd
import psycopg2
from datasets import load_dataset
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUT_DIR = _BENCHMARK_DIR / "processed"

# ── Supabase credentials (public SELECT via RLS) ───────────────────────────
# Connection string from public notebook output in the
# terminal-bench-experiments repo (hero_bar_chart.ipynb).
DB_CONFIG = {
    "host": "db.jccajjvblmajkbwqsmaz.supabase.co",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "GG8O8Ok5dfUdeTDy",
    "sslmode": "require",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _query_to_df(conn, query, params=None):
    """Execute *query* and return the result as a DataFrame."""
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _parse_metadata(x):
    """Robustly parse a metadata column that may be dict, JSON str, or repr."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return {}
    return {}


# ── Step 1: download ───────────────────────────────────────────────────────

def download():
    """Download task metadata from HuggingFace and query Supabase DB."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- HuggingFace task metadata ----
    print("Loading Terminal-Bench dataset from HuggingFace (ia03/terminal-bench)...")
    ds = load_dataset("ia03/terminal-bench", split="test")
    print(f"Loaded {len(ds)} rows  (columns: {ds.column_names})")

    hf_records = []
    for row in ds:
        hf_records.append({
            "task_id": row.get("task_id"),
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "base_description": row.get("base_description"),
            "tags": row.get("tags"),
            "max_agent_timeout_sec": row.get("max_agent_timeout_sec"),
            "max_test_timeout_sec": row.get("max_test_timeout_sec"),
            "archive_bytes": row.get("archive_bytes"),
            "n_files": row.get("n_files"),
        })

    hf_df = pd.DataFrame(hf_records)
    hf_df.to_csv(RAW_DIR / "task_metadata.csv", index=False)
    with open(RAW_DIR / "task_metadata.json", "w") as f:
        json.dump(hf_records, f, indent=2)
    print(f"Saved task_metadata.csv / .json ({len(hf_df)} tasks)")

    # Also save the processed all_tasks_metadata used later by merge_metadata
    hf_csv = hf_df.copy()
    hf_csv["tags"] = hf_csv["tags"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )
    hf_csv.to_csv(OUT_DIR / "all_tasks_metadata.csv", index=False)
    with open(OUT_DIR / "all_tasks_metadata.json", "w") as f:
        json.dump(hf_records, f, indent=2, default=str)
    print(f"Saved all_tasks_metadata.csv / .json ({len(hf_df)} tasks)")

    # ---- Supabase DB queries ----
    print("\nConnecting to Terminal-Bench database...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("Connected successfully!")

    # Tasks in Terminal-Bench 2.0
    print("\n=== Querying tasks ===")
    tasks_df = _query_to_df(conn, """
        SELECT task.name as task_name, task.checksum, task.instruction,
               task.metadata
        FROM task
        INNER JOIN dataset_task AS dt ON dt.task_checksum = task.checksum
        WHERE dt.dataset_name = 'terminal-bench'
          AND dt.dataset_version = '2.0'
        ORDER BY task.name
    """)
    print(f"Found {len(tasks_df)} tasks in Terminal-Bench 2.0")
    tasks_df.to_csv(RAW_DIR / "db_tasks.csv", index=False)

    # Agents
    print("\n=== Querying agents ===")
    agents_df = _query_to_df(conn, "SELECT * FROM agent ORDER BY name")
    print(f"Found {len(agents_df)} agents")
    agents_df.to_csv(RAW_DIR / "db_agents.csv", index=False)

    # Models
    print("\n=== Querying models ===")
    models_df = _query_to_df(conn, "SELECT * FROM model ORDER BY name")
    print(f"Found {len(models_df)} models")
    models_df.to_csv(RAW_DIR / "db_models.csv", index=False)

    # Per-task per-model per-agent aggregated results
    print("\n=== Querying per-task per-model results ===")
    results_df = _query_to_df(conn, """
        SELECT
            t.agent_name,
            t.agent_version,
            tm.model_name,
            tm.model_provider,
            task.name AS task_name,
            AVG(COALESCE(t.reward, 0)) AS p_hat,
            COUNT(*) AS n_trials,
            SUM(CASE WHEN t.exception_info IS NULL THEN 0 ELSE 1 END) AS n_errors,
            CASE
                WHEN COUNT(*) > 1
                THEN AVG(COALESCE(t.reward, 0)) * (1 - AVG(COALESCE(t.reward, 0))) / (COUNT(*) - 1)
                ELSE NULL
            END AS partial_var,
            AVG(tm.n_input_tokens) AS avg_input_tokens,
            AVG(tm.n_output_tokens) AS avg_output_tokens,
            AVG(EXTRACT(EPOCH FROM (
                t.agent_execution_ended_at - t.agent_execution_started_at
            ))) AS avg_execution_time_sec
        FROM trial AS t
            INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
            INNER JOIN task ON task.checksum = dt.task_checksum
            INNER JOIN trial_model AS tm ON tm.trial_id = t.id
            INNER JOIN job AS j ON j.id = t.job_id
        WHERE dt.dataset_name = 'terminal-bench'
            AND dt.dataset_version = '2.0'
            AND (
                t.exception_info IS NULL
                OR t.exception_info->>'exception_type' IN (
                    'AgentTimeoutError', 'VerifierTimeoutError'
                )
            )
        GROUP BY t.agent_name, t.agent_version, tm.model_name,
                 tm.model_provider, task.name
        ORDER BY t.agent_name, tm.model_name, task.name
    """)
    print(f"Got {len(results_df)} per-task per-model rows")
    results_df.to_csv(RAW_DIR / "db_per_task_results.csv", index=False)

    # Individual trial results (for binary response matrix)
    print("\n=== Querying individual trial results ===")
    trials_df = _query_to_df(conn, """
        SELECT
            t.id AS trial_id,
            t.agent_name,
            t.agent_version,
            tm.model_name,
            tm.model_provider,
            task.name AS task_name,
            COALESCE(t.reward, 0) AS reward,
            t.exception_info->>'exception_type' AS exception_type,
            tm.n_input_tokens,
            tm.n_output_tokens,
            EXTRACT(EPOCH FROM (
                t.agent_execution_ended_at - t.agent_execution_started_at
            )) AS execution_time_sec,
            t.created_at
        FROM trial AS t
            INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
            INNER JOIN task ON task.checksum = dt.task_checksum
            INNER JOIN trial_model AS tm ON tm.trial_id = t.id
            INNER JOIN job AS j ON j.id = t.job_id
        WHERE dt.dataset_name = 'terminal-bench'
            AND dt.dataset_version = '2.0'
            AND (
                t.exception_info IS NULL
                OR t.exception_info->>'exception_type' IN (
                    'AgentTimeoutError', 'VerifierTimeoutError'
                )
            )
        ORDER BY t.agent_name, tm.model_name, task.name, t.created_at
    """)
    print(f"Got {len(trials_df)} individual trial results")
    trials_df.to_csv(RAW_DIR / "db_individual_trials.csv", index=False)

    conn.close()

    # Quick summary
    print(f"\n=== Download summary ===")
    print(f"HF tasks: {len(hf_df)}")
    print(f"DB tasks: {len(tasks_df)}")
    print(f"Agents: {len(agents_df)}, Models: {len(models_df)}")
    print(f"Per-task rows: {len(results_df)}, Individual trials: {len(trials_df)}")


# ── Step 2: build_response_matrix ──────────────────────────────────────────

def build_response_matrix():
    """Pivot trial data into (agent-model x task) matrices."""
    print("\n" + "=" * 60)
    print("Building response matrices")
    print("=" * 60)

    # Load individual trial data
    trials = pd.read_csv(RAW_DIR / "db_individual_trials.csv")
    print(f"Total trials: {len(trials)}")
    trials["agent_model"] = trials["agent_name"] + " | " + trials["model_name"]

    # Load per-task results
    per_task = pd.read_csv(RAW_DIR / "db_per_task_results.csv")
    per_task["agent_model"] = per_task["agent_name"] + " | " + per_task["model_name"]

    # ---- Resolution rate matrix ----
    print("\n=== Building Resolution Rate Matrix ===")
    rate_matrix = per_task.pivot_table(
        index="agent_model", columns="task_name", values="p_hat", aggfunc="mean"
    )
    full_coverage = rate_matrix.dropna(thresh=89)
    partial_coverage = rate_matrix[~rate_matrix.index.isin(full_coverage.index)]
    print(f"Full coverage (89 tasks): {len(full_coverage)}")
    print(f"Partial coverage: {len(partial_coverage)}")

    rate_matrix.to_csv(OUT_DIR / "resolution_rate_matrix.csv")
    rate_matrix.to_csv(OUT_DIR / "response_matrix.csv")
    print(f"Saved resolution_rate_matrix.csv & response_matrix.csv ({rate_matrix.shape})")

    # ---- Binary per-trial outcomes ----
    print("\n=== Building Binary Response Matrix ===")
    trials_sorted = trials.sort_values(["agent_model", "task_name", "created_at"])
    trials_sorted["trial_num"] = trials_sorted.groupby(
        ["agent_model", "task_name"]
    ).cumcount()

    response_lists = (
        trials_sorted.groupby(["agent_model", "task_name"])["reward"]
        .apply(list)
        .reset_index()
    )
    response_lists.columns = ["agent_model", "task_name", "outcomes"]
    response_wide = response_lists.pivot(
        index="agent_model", columns="task_name", values="outcomes"
    )
    response_wide.to_json(OUT_DIR / "binary_response_matrix.json", orient="index")
    print(f"Saved binary_response_matrix.json ({response_wide.shape})")

    # ---- Majority-vote binary matrix ----
    print("\n=== Building Majority-Vote Binary Matrix ===")
    binary_matrix = (rate_matrix >= 0.5).astype(int)
    binary_matrix.to_csv(OUT_DIR / "binary_majority_matrix.csv")
    print(f"Saved binary_majority_matrix.csv ({binary_matrix.shape})")

    # ---- Trials count matrix ----
    print("\n=== Building Trials Count Matrix ===")
    count_matrix = per_task.pivot_table(
        index="agent_model", columns="task_name", values="n_trials", aggfunc="sum"
    )
    count_matrix.to_csv(OUT_DIR / "trials_count_matrix.csv")

    # ---- Agent-model summary ----
    print("\n=== Per-Model Summary ===")
    agg = per_task.groupby(["agent_name", "model_name", "agent_model"]).agg(
        avg_resolution_rate=("p_hat", "mean"),
        n_tasks=("task_name", "nunique"),
        total_trials=("n_trials", "sum"),
    ).reset_index()
    agg = agg.sort_values("avg_resolution_rate", ascending=False)
    agg.to_csv(OUT_DIR / "agent_model_summary.csv", index=False)
    print(f"Saved agent_model_summary.csv ({len(agg)} combos)")

    # ---- Empirical task difficulty ----
    print("\n=== Task Difficulty (empirical) ===")
    full_rate = rate_matrix.loc[full_coverage.index]
    task_difficulty = full_rate.mean(axis=0).sort_values()
    task_difficulty.to_csv(
        OUT_DIR / "task_empirical_difficulty.csv", header=["avg_resolution_rate"]
    )
    print(f"Saved task_empirical_difficulty.csv")

    # ---- Summary stats ----
    print(f"\n=== Overall Statistics ===")
    print(f"Tasks: {rate_matrix.shape[1]}")
    print(f"Agent-model combos: {rate_matrix.shape[0]} "
          f"(full={len(full_coverage)}, partial={len(partial_coverage)})")
    print(f"Individual trials: {len(trials)}")
    print(f"Matrix density: {rate_matrix.notna().mean().mean():.1%}")

    trials_per_task = count_matrix.values[count_matrix.notna().values]
    print(f"Trials per cell: min={np.nanmin(trials_per_task):.0f}, "
          f"max={np.nanmax(trials_per_task):.0f}, "
          f"median={np.nanmedian(trials_per_task):.0f}")


# ── Step 3: merge_metadata ─────────────────────────────────────────────────

def merge_metadata():
    """Reconcile HF + DB task metadata, enrich with empirical difficulty."""
    print("\n" + "=" * 60)
    print("Merging metadata")
    print("=" * 60)

    # DB tasks (the actual TB 2.0 tasks used in evaluations)
    db_tasks = pd.read_csv(RAW_DIR / "db_tasks.csv")
    print(f"DB tasks (TB 2.0 evaluated): {len(db_tasks)}")

    db_tasks["metadata_parsed"] = db_tasks["metadata"].apply(_parse_metadata)

    db_tasks["category"] = db_tasks["metadata_parsed"].apply(lambda x: x.get("category", ""))
    db_tasks["difficulty"] = db_tasks["metadata_parsed"].apply(lambda x: x.get("difficulty", ""))
    db_tasks["tags"] = db_tasks["metadata_parsed"].apply(lambda x: x.get("tags", []))
    db_tasks["expert_time_min"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("expert_time_estimate_min")
    )
    db_tasks["junior_time_min"] = db_tasks["metadata_parsed"].apply(
        lambda x: x.get("junior_time_estimate_min")
    )
    db_tasks["author"] = db_tasks["metadata_parsed"].apply(lambda x: x.get("author", ""))

    # HuggingFace tasks
    hf_tasks = pd.read_csv(OUT_DIR / "all_tasks_metadata.csv")
    print(f"HuggingFace tasks: {len(hf_tasks)}")

    # Empirical difficulty
    emp_diff = pd.read_csv(OUT_DIR / "task_empirical_difficulty.csv", index_col=0)
    emp_diff.columns = ["empirical_avg_resolution_rate"]

    # Merge DB info with empirical difficulty
    merged = db_tasks[[
        "task_name", "category", "difficulty", "tags",
        "expert_time_min", "junior_time_min", "instruction", "author",
    ]].copy()
    merged = merged.merge(emp_diff, left_on="task_name", right_index=True, how="left")

    # Merge with HuggingFace for timeout info
    hf_match = hf_tasks[["task_id", "max_agent_timeout_sec", "max_test_timeout_sec"]].copy()
    merged = merged.merge(
        hf_match, left_on="task_name", right_on="task_id", how="left"
    ).drop(columns=["task_id"], errors="ignore")

    merged = merged.sort_values("empirical_avg_resolution_rate")

    merged["empirical_difficulty"] = pd.cut(
        merged["empirical_avg_resolution_rate"],
        bins=[0, 0.1, 0.3, 0.5, 0.7, 1.01],
        labels=["very_hard", "hard", "medium", "easy", "very_easy"],
    )
    merged["description_length"] = merged["instruction"].str.len()

    # Save CSV
    merged_csv = merged.copy()
    merged_csv["tags"] = merged_csv["tags"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )
    merged_csv.to_csv(OUT_DIR / "tasks_complete_metadata.csv", index=False)

    # Save JSON (preserves lists)
    merged_json = merged.copy()
    merged_json["tags"] = merged_json["tags"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged_json.to_json(
        OUT_DIR / "tasks_complete_metadata.json", orient="records", indent=2
    )
    print(f"Saved tasks_complete_metadata.csv / .json ({len(merged)} tasks)")

    # ---- item_content.csv ----
    print("\n=== Saving item_content.csv ===")
    item_content = merged[["task_name", "instruction"]].copy()
    item_content.columns = ["task_id", "description"]
    item_content = item_content.sort_values("task_id")
    item_content.to_csv(OUT_DIR / "item_content.csv", index=False)
    print(f"Saved item_content.csv ({len(item_content)} tasks)")

    # Summary
    print(f"\n=== Metadata Summary ({len(merged)} tasks) ===")
    print(f"Category distribution:\n{merged['category'].value_counts().to_string()}")
    print(f"\nHuman difficulty:\n{merged['difficulty'].value_counts().to_string()}")
    print(f"\nEmpirical difficulty:\n{merged['empirical_difficulty'].value_counts().to_string()}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    download()
    build_response_matrix()
    merge_metadata()
    print("\nDone!")


if __name__ == "__main__":
    main()
