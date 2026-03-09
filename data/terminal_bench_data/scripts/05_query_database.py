"""
Query the Terminal-Bench Supabase database for per-task per-model results.

Database credentials found in public notebook output in the
terminal-bench-experiments repo (hero_bar_chart.ipynb).
Database has public SELECT access via RLS policies.
"""

import json
import pandas as pd
import psycopg2
from pathlib import Path

# Connection string from public notebook output
DB_CONFIG = {
    "host": "db.jccajjvblmajkbwqsmaz.supabase.co",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "GG8O8Ok5dfUdeTDy",
    "sslmode": "require",
}

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_BENCHMARK_DIR / "raw")


def query_to_df(conn, query, params=None):
    """Execute query and return as DataFrame."""
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def main():
    print("Connecting to Terminal-Bench database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected successfully!")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 1. Get list of tasks in Terminal-Bench 2.0
    print("\n=== Querying tasks ===")
    tasks_df = query_to_df(conn, """
        SELECT task.name as task_name, task.checksum, task.instruction,
               task.metadata
        FROM task
        INNER JOIN dataset_task AS dt ON dt.task_checksum = task.checksum
        WHERE dt.dataset_name = 'terminal-bench'
          AND dt.dataset_version = '2.0'
        ORDER BY task.name
    """)
    print(f"Found {len(tasks_df)} tasks in Terminal-Bench 2.0")
    tasks_df.to_csv(f"{OUTPUT_DIR}/db_tasks.csv", index=False)

    # 2. Get agents
    print("\n=== Querying agents ===")
    agents_df = query_to_df(conn, "SELECT * FROM agent ORDER BY name")
    print(f"Found {len(agents_df)} agents")
    agents_df.to_csv(f"{OUTPUT_DIR}/db_agents.csv", index=False)

    # 3. Get models
    print("\n=== Querying models ===")
    models_df = query_to_df(conn, "SELECT * FROM model ORDER BY name")
    print(f"Found {len(models_df)} models")
    models_df.to_csv(f"{OUTPUT_DIR}/db_models.csv", index=False)

    # 4. Get per-task per-model per-agent results (the key query!)
    print("\n=== Querying per-task per-model results ===")
    results_df = query_to_df(conn, """
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
    results_df.to_csv(f"{OUTPUT_DIR}/db_per_task_results.csv", index=False)

    # 5. Get individual trial results (for binary response matrix)
    print("\n=== Querying individual trial results ===")
    trials_df = query_to_df(conn, """
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
    trials_df.to_csv(f"{OUTPUT_DIR}/db_individual_trials.csv", index=False)

    # Summary statistics
    print(f"\n=== Summary ===")
    print(f"Unique agents: {results_df['agent_name'].nunique()}")
    print(f"Unique models: {results_df['model_name'].nunique()}")
    print(f"Unique tasks: {results_df['task_name'].nunique()}")
    print(f"Total individual trials: {len(trials_df)}")

    print(f"\nAgent-model combinations:")
    combos = results_df.groupby(['agent_name', 'model_name']).agg(
        n_tasks=('task_name', 'nunique'),
        avg_p_hat=('p_hat', 'mean')
    ).reset_index()
    combos = combos.sort_values('avg_p_hat', ascending=False)
    for _, row in combos.head(20).iterrows():
        print(f"  {row['agent_name']:20s} + {row['model_name']:35s} "
              f"tasks={row['n_tasks']:3d}  avg={row['avg_p_hat']*100:5.1f}%")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
