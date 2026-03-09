"""
Fetch sample agent traces from the Terminal-Bench Supabase database.

Traces (trial archives) are stored in Supabase Storage.
This script demonstrates how to retrieve them and documents the process.
"""

import json
import os
import pandas as pd
import psycopg2

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")
OUT_DIR = str(_BENCHMARK_DIR / "processed")
TRACE_DIR = str(_BENCHMARK_DIR / "traces")

DB_CONFIG = {
    "host": "db.jccajjvblmajkbwqsmaz.supabase.co",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "GG8O8Ok5dfUdeTDy",
    "sslmode": "require",
}


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)

    # 1. Query trial_uri availability (links to trace archives in Supabase Storage)
    print("\n=== Checking trace availability ===")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total_trials,
                COUNT(trial_uri) AS trials_with_uri,
                COUNT(*) - COUNT(trial_uri) AS trials_without_uri
            FROM trial AS t
                INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
            WHERE dt.dataset_name = 'terminal-bench'
              AND dt.dataset_version = '2.0'
        """)
        row = cur.fetchone()
        print(f"Total trials: {row[0]}")
        print(f"Trials with trace URI: {row[1]}")
        print(f"Trials without trace URI: {row[2]}")

    # 2. Get sample trial URIs for a few agent-model-task combos
    print("\n=== Sample trace URIs ===")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                t.id AS trial_id,
                t.agent_name,
                tm.model_name,
                task.name AS task_name,
                t.trial_uri,
                t.trial_name,
                COALESCE(t.reward, 0) AS reward,
                t.exception_info->>'exception_type' AS exception_type,
                tm.n_input_tokens,
                tm.n_output_tokens,
                EXTRACT(EPOCH FROM (
                    t.agent_execution_ended_at - t.agent_execution_started_at
                )) AS execution_time_sec,
                t.agent_metadata
            FROM trial AS t
                INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
                INNER JOIN task ON task.checksum = dt.task_checksum
                INNER JOIN trial_model AS tm ON tm.trial_id = t.id
            WHERE dt.dataset_name = 'terminal-bench'
              AND dt.dataset_version = '2.0'
              AND t.trial_uri IS NOT NULL
              AND t.exception_info IS NULL
            ORDER BY t.agent_name, tm.model_name, task.name
            LIMIT 20
        """)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    samples = pd.DataFrame(rows, columns=columns)
    print(f"Got {len(samples)} sample trials with trace URIs")

    # Print samples
    for _, row in samples.iterrows():
        uri = row["trial_uri"] or "N/A"
        print(f"  {row['agent_name']:20s} | {row['model_name']:30s} | "
              f"{row['task_name']:30s} | reward={row['reward']}")
        print(f"    URI: {uri[:100]}...")
        if row["agent_metadata"]:
            meta = row["agent_metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            api_times = meta.get("api_request_times_msec", [])
            print(f"    API calls: {len(api_times)}, "
                  f"exec_time: {row['execution_time_sec']:.0f}s")

    # 3. Save trace metadata for all trials
    print("\n=== Saving trace metadata for all trials ===")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                t.id AS trial_id,
                t.agent_name,
                t.agent_version,
                tm.model_name,
                tm.model_provider,
                task.name AS task_name,
                t.trial_uri,
                COALESCE(t.reward, 0) AS reward,
                t.exception_info->>'exception_type' AS exception_type,
                tm.n_input_tokens,
                tm.n_output_tokens,
                EXTRACT(EPOCH FROM (
                    t.agent_execution_ended_at - t.agent_execution_started_at
                )) AS execution_time_sec
            FROM trial AS t
                INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
                INNER JOIN task ON task.checksum = dt.task_checksum
                INNER JOIN trial_model AS tm ON tm.trial_id = t.id
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
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    traces_meta = pd.DataFrame(rows, columns=columns)
    traces_meta.to_csv(f"{TRACE_DIR}/all_trials_trace_metadata.csv", index=False)
    print(f"Saved {len(traces_meta)} trial trace metadata entries")

    has_uri = traces_meta["trial_uri"].notna().sum()
    print(f"Trials with downloadable traces (trial_uri): {has_uri} "
          f"({has_uri/len(traces_meta):.1%})")

    # 4. Document how to download actual trace files
    doc = """
# How to Download Agent Traces from Terminal-Bench

## Trace Structure
Each trial produces a structured output directory stored in Supabase Storage:
```
runs/{run_id}/{task_id}/{trial_name}/
  panes/
    pre-agent.txt       # terminal state before agent
    post-agent.txt      # terminal state after agent
    post-test.txt       # terminal state after tests
  sessions/*.cast       # asciinema recordings
  commands.txt          # command history
  agent-logs/           # agent-specific debug logs
  results.json          # TrialResults (is_resolved, failure_mode, tokens, etc.)
```

## Access Method
Traces are stored in a Supabase Storage bucket called "trials".
Each trial has a `trial_uri` field pointing to its storage location.

### Using the Supabase client (requires SUPABASE_PUBLISHABLE_KEY):
```python
from supabase import create_client
import os
from pathlib import Path

client = create_client(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_PUBLISHABLE_KEY"],
)

# List files for a specific trial
trial_id = "some-trial-uuid"
files = client.storage.from_("trials").list(trial_id)

# Download a specific file
data = client.storage.from_("trials").download(f"{trial_id}/commands.txt")
```

### Using direct PostgreSQL access:
The trial_uri column in the trial table contains the storage path.
Use the Supabase Storage API to download files at that path.

## Available Data
- Total trials with trace URIs: see all_trials_trace_metadata.csv
- Not all trials have downloadable traces (some may have been cleaned up)
- The trial metadata (tokens, execution time, reward) is always available

## Contacting Authors
For bulk trace access, contact:
- Mike Merrill: mikeam@cs.stanford.edu
- Alex (Laude Institute): alex@laude.org
"""
    with open(f"{TRACE_DIR}/HOW_TO_DOWNLOAD_TRACES.md", "w") as f:
        f.write(doc)
    print(f"\nSaved HOW_TO_DOWNLOAD_TRACES.md")

    conn.close()


if __name__ == "__main__":
    main()
