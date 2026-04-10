# Terminal-Bench Data Curation

Ground-truth data for the Terminal-Bench 2.0 benchmark, collected from the
public Supabase database, HuggingFace dataset, and tbench.ai leaderboard.

**Source**: Terminal-Bench paper ([arXiv:2601.11868](https://arxiv.org/abs/2601.11868))
| [GitHub](https://github.com/laude-institute/terminal-bench)
| [Leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0)

## Data Summary

| Item | Count | Source |
|------|-------|--------|
| Tasks (TB 2.0) | 89 | Supabase DB |
| Agent-model combos | 128 | Supabase DB |
| Full-coverage combos (89/89 tasks) | 123 | Supabase DB |
| Individual trials | 56,988 | Supabase DB |
| Unique agents | 31 | Supabase DB |
| Unique models | 42 | Supabase DB |
| Leaderboard entries | 111 | tbench.ai |

## Directory Structure

```
terminal_bench_data/
├── README.md                    # This file
├── raw/                         # Raw data from sources
│   ├── db_tasks.csv             # 89 tasks from DB (name, instruction, metadata)
│   ├── db_agents.csv            # 46 agents
│   ├── db_models.csv            # 61 models
│   ├── db_per_task_results.csv  # 11,288 (agent, model, task) p_hat aggregates
│   ├── db_individual_trials.csv # 56,988 individual trial pass/fail results
│   ├── leaderboard_results.csv  # 111 leaderboard entries (scraped from tbench.ai)
│   ├── paper_table2_results.csv # 55 model-agent results from the paper
│   ├── task_metadata.csv        # 112 tasks from HuggingFace (different task set)
│   └── task_metadata.json
├── processed/                   # Organized, analysis-ready data
│   ├── resolution_rate_matrix.csv     # (128 combos × 89 tasks) resolution rates
│   ├── binary_majority_matrix.csv     # (128 × 89) binary pass/fail (majority vote)
│   ├── binary_response_matrix.json    # (128 × 89) per-trial binary outcomes
│   ├── trials_count_matrix.csv        # (128 × 89) number of trials per cell
│   ├── agent_model_summary.csv        # 128 combos with overall accuracy
│   ├── tasks_complete_metadata.csv    # 89 tasks with merged metadata
│   ├── tasks_complete_metadata.json
│   ├── task_empirical_difficulty.csv   # Avg resolution rate per task
│   ├── task_list_summary.csv
│   ├── all_tasks_metadata.csv         # 112 HuggingFace tasks
│   └── quality_report.txt             # Data quality analysis
├── traces/
│   ├── all_trials_trace_metadata.csv  # 56,988 trial metadata entries
│   └── HOW_TO_DOWNLOAD_TRACES.md      # Instructions for trace access
└── scripts/                     # Data collection scripts
    ├── 01_download_task_metadata.py
    ├── 02_scrape_leaderboard.py
    ├── 02b_parse_leaderboard_table.py
    ├── 03_extract_paper_results.py
    ├── 04_process_task_metadata.py
    ├── 05_query_database.py
    ├── 06_build_response_matrix.py
    ├── 07_quality_checks.py
    ├── 08_merge_metadata.py
    └── 09_fetch_sample_traces.py
```

## Key Files for the Routing Experiment

### Response Matrix
- **`processed/resolution_rate_matrix.csv`** — The (agent-model × task) matrix of
  resolution rates (0.0 to 1.0). 128 rows × 89 columns. 123 combos have full
  coverage across all 89 tasks.
- **`processed/binary_majority_matrix.csv`** — Binarized version (1 if resolution
  rate >= 0.5, else 0).
- **`raw/db_individual_trials.csv`** — All 56,988 individual trial results for
  computing custom aggregations.

### Task Metadata
- **`processed/tasks_complete_metadata.csv`** — All 89 tasks with: category
  (16 categories), human difficulty (easy/medium/hard), empirical difficulty
  (very_easy to very_hard), expert/junior time estimates, and task description.

### Agent Traces
- **`traces/all_trials_trace_metadata.csv`** — Per-trial metadata including
  token counts, execution time, and reward. 98% of trials have a `trial_uri`
  pointing to local trace archives on the job runners. Actual command-level
  traces are **not directly downloadable** — they're stored on the evaluation
  infrastructure, not in public cloud storage.

## Data Quality Notes

See `processed/quality_report.txt` for the full quality report. Key findings:

1. **Coverage**: 123/128 agent-model combos have results on all 89 tasks.
   5 combos are partial (2-88 tasks).
2. **Trial counts**: Median 5 trials per (combo, task). 88% of cells have exactly
   5 trials. Some have up to 11.
3. **Non-determinism**: 33% of cells show mixed outcomes (0 < p_hat < 1).
   43% always fail, 24% always pass.
4. **Error rate**: 16% of trials had errors (AgentTimeoutError,
   VerifierTimeoutError, etc.).
5. **Duplicate model names**: `GLM-4.7` vs `glm-4.7` — same model, different casing.
6. **HuggingFace mismatch**: The HuggingFace dataset (112 tasks) and DB (89 tasks)
   use different task sets. Only 37 task IDs overlap. The DB contains the actual
   evaluated task set.

## Data Access

The raw data was queried from the Terminal-Bench Supabase PostgreSQL database.
Database credentials were found in a public notebook output in the
[terminal-bench-experiments](https://github.com/laude-institute/terminal-bench-experiments)
repo. To re-query:

```python
import psycopg2
conn = psycopg2.connect(
    host="db.jccajjvblmajkbwqsmaz.supabase.co",
    port=5432, dbname="postgres", user="postgres",
    password="GG8O8Ok5dfUdeTDy", sslmode="require"
)
```

The database has RLS policies allowing public SELECT on all tables.

## Contact

For additional data or trace access, contact the Terminal-Bench authors:
- Mike Merrill: mikeam@cs.stanford.edu
- Alex (Laude Institute): alex@laude.org
