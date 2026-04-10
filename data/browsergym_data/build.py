"""
Build BrowserGym response matrices from the official BrowserGym leaderboard data.

Data source:
  - ServiceNow/browsergym-leaderboard (HuggingFace Spaces)
    https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard
  - Per-agent JSON files containing aggregate benchmark scores
  - 18 agents evaluated across up to 8 benchmarks

Benchmarks included:
  - MiniWoB: Small web interaction tasks (125 tasks)
  - WebArena: Realistic web tasks (812 tasks)
  - VisualWebArena: Vision-based web tasks
  - WorkArena-L1: Enterprise tasks, Level 1 (33 tasks)
  - WorkArena-L2: Enterprise tasks, Level 2 (compositional)
  - WorkArena-L3: Enterprise tasks, Level 3 (complex compositional)
  - AssistantBench: Web assistant tasks
  - WebLINX: Web interaction traces

Note on data granularity:
  The BrowserGym leaderboard publishes AGGREGATE scores per agent per benchmark
  (success rate +/- standard error), NOT per-task binary pass/fail results.
  Per-task results exist only in the 207GB AgentLab traces dataset
  (https://huggingface.co/datasets/agentlabtraces/agentlabtraces), which contains
  raw experiment traces including screenshots and step-by-step actions.
  Downloading and parsing that dataset is impractical for response matrix extraction.

  As a result, the response matrix here is agents x benchmarks (18 x 8), where each
  cell is the success rate (0-100) for that agent on that benchmark. Missing entries
  (where an agent was not evaluated on a benchmark) are marked as NaN.

Outputs:
  - response_matrix.csv: Success rates (agents x benchmarks), primary matrix
  - response_matrix_with_stderr.csv: Success rates with standard errors
  - agent_metadata.csv: Per-agent summary and metadata
"""

import os
import sys
import json
import urllib.request

import numpy as np
import pandas as pd


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# BrowserGym leaderboard configuration
HF_API_BASE = "https://huggingface.co/api/spaces/ServiceNow/browsergym-leaderboard/tree/main"
HF_RAW_BASE = "https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard/raw/main"

BENCHMARKS = [
    "MiniWoB",
    "WebArena",
    "VisualWebArena",
    "WorkArena-L1",
    "WorkArena-L2",
    "WorkArena-L3",
    "AssistantBench",
    "WebLINX",
]


def download_leaderboard_data():
    """Download all agent results from the BrowserGym HuggingFace leaderboard."""
    print("Downloading BrowserGym leaderboard data from HuggingFace...")

    # Get list of agents
    url = f"{HF_API_BASE}/results"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    items = json.loads(resp.read())
    agents = [
        item["path"].split("/")[-1]
        for item in items
        if item["type"] == "directory"
    ]
    print(f"  Found {len(agents)} agents on leaderboard")

    all_results = {}
    for agent in agents:
        # List files for this agent
        url = f"{HF_API_BASE}/results/{agent}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        files = json.loads(resp.read())
        json_files = [f["path"] for f in files if f["path"].endswith(".json")]

        agent_results = []
        for jf in json_files:
            raw_url = f"{HF_RAW_BASE}/{jf}"
            req2 = urllib.request.Request(
                raw_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            resp2 = urllib.request.urlopen(req2, timeout=15)
            data = json.loads(resp2.read())
            if isinstance(data, list):
                agent_results.extend(data)
            else:
                agent_results.append(data)

        all_results[agent] = agent_results
        n_benchmarks = len(set(
            r.get("benchmark") for r in agent_results
            if r.get("original_or_reproduced") == "Original"
        ))
        print(f"    {agent}: {n_benchmarks} benchmarks")

    # Save raw combined JSON
    output_path = os.path.join(RAW_DIR, "browsergym_leaderboard_all.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved raw data: {output_path}")

    # Also save individual agent files
    agents_dir = os.path.join(RAW_DIR, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    for agent in agents:
        agent_dir = os.path.join(agents_dir, agent)
        os.makedirs(agent_dir, exist_ok=True)

        url = f"{HF_API_BASE}/results/{agent}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        files = json.loads(resp.read())

        for f_info in files:
            fname = f_info["path"].split("/")[-1]
            raw_url = f"{HF_RAW_BASE}/{f_info['path']}"
            local_path = os.path.join(agent_dir, fname)
            try:
                req2 = urllib.request.Request(
                    raw_url, headers={"User-Agent": "Mozilla/5.0"}
                )
                resp2 = urllib.request.urlopen(req2, timeout=15)
                with open(local_path, "wb") as outf:
                    outf.write(resp2.read())
            except Exception as e:
                print(f"    Warning: could not download {fname} for {agent}: {e}")

    return all_results


def load_leaderboard_data():
    """Load leaderboard data from raw JSON (download if not present)."""
    raw_path = os.path.join(RAW_DIR, "browsergym_leaderboard_all.json")
    if not os.path.exists(raw_path):
        return download_leaderboard_data()

    with open(raw_path, "r") as f:
        return json.load(f)


def build_response_matrix(all_results):
    """Build agents x benchmarks response matrix from leaderboard data.

    Returns:
        score_df: DataFrame with agents as index, benchmarks as columns, scores as values
        stderr_df: DataFrame with standard errors
    """
    rows_score = []
    rows_stderr = []

    for agent in sorted(all_results.keys()):
        results = all_results[agent]
        score_row = {"Agent": agent}
        stderr_row = {"Agent": agent}

        for r in results:
            if r.get("original_or_reproduced") != "Original":
                continue
            bm = r.get("benchmark")
            if bm in BENCHMARKS:
                score_row[bm] = r.get("score")
                stderr_row[bm] = r.get("std_err")

        rows_score.append(score_row)
        rows_stderr.append(stderr_row)

    score_df = pd.DataFrame(rows_score).set_index("Agent")
    stderr_df = pd.DataFrame(rows_stderr).set_index("Agent")

    # Reorder columns to match BENCHMARKS order
    score_df = score_df.reindex(columns=BENCHMARKS)
    stderr_df = stderr_df.reindex(columns=BENCHMARKS)

    return score_df, stderr_df


def print_matrix_stats(score_df, label):
    """Print comprehensive summary statistics for a response matrix."""
    n_agents = len(score_df)
    n_benchmarks = len(score_df.columns)
    total_cells = n_agents * n_benchmarks
    n_filled = score_df.notna().sum().sum()
    n_missing = total_cells - n_filled
    fill_rate = n_filled / total_cells

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Agents:          {n_agents}")
    print(f"  Benchmarks:      {n_benchmarks}")
    print(f"  Matrix dims:     {n_agents} x {n_benchmarks}")
    print(f"  Total cells:     {total_cells}")
    print(f"  Filled cells:    {n_filled} ({n_filled/total_cells*100:.1f}%)")
    print(f"  Missing cells:   {n_missing} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:       {fill_rate*100:.1f}%")

    # Overall score statistics (across filled cells)
    all_scores = score_df.values[~np.isnan(score_df.values)]
    if len(all_scores) > 0:
        print(f"\n  Score distribution (across all filled cells):")
        print(f"    Min:    {all_scores.min():.1f}%")
        print(f"    Max:    {all_scores.max():.1f}%")
        print(f"    Mean:   {all_scores.mean():.1f}%")
        print(f"    Median: {np.median(all_scores):.1f}%")
        print(f"    Std:    {all_scores.std():.1f}%")

    # Per-agent stats
    per_agent_mean = score_df.mean(axis=1)
    per_agent_count = score_df.notna().sum(axis=1)
    print(f"\n  Per-agent statistics:")
    print(f"    Benchmarks evaluated (min): {per_agent_count.min()}")
    print(f"    Benchmarks evaluated (max): {per_agent_count.max()}")
    print(f"    Benchmarks evaluated (mean): {per_agent_count.mean():.1f}")

    # Best agent per benchmark
    print(f"\n  Best agent per benchmark:")
    for bm in BENCHMARKS:
        if bm in score_df.columns and score_df[bm].notna().any():
            best_idx = score_df[bm].idxmax()
            best_val = score_df[bm].max()
            n_eval = score_df[bm].notna().sum()
            print(f"    {bm:<20s}: {best_val:5.1f}%  ({best_idx}, {n_eval} agents)")
        else:
            print(f"    {bm:<20s}: no data")

    # Per-benchmark stats
    print(f"\n  Per-benchmark statistics:")
    print(f"    {'Benchmark':<20s}  {'Agents':>6s}  {'Mean':>6s}  {'Std':>6s}  "
          f"{'Min':>6s}  {'Max':>6s}")
    print(f"    {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for bm in BENCHMARKS:
        if bm in score_df.columns:
            vals = score_df[bm].dropna()
            if len(vals) > 0:
                print(f"    {bm:<20s}  {len(vals):>6d}  {vals.mean():>5.1f}%  "
                      f"{vals.std():>5.1f}%  {vals.min():>5.1f}%  {vals.max():>5.1f}%")
            else:
                print(f"    {bm:<20s}  {'0':>6s}  {'N/A':>6s}  {'N/A':>6s}  "
                      f"{'N/A':>6s}  {'N/A':>6s}")

    # Agents sorted by mean score across available benchmarks
    print(f"\n  Agents ranked by mean score (across evaluated benchmarks):")
    agent_means = score_df.mean(axis=1).sort_values(ascending=False)
    for agent in agent_means.index:
        mean_val = agent_means[agent]
        n_bm = score_df.loc[agent].notna().sum()
        if not np.isnan(mean_val):
            print(f"    {agent:<45s}  mean={mean_val:5.1f}%  ({n_bm} benchmarks)")


def build_agent_metadata(all_results, score_df):
    """Build agent metadata summary."""
    rows = []
    for agent in sorted(all_results.keys()):
        results = all_results[agent]
        row = {"agent": agent}

        # Extract study IDs and dates
        study_ids = set()
        dates = []
        for r in results:
            if r.get("original_or_reproduced") == "Original":
                sid = r.get("study_id", "")
                if sid and sid != "study_id":
                    study_ids.add(sid)
                dt = r.get("date_time", "")
                if dt and dt != "2021-01-01 12:00:00":
                    dates.append(dt)

        row["n_benchmarks_evaluated"] = score_df.loc[agent].notna().sum() \
            if agent in score_df.index else 0
        row["mean_score"] = score_df.loc[agent].mean() \
            if agent in score_df.index else np.nan

        # Check agent type
        if agent.startswith("GenericAgent-"):
            row["agent_framework"] = "GenericAgent"
            row["model"] = agent.replace("GenericAgent-", "")
        elif agent.startswith("OrbyAgent-"):
            row["agent_framework"] = "OrbyAgent"
            row["model"] = agent.replace("OrbyAgent-", "")
        else:
            row["agent_framework"] = "unknown"
            row["model"] = agent

        # Reproducibility flags from first original result
        for r in results:
            if r.get("original_or_reproduced") == "Original":
                row["benchmark_specific"] = r.get("benchmark_specific", "")
                row["benchmark_tuned"] = r.get("benchmark_tuned", "")
                row["reproducible"] = r.get("reproducible", "")
                break

        # Add individual benchmark scores
        for bm in BENCHMARKS:
            col_name = bm.lower().replace("-", "_").replace("+", "p")
            row[f"{col_name}_score"] = score_df.loc[agent, bm] \
                if agent in score_df.index and bm in score_df.columns else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("BrowserGym Response Matrix Builder")
    print("=" * 65)
    print()
    print("Data source: ServiceNow/browsergym-leaderboard (HuggingFace)")
    print("URL: https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard")
    print()
    print("NOTE: BrowserGym leaderboard provides aggregate success rates per")
    print("benchmark, not per-task binary pass/fail. Per-task results exist only")
    print("in the 207GB AgentLab traces dataset. The response matrix here uses")
    print("success rates (0-100) as the per-benchmark scores.")

    # Step 1: Load or download data
    print(f"\n{'='*65}")
    print("  STEP 1: Load/download raw data")
    print(f"{'='*65}")
    all_results = load_leaderboard_data()
    print(f"  Loaded data for {len(all_results)} agents")

    # Step 2: Build response matrix
    print(f"\n{'='*65}")
    print("  STEP 2: Build response matrices")
    print(f"{'='*65}")
    score_df, stderr_df = build_response_matrix(all_results)

    # Step 3: Print statistics
    print_matrix_stats(score_df, "BrowserGym Response Matrix (Agents x Benchmarks)")

    # Step 4: Save response matrices
    # Primary: scores only
    output_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    score_df.to_csv(output_path)
    print(f"\n  Saved: {output_path}")

    # With standard errors: combined format
    combined_rows = []
    for agent in score_df.index:
        row = {"Agent": agent}
        for bm in BENCHMARKS:
            score = score_df.loc[agent, bm] if bm in score_df.columns else np.nan
            stderr = stderr_df.loc[agent, bm] if bm in stderr_df.columns else np.nan
            row[bm] = score
            row[f"{bm}_stderr"] = stderr
        combined_rows.append(row)
    combined_df = pd.DataFrame(combined_rows).set_index("Agent")
    output_path2 = os.path.join(PROCESSED_DIR, "response_matrix_with_stderr.csv")
    combined_df.to_csv(output_path2)
    print(f"  Saved: {output_path2}")

    # Step 5: Build and save agent metadata
    metadata_df = build_agent_metadata(all_results, score_df)
    metadata_path = os.path.join(PROCESSED_DIR, "agent_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  Saved: {metadata_path}")

    # Step 6: Final summary
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"\n  PRIMARY response matrix (Agents x Benchmarks):")
    print(f"    Dimensions:  {score_df.shape[0]} agents x {score_df.shape[1]} benchmarks")
    n_filled = score_df.notna().sum().sum()
    total = score_df.shape[0] * score_df.shape[1]
    print(f"    Fill rate:   {n_filled}/{total} ({n_filled/total*100:.1f}%)")
    all_scores = score_df.values[~np.isnan(score_df.values)]
    if len(all_scores) > 0:
        print(f"    Mean score:  {all_scores.mean():.1f}%")

    print(f"\n  Data granularity note:")
    print(f"    This matrix uses benchmark-level aggregate success rates.")
    print(f"    Per-task binary results are available only in the 207GB")
    print(f"    AgentLab traces dataset on HuggingFace:")
    print(f"    https://huggingface.co/datasets/agentlabtraces/agentlabtraces")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:50s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
