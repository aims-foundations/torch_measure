#!/usr/bin/env python3
"""
Build GAIA (General AI Assistants) response matrices from multiple data sources.

GAIA evaluates AI assistants on 450+ multi-step reasoning problems across three
difficulty levels. The benchmark has two main leaderboards:
  1) HuggingFace official leaderboard (gaia-benchmark/results_public)
  2) HAL (Holistic Agent Leaderboard) at Princeton (hal.cs.princeton.edu/gaia)

Data sources used:
  - HuggingFace: gaia-benchmark/results_public (aggregate scores, public)
    Contains per-model aggregate scores for test and validation splits.
    ~3000 test submissions, ~96 validation submissions.
  - HAL Leaderboard: per-item heatmap data scraped from hal.cs.princeton.edu/gaia
    Contains per-task binary results for 32 agent configurations on 165 validation
    tasks. This is the primary source for the per-item response matrix.
  - HuggingFace: gaia-benchmark/submissions_public (per-item JSONL, GATED)
    Contains 133 JSONL files with per-task scores for validation set.
    ACCESS RESTRICTED: requires manual approval at HuggingFace.
  - HuggingFace: gaia-benchmark/GAIA (questions + metadata, GATED)
    Contains task_id, Question, Level, Final answer for all tasks.
    ACCESS RESTRICTED: requires manual approval at HuggingFace.

Outputs:
  - response_matrix_hal.csv: Binary response matrix from HAL (32 agents x 165 tasks)
  - response_matrix_hf_validation.csv: Aggregate scores for HF validation submissions
  - response_matrix_hf_test.csv: Aggregate scores for HF test submissions
  - model_summary.csv: Combined per-model summary statistics
  - data_source_report.txt: Documentation of all data sources and access status

Notes:
  - The HAL heatmap provides per-item data but uses fraction-of-runs (0.0-1.0).
    We binarize at threshold 0.5 for the primary response matrix.
  - The HF results_public only has aggregate scores, not per-item results.
  - For full per-item data across all 133+ HF submissions, access to the gated
    gaia-benchmark/submissions_public dataset is required.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace token (optional, for gated datasets)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ---------------------------------------------------------------------------
# Step 1: Download HAL leaderboard heatmap data (per-item)
# ---------------------------------------------------------------------------
def download_hal_heatmap():
    """Scrape per-item heatmap data from HAL GAIA leaderboard page.

    The HAL page embeds Plotly chart data as JSON in the HTML source.
    We extract the heatmap trace which contains:
      x: 165 task_ids (UUIDs)
      y: 32 agent names
      z: 32x165 matrix of fraction-of-runs (0.0 to 1.0)
    """
    import urllib.request

    heatmap_path = RAW_DIR / "hal_gaia_heatmap.json"
    scatter_path = RAW_DIR / "hal_gaia_scatter.json"
    table_path = RAW_DIR / "hal_gaia_table.json"
    html_path = RAW_DIR / "hal_gaia_page.html"

    if heatmap_path.exists() and scatter_path.exists():
        print(f"HAL data already downloaded: {heatmap_path}")
        return

    print("Downloading HAL GAIA leaderboard page...")
    url = "https://hal.cs.princeton.edu/gaia"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8")

    html_path.write_text(html, encoding="utf-8")
    print(f"  Saved HTML: {html_path} ({len(html):,} bytes)")

    # Extract heatmap_data
    match = re.search(r"heatmap_data = JSON\.parse\('(.+?)'\)", html, re.DOTALL)
    if not match:
        print("ERROR: Could not find heatmap_data in HAL page", file=sys.stderr)
        return
    raw = match.group(1).replace('\\"', '"')
    heatmap_data = json.loads(raw)
    heatmap_path.write_text(json.dumps(heatmap_data, indent=2), encoding="utf-8")
    print(f"  Saved heatmap: {heatmap_path}")

    # Extract scatter_plot_data (contains agent names, accuracy, cost)
    match = re.search(
        r"scatter_plot_data = JSON\.parse\('(.+?)'\)", html, re.DOTALL
    )
    if match:
        raw = match.group(1).replace('\\"', '"')
        scatter_data = json.loads(raw)
        scatter_path.write_text(
            json.dumps(scatter_data, indent=2), encoding="utf-8"
        )
        print(f"  Saved scatter: {scatter_path}")

    # Extract leaderboard table
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    table_data = []
    for row in rows[1:]:  # Skip header
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(clean) >= 9:
            acc_m = re.match(r"([\d.]+)%", clean[4])
            acc = float(acc_m.group(1)) if acc_m else None
            entry = {
                "rank": int(clean[0]) if clean[0].isdigit() else None,
                "scaffold": clean[1].replace("Pareto optimal", "").strip(),
                "primary_model": clean[2],
                "verified": "\u2713" in clean[3],
                "accuracy": acc,
            }
            table_data.append(entry)
    table_path.write_text(json.dumps(table_data, indent=2), encoding="utf-8")
    print(f"  Saved table: {table_path} ({len(table_data)} entries)")


# ---------------------------------------------------------------------------
# Step 2: Download HuggingFace results_public (aggregate scores)
# ---------------------------------------------------------------------------
def download_hf_results():
    """Download aggregate leaderboard scores from gaia-benchmark/results_public.

    This public dataset contains per-model aggregate scores (overall, L1, L2, L3)
    for both test and validation splits, but NOT per-item results.
    """
    test_path = RAW_DIR / "hf_results_test.parquet"
    val_path = RAW_DIR / "hf_results_validation.parquet"

    if test_path.exists() and val_path.exists():
        print(f"HF results already downloaded: {test_path}")
        return

    print("Downloading HuggingFace results_public...")
    base_url = (
        "https://huggingface.co/api/datasets/"
        "gaia-benchmark/results_public/parquet/2023"
    )

    for split, out_path in [("test", test_path), ("validation", val_path)]:
        url = f"{base_url}/{split}/0.parquet"
        print(f"  Fetching {split} split from: {url}")
        df = pd.read_parquet(url)
        df.to_parquet(out_path)
        print(f"  Saved: {out_path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Step 3: Try to download gated submissions_public (per-item)
# ---------------------------------------------------------------------------
def try_download_submissions():
    """Attempt to download per-item scored submissions from HF.

    The gaia-benchmark/submissions_public dataset is GATED and requires manual
    approval. This function attempts to list and download files, and gracefully
    handles access denial.

    Each file is a JSONL with lines like:
      {"id": "task_id", "model_answer": "...", "score": true/false, "level": 1}
    """
    submissions_dir = RAW_DIR / "submissions_public"

    if submissions_dir.exists() and any(submissions_dir.rglob("*.jsonl")):
        n_files = len(list(submissions_dir.rglob("*.jsonl")))
        print(f"Submissions already present: {n_files} JSONL files")
        return True

    if not HF_TOKEN:
        print(
            "WARNING: No HF_TOKEN set. Cannot attempt gated dataset download."
        )
        print(
            "  Set HF_TOKEN env var and request access at:"
        )
        print(
            "  https://huggingface.co/datasets/gaia-benchmark/submissions_public"
        )
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=HF_TOKEN)

        # List all files
        print("Attempting to access gaia-benchmark/submissions_public...")
        files = list(
            api.list_repo_tree(
                "gaia-benchmark/submissions_public",
                repo_type="dataset",
                recursive=True,
            )
        )
        jsonl_files = [
            f for f in files
            if hasattr(f, "path") and f.path.endswith(".jsonl")
        ]
        print(f"  Found {len(jsonl_files)} JSONL files")

        if not jsonl_files:
            return False

        # Download each file
        submissions_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        downloaded = 0
        for finfo in jsonl_files:
            try:
                local = hf_hub_download(
                    repo_id="gaia-benchmark/submissions_public",
                    filename=finfo.path,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    local_dir=str(submissions_dir),
                )
                downloaded += 1
            except Exception as e:
                print(f"  Failed to download {finfo.path}: {e}")
                break  # Likely gated access error

        print(f"  Downloaded {downloaded}/{len(jsonl_files)} files")
        return downloaded > 0

    except Exception as e:
        print(f"WARNING: Cannot access submissions_public: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4: Build HAL per-item response matrix
# ---------------------------------------------------------------------------
def build_hal_response_matrix():
    """Build per-item response matrix from HAL heatmap data.

    Returns:
        response_matrix: DataFrame (agents x tasks) with continuous values
        binary_matrix: DataFrame (agents x tasks) with 0/1 values
    """
    heatmap_path = RAW_DIR / "hal_gaia_heatmap.json"
    if not heatmap_path.exists():
        print("ERROR: HAL heatmap data not found", file=sys.stderr)
        return None, None

    with open(heatmap_path) as f:
        data = json.load(f)

    # Find the heatmap trace
    heatmap_trace = None
    for trace in data["data"]:
        if trace.get("type") == "heatmap" and "z" in trace:
            heatmap_trace = trace
            break

    if heatmap_trace is None:
        print("ERROR: No heatmap trace found in data", file=sys.stderr)
        return None, None

    task_ids = heatmap_trace["x"]
    agent_names = heatmap_trace["y"]
    z_values = heatmap_trace["z"]

    # Remove the summary row (last row is typically "Tasks Solved: X/165")
    clean_agents = []
    clean_z = []
    for name, row in zip(agent_names, z_values):
        if "Tasks Solved" in name or "<b>" in name:
            continue
        clean_agents.append(name)
        clean_z.append(row)

    print(f"  HAL heatmap: {len(clean_agents)} agents x {len(task_ids)} tasks")

    # Build DataFrame
    matrix = np.array(clean_z, dtype=float)
    response_df = pd.DataFrame(matrix, index=clean_agents, columns=task_ids)
    response_df.index.name = "agent"

    # Binarize: >= 0.5 means correct (handles multi-run fraction)
    binary_df = (response_df >= 0.5).astype(int)

    return response_df, binary_df


# ---------------------------------------------------------------------------
# Step 5: Build HF submissions response matrix (if accessible)
# ---------------------------------------------------------------------------
def build_submissions_response_matrix():
    """Build per-item response matrix from HF submissions_public JSONL files.

    Each JSONL file has lines with:
      {"id": "task_id", "model_answer": "...", "score": true/false, "level": N}

    Returns DataFrame (models x tasks) with binary 0/1 values, or None.
    """
    submissions_dir = RAW_DIR / "submissions_public"
    if not submissions_dir.exists():
        return None

    jsonl_files = list(submissions_dir.rglob("*.jsonl"))
    if not jsonl_files:
        return None

    print(f"  Loading {len(jsonl_files)} JSONL submission files...")

    model_results = {}
    all_task_ids = set()

    for fpath in sorted(jsonl_files):
        # Derive model name from directory structure
        rel_path = fpath.relative_to(submissions_dir)
        parts = list(rel_path.parts)

        # Name format: org/model/timestamp.jsonl or org/timestamp.jsonl
        if len(parts) >= 3:
            model_name = f"{parts[0]}/{parts[1]}"
        elif len(parts) >= 2:
            model_name = parts[0]
        else:
            model_name = fpath.stem

        # Skip test/debug submissions
        skip_names = {"bla", "blab", "test", "TEST", "aa", "ea", "plop"}
        if parts[0] in skip_names:
            continue

        # If multiple timestamps for same model, keep the latest
        if model_name in model_results:
            # Compare file modification times
            existing_time = model_results[model_name].get("_file_time", "")
            current_time = fpath.stem
            if current_time <= existing_time:
                continue

        scores = {}
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    tid = entry.get("id", entry.get("task_id", ""))
                    score = entry.get("score", False)
                    if tid:
                        scores[tid] = 1 if score else 0
                        all_task_ids.add(tid)
        except (json.JSONDecodeError, IOError) as e:
            print(f"    WARNING: Error reading {fpath}: {e}", file=sys.stderr)
            continue

        if scores:
            scores["_file_time"] = fpath.stem
            model_results[model_name] = scores

    if not model_results:
        return None

    # Build matrix
    task_ids = sorted(all_task_ids)
    rows = {}
    for model_name, scores in sorted(model_results.items()):
        rows[model_name] = [
            scores.get(tid, np.nan) for tid in task_ids
        ]

    df = pd.DataFrame.from_dict(rows, orient="index", columns=task_ids)
    df.index.name = "model"
    print(f"  Submissions matrix: {df.shape[0]} models x {df.shape[1]} tasks")
    return df


# ---------------------------------------------------------------------------
# Step 6: Build model summary from HF aggregate data
# ---------------------------------------------------------------------------
def build_hf_aggregate_summary():
    """Build summary from HuggingFace results_public aggregate scores."""
    test_path = RAW_DIR / "hf_results_test.parquet"
    val_path = RAW_DIR / "hf_results_validation.parquet"

    summaries = {}

    for split, path in [("test", test_path), ("validation", val_path)]:
        if not path.exists():
            continue

        df = pd.read_parquet(path)

        # Deduplicate: keep latest submission per model
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False).drop_duplicates(
            subset=["model"], keep="first"
        )
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        summaries[split] = df

    return summaries


# ---------------------------------------------------------------------------
# Printing utilities
# ---------------------------------------------------------------------------
def print_matrix_statistics(response_df, binary_df, label):
    """Print comprehensive statistics for a response matrix."""
    n_agents, n_tasks = binary_df.shape
    total_cells = n_agents * n_tasks
    n_evaluated = binary_df.notna().sum().sum()
    n_correct = int(binary_df.sum().sum())
    n_incorrect = int(n_evaluated) - n_correct

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Agents/Models (rows):  {n_agents}")
    print(f"  Tasks (columns):       {n_tasks}")
    print(f"  Matrix dimensions:     {n_agents} x {n_tasks}")
    print(f"  Total cells:           {total_cells:,}")
    print(f"  Evaluated cells:       {int(n_evaluated):,} "
          f"({n_evaluated / total_cells * 100:.1f}%)")
    print(f"  Correct cells:         {n_correct:,} "
          f"({n_correct / n_evaluated * 100:.1f}%)")
    print(f"  Incorrect cells:       {n_incorrect:,} "
          f"({n_incorrect / n_evaluated * 100:.1f}%)")

    # Per-agent statistics
    per_agent = binary_df.mean(axis=1)
    best = per_agent.idxmax()
    worst = per_agent.idxmin()
    print(f"\n  Per-agent accuracy:")
    print(f"    Best:    {per_agent.max() * 100:.1f}% ({best})")
    print(f"    Worst:   {per_agent.min() * 100:.1f}% ({worst})")
    print(f"    Median:  {per_agent.median() * 100:.1f}%")
    print(f"    Mean:    {per_agent.mean() * 100:.1f}%")
    print(f"    Std:     {per_agent.std() * 100:.1f}%")

    # Per-task statistics
    per_task = binary_df.mean(axis=0)
    print(f"\n  Per-task solve rate (across agents):")
    print(f"    Min:     {per_task.min() * 100:.1f}%")
    print(f"    Max:     {per_task.max() * 100:.1f}%")
    print(f"    Median:  {per_task.median() * 100:.1f}%")
    print(f"    Mean:    {per_task.mean() * 100:.1f}%")
    print(f"    Std:     {per_task.std() * 100:.1f}%")

    # Task difficulty distribution
    unsolved = (per_task == 0).sum()
    hard = ((per_task > 0) & (per_task <= 0.2)).sum()
    medium = ((per_task > 0.2) & (per_task <= 0.8)).sum()
    easy = (per_task > 0.8).sum()
    solved_all = (per_task == 1.0).sum()
    print(f"\n  Task difficulty distribution:")
    print(f"    Unsolved (0%):       {unsolved}")
    print(f"    Hard (0-20%]:        {hard}")
    print(f"    Medium (20-80%]:     {medium}")
    print(f"    Easy (>80%):         {easy}")
    print(f"    Solved by ALL:       {solved_all}")

    # Top 10 agents
    print(f"\n  Top 10 agents:")
    print(f"  {'Rank':<5} {'Agent':<60} {'Accuracy':>8}")
    print(f"  {'-' * 5} {'-' * 60} {'-' * 8}")
    for i, (agent, acc) in enumerate(
        per_agent.sort_values(ascending=False).head(10).items()
    ):
        print(f"  {i + 1:<5} {agent[:60]:<60} {acc * 100:>7.1f}%")

    if response_df is not None and not response_df.equals(binary_df):
        # Continuous value statistics (for HAL heatmap fractions)
        all_vals = response_df.values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        print(f"\n  Continuous (fraction-of-runs) statistics:")
        print(f"    Mean:    {all_vals.mean():.4f}")
        print(f"    Median:  {np.median(all_vals):.4f}")
        print(f"    Std:     {all_vals.std():.4f}")
        n_zero = (all_vals == 0).sum()
        n_one = (all_vals == 1).sum()
        n_partial = ((all_vals > 0) & (all_vals < 1)).sum()
        print(f"    Exact 0: {n_zero:,} ({n_zero / len(all_vals) * 100:.1f}%)")
        print(f"    Exact 1: {n_one:,} ({n_one / len(all_vals) * 100:.1f}%)")
        print(f"    Partial: {n_partial:,} "
              f"({n_partial / len(all_vals) * 100:.1f}%)")


def print_hf_aggregate_statistics(summaries):
    """Print statistics for HuggingFace aggregate leaderboard data."""
    for split, df in summaries.items():
        print(f"\n{'=' * 70}")
        print(f"  HF RESULTS_PUBLIC: {split.upper()} SPLIT (Aggregate Scores)")
        print(f"{'=' * 70}")
        print(f"  Total unique models: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} to "
              f"{df['date'].max().date()}")

        print(f"\n  Score distribution:")
        for col, label in [
            ("score", "Overall"),
            ("score_level1", "Level 1"),
            ("score_level2", "Level 2"),
            ("score_level3", "Level 3"),
        ]:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"    {label:10s}: mean={vals.mean() * 100:.1f}%  "
                      f"median={vals.median() * 100:.1f}%  "
                      f"max={vals.max() * 100:.1f}%  "
                      f"min={vals.min() * 100:.1f}%")

        # Top 10
        print(f"\n  Top 10 models (by overall score):")
        print(f"  {'Rank':<5} {'Model':<50} {'Score':>7} "
              f"{'L1':>7} {'L2':>7} {'L3':>7}")
        print(f"  {'-' * 5} {'-' * 50} {'-' * 7} "
              f"{'-' * 7} {'-' * 7} {'-' * 7}")
        for i, (_, row) in enumerate(df.head(10).iterrows()):
            print(
                f"  {i + 1:<5} {str(row['model'])[:50]:<50} "
                f"{row['score'] * 100:>6.1f}% "
                f"{row['score_level1'] * 100:>6.1f}% "
                f"{row['score_level2'] * 100:>6.1f}% "
                f"{row['score_level3'] * 100:>6.1f}%"
            )

        # Organisation breakdown
        if "organisation" in df.columns:
            orgs = df["organisation"].value_counts()
            orgs_with_entries = orgs[orgs.index != ""]
            print(f"\n  Top organizations (by submission count):")
            for org, count in orgs_with_entries.head(10).items():
                best = df[df["organisation"] == org]["score"].max()
                print(f"    {org:30s}: {count:3d} submissions, "
                      f"best={best * 100:.1f}%")


# ---------------------------------------------------------------------------
# Step 7: Write data source report
# ---------------------------------------------------------------------------
def write_data_source_report(
    hal_shape, hf_summaries, submissions_available, hf_submissions_shape
):
    """Write a comprehensive report about data sources and access status."""
    report_path = PROCESSED_DIR / "data_source_report.txt"
    lines = []
    lines.append("GAIA Benchmark - Data Source Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("GAIA (General AI Assistants) evaluates AI assistants on 450+")
    lines.append("multi-step reasoning problems. Three difficulty levels: L1")
    lines.append("(easy), L2 (medium), L3 (hard). 165 validation + 300 test.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SOURCE 1: HAL Leaderboard (per-item, validation set)")
    lines.append("-" * 70)
    lines.append("URL: https://hal.cs.princeton.edu/gaia")
    lines.append("Access: PUBLIC (scraped from embedded Plotly chart data)")
    lines.append("Content: Per-task success rates for 32 agent configurations")
    lines.append("         on 165 GAIA validation problems.")
    lines.append("Format: Heatmap with fraction-of-runs (0.0-1.0) per cell.")
    lines.append("        Binarized at 0.5 threshold for response matrix.")
    if hal_shape:
        lines.append(f"Matrix: {hal_shape[0]} agents x {hal_shape[1]} tasks")
    lines.append("Status: DOWNLOADED AND PROCESSED")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SOURCE 2: HuggingFace results_public (aggregate scores)")
    lines.append("-" * 70)
    lines.append("URL: https://huggingface.co/datasets/gaia-benchmark/results_public")
    lines.append("Access: PUBLIC")
    lines.append("Content: Per-model aggregate scores (overall, L1, L2, L3)")
    lines.append("         for both test and validation splits.")
    lines.append("Format: Parquet with columns: model, model_family, score,")
    lines.append("         score_level1, score_level2, score_level3, date, etc.")
    if "test" in hf_summaries:
        lines.append(
            f"Test split: {len(hf_summaries['test'])} unique models"
        )
    if "validation" in hf_summaries:
        lines.append(
            f"Validation split: {len(hf_summaries['validation'])} unique models"
        )
    lines.append("NOTE: Only aggregate scores, NOT per-item results.")
    lines.append("Status: DOWNLOADED AND PROCESSED")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SOURCE 3: HuggingFace submissions_public (per-item, GATED)")
    lines.append("-" * 70)
    lines.append(
        "URL: https://huggingface.co/datasets/gaia-benchmark/submissions_public"
    )
    lines.append("Access: GATED (requires manual approval)")
    lines.append("Content: 133 JSONL files with per-task scored results for")
    lines.append("         validation set. ~50+ unique agent/model combos.")
    lines.append("Format: JSONL with {id, model_answer, score, level} per line.")
    if submissions_available and hf_submissions_shape:
        lines.append(
            f"Matrix: {hf_submissions_shape[0]} models x "
            f"{hf_submissions_shape[1]} tasks"
        )
        lines.append("Status: DOWNLOADED AND PROCESSED")
    else:
        lines.append("Status: ACCESS DENIED - Request access at URL above")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SOURCE 4: HuggingFace GAIA dataset (questions, GATED)")
    lines.append("-" * 70)
    lines.append("URL: https://huggingface.co/datasets/gaia-benchmark/GAIA")
    lines.append("Access: GATED (requires manual approval)")
    lines.append("Content: Task questions, difficulty levels, and answers.")
    lines.append("Format: Parquet/JSONL with task_id, Question, Level, etc.")
    lines.append("Status: ACCESS DENIED - Request access at URL above")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SOURCE 5: HAL Traces (encrypted, HuggingFace)")
    lines.append("-" * 70)
    lines.append("URL: https://huggingface.co/datasets/agent-evals/hal_traces")
    lines.append("Access: PUBLIC but encrypted")
    lines.append("Content: 37 GAIA evaluation traces (ZIP files, encrypted).")
    lines.append("Format: Encrypted JSON in ZIP archives.")
    lines.append("Status: NOT USED (requires decryption key from HAL team)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RECOMMENDATION FOR COMPLETE PER-ITEM DATA")
    lines.append("-" * 70)
    lines.append("To get the most comprehensive per-item response matrix:")
    lines.append("1. Request access to gaia-benchmark/submissions_public on HF")
    lines.append("2. Set HF_TOKEN env var with an authorized token")
    lines.append("3. Re-run this script")
    lines.append("This will provide ~50+ models x 165 tasks per-item data.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved data source report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GAIA Response Matrix Builder")
    print("=" * 70)

    # Step 1: Download data
    print("\n--- Step 1: Download HAL leaderboard data ---")
    download_hal_heatmap()

    print("\n--- Step 2: Download HF results_public ---")
    download_hf_results()

    print("\n--- Step 3: Attempt HF submissions_public download ---")
    submissions_ok = try_download_submissions()

    # Step 2: Build response matrices
    print("\n--- Step 4: Build HAL per-item response matrix ---")
    hal_response, hal_binary = build_hal_response_matrix()

    hf_submissions_df = None
    hf_submissions_shape = None
    if submissions_ok:
        print("\n--- Step 5: Build HF submissions response matrix ---")
        hf_submissions_df = build_submissions_response_matrix()
        if hf_submissions_df is not None:
            hf_submissions_shape = hf_submissions_df.shape

    print("\n--- Step 6: Build HF aggregate summary ---")
    hf_summaries = build_hf_aggregate_summary()

    # Step 3: Save outputs
    print("\n--- Step 7: Save outputs ---")

    # HAL response matrix (primary per-item output)
    if hal_binary is not None:
        # Save continuous (fraction of runs)
        cont_path = PROCESSED_DIR / "response_matrix_hal_continuous.csv"
        hal_response.to_csv(cont_path)
        print(f"Saved HAL continuous matrix: {cont_path}")

        # Save binary (primary)
        bin_path = PROCESSED_DIR / "response_matrix_hal.csv"
        hal_binary.to_csv(bin_path)
        print(f"Saved HAL binary matrix: {bin_path}")

    # HF submissions response matrix (if available)
    if hf_submissions_df is not None:
        sub_path = PROCESSED_DIR / "response_matrix_hf_submissions.csv"
        hf_submissions_df.to_csv(sub_path)
        print(f"Saved HF submissions matrix: {sub_path}")

    # HF aggregate scores (for test and validation)
    for split, df in hf_summaries.items():
        agg_path = PROCESSED_DIR / f"hf_leaderboard_{split}.csv"
        df.to_csv(agg_path, index=False)
        print(f"Saved HF {split} leaderboard: {agg_path}")

    # Model summary combining all sources
    summary_rows = []

    # From HAL
    if hal_binary is not None:
        for agent in hal_binary.index:
            row = {
                "agent": agent,
                "source": "HAL",
                "accuracy": hal_binary.loc[agent].mean(),
                "n_tasks_evaluated": int(hal_binary.loc[agent].notna().sum()),
                "n_correct": int(hal_binary.loc[agent].sum()),
            }
            summary_rows.append(row)

    # From HF aggregate
    if "validation" in hf_summaries:
        for _, hf_row in hf_summaries["validation"].iterrows():
            summary_rows.append({
                "agent": str(hf_row["model"]),
                "source": "HF-validation",
                "accuracy": hf_row["score"],
                "score_level1": hf_row["score_level1"],
                "score_level2": hf_row["score_level2"],
                "score_level3": hf_row["score_level3"],
                "organisation": hf_row.get("organisation", ""),
                "model_family": hf_row.get("model_family", ""),
                "date": str(hf_row.get("date", "")),
            })

    if "test" in hf_summaries:
        for _, hf_row in hf_summaries["test"].iterrows():
            summary_rows.append({
                "agent": str(hf_row["model"]),
                "source": "HF-test",
                "accuracy": hf_row["score"],
                "score_level1": hf_row["score_level1"],
                "score_level2": hf_row["score_level2"],
                "score_level3": hf_row["score_level3"],
                "organisation": hf_row.get("organisation", ""),
                "model_family": hf_row.get("model_family", ""),
                "date": str(hf_row.get("date", "")),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model summary: {summary_path} ({len(summary_df)} entries)")

    # Step 4: Print statistics
    print("\n--- Step 8: Statistics ---")

    if hal_binary is not None:
        print_matrix_statistics(
            hal_response, hal_binary,
            "HAL PER-ITEM RESPONSE MATRIX (Primary)"
        )

    if hf_submissions_df is not None:
        print_matrix_statistics(
            hf_submissions_df, hf_submissions_df,
            "HF SUBMISSIONS PER-ITEM RESPONSE MATRIX"
        )

    print_hf_aggregate_statistics(hf_summaries)

    # Write report
    hal_shape = hal_binary.shape if hal_binary is not None else None
    write_data_source_report(
        hal_shape, hf_summaries, submissions_ok, hf_submissions_shape
    )

    # Final file listing
    print(f"\n{'=' * 70}")
    print(f"  OUTPUT FILES")
    print(f"{'=' * 70}")
    for f in sorted(PROCESSED_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s}  {size_kb:>8.1f} KB")

    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")
    if hal_binary is not None:
        print(f"  PRIMARY response matrix (HAL per-item):")
        print(f"    Dimensions: {hal_binary.shape[0]} agents x "
              f"{hal_binary.shape[1]} tasks")
        fill = hal_binary.notna().sum().sum() / (
            hal_binary.shape[0] * hal_binary.shape[1]
        )
        print(f"    Fill rate:  {fill * 100:.1f}%")
        print(f"    Mean accuracy: {hal_binary.values.mean() * 100:.1f}%")
    if "test" in hf_summaries:
        print(f"\n  HF test leaderboard:")
        print(f"    Models: {len(hf_summaries['test'])}")
        print(f"    Top score: "
              f"{hf_summaries['test']['score'].max() * 100:.1f}%")
    if submissions_ok and hf_submissions_df is not None:
        print(f"\n  HF submissions per-item matrix:")
        print(f"    Dimensions: {hf_submissions_df.shape[0]} models x "
              f"{hf_submissions_df.shape[1]} tasks")
    elif not submissions_ok:
        print(f"\n  HF submissions_public: ACCESS DENIED (gated dataset)")
        print(f"    Request access at: https://huggingface.co/datasets/"
              f"gaia-benchmark/submissions_public")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
