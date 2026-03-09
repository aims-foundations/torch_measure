"""
Build WorkArena / WorkArena++ response matrices from multiple public data sources.

Data sources:
  1. BrowserGym Leaderboard (HuggingFace Space: ServiceNow/browsergym-leaderboard)
     - Aggregate scores per agent for WorkArena-L1, L2, L3
     - ~16 agents evaluated (GPT-4o, GPT-5, Claude-3.5-Sonnet, Llama, etc.)
     - Format: single JSON per agent per benchmark with score + std_err

  2. WorkArena paper (ICML 2024, arXiv:2403.07718) — Table 2
     - Per-category success rates for 4 models on L1 tasks
     - 7 categories: Dashboard, Form, Knowledge, List-filter, List-sort, Menu, Service Catalog
     - Models: GPT-4o, GPT-4o-V, GPT-3.5, Llama3-70B

  3. AgentRewardBench (HuggingFace Dataset: McGill-NLP/agent-reward-bench)
     - Per-task functional evaluation results for 4 agents on WorkArena L1+L2 tasks
     - Agents: GPT-4o-2024-11-20, Claude-3.7-Sonnet, Llama-3.3-70B, Qwen2.5-VL-72B
     - Binary success/failure per task instance (cum_reward in summary_info)

Outputs:
  - response_matrix_leaderboard.csv: Agents x Benchmarks (L1/L2/L3) aggregate scores
  - response_matrix_paper_l1_categories.csv: Models x L1 task categories
  - response_matrix_agentrewardbench.csv: Agents x Tasks binary matrix (from ARB)
  - task_metadata.csv: All 33 L1 tasks with instance counts and categories
  - data_availability_report.txt: Summary of what data exists and limitations

Note on per-item data availability:
  WorkArena is a *live environment* benchmark — agents interact with a running ServiceNow
  instance in real time. Unlike static benchmarks, there is no fixed per-item result table
  published by the authors. The BrowserGym leaderboard only stores aggregate scores.
  The most granular public data comes from AgentRewardBench, which provides per-task
  functional evaluation results for 4 agents on a subset of WorkArena tasks.
"""

import os
import json
import sys
import time
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HF_LEADERBOARD_BASE = (
    "https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard/raw/main/results"
)
HF_ARB_BASE = (
    "https://huggingface.co/datasets/McGill-NLP/agent-reward-bench/raw/main"
)

# ──────────────────────────────────────────────────────────────────────
# Source 1: BrowserGym Leaderboard — aggregate scores
# ──────────────────────────────────────────────────────────────────────
LEADERBOARD_AGENTS = [
    "GenericAgent-AgentTrek-1.0-32b",
    "GenericAgent-Claude-3.5-Sonnet",
    "GenericAgent-Claude-3.7-Sonnet",
    "GenericAgent-Claude-4-Sonnet",
    "GenericAgent-GPT-4.1-Mini",
    "GenericAgent-GPT-4o-mini",
    "GenericAgent-GPT-4o",
    "GenericAgent-GPT-5-mini",
    "GenericAgent-GPT-5-nano",
    "GenericAgent-GPT-5",
    "GenericAgent-GPT-o1-mini",
    "GenericAgent-GPT-oss-120b",
    "GenericAgent-GPT-oss-20b",
    "GenericAgent-Llama-3.1-405b",
    "GenericAgent-Llama-3.1-70b",
    "GenericAgent-o3-mini",
    "OrbyAgent-ActIO-72b",
    "OrbyAgent-Claude-3.5-Sonnet",
]

LEADERBOARD_BENCHMARKS = ["workarena-l1", "workarena-l2", "workarena-l3"]


def fetch_leaderboard_data():
    """Download aggregate WorkArena scores from BrowserGym leaderboard."""
    print("\n[Source 1] Fetching BrowserGym Leaderboard data...")
    out_dir = RAW_DIR / "leaderboard"
    out_dir.mkdir(exist_ok=True)

    results = []
    for agent in LEADERBOARD_AGENTS:
        for bm in LEADERBOARD_BENCHMARKS:
            url = f"{HF_LEADERBOARD_BASE}/{agent}/{bm}.json"
            out_file = out_dir / f"{agent}__{bm}.json"

            if out_file.exists():
                with open(out_file) as f:
                    data = json.load(f)
                results.append(data[0])
                continue

            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    with open(out_file, "w") as f:
                        json.dump(data, f, indent=2)
                    results.append(data[0])
                    print(f"  OK: {agent} / {bm} -> score={data[0]['score']}")
                else:
                    print(f"  MISS: {agent} / {bm} -> HTTP {r.status_code}")
            except Exception as e:
                print(f"  ERR: {agent} / {bm} -> {e}")
            time.sleep(0.1)

    print(f"  Total results fetched: {len(results)}")
    return results


def build_leaderboard_matrix(results):
    """Build agents x benchmarks matrix from leaderboard data."""
    rows = {}
    for r in results:
        agent = r["agent_name"]
        bm = r["benchmark"]
        score = r["score"]
        std_err = r.get("std_err", np.nan)
        if agent not in rows:
            rows[agent] = {}
        rows[agent][bm] = score
        rows[agent][f"{bm}_std_err"] = std_err

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Agent"

    # Reorder columns
    col_order = []
    for bm in ["WorkArena-L1", "WorkArena-L2", "WorkArena-L3"]:
        if bm in df.columns:
            col_order.append(bm)
        se = f"{bm}_std_err"
        if se in df.columns:
            col_order.append(se)
    remaining = [c for c in df.columns if c not in col_order]
    df = df[col_order + remaining]

    df = df.sort_values("WorkArena-L1", ascending=False, na_position="last")

    out_path = PROCESSED_DIR / "response_matrix_leaderboard.csv"
    df.to_csv(out_path)
    return df


# ──────────────────────────────────────────────────────────────────────
# Source 2: WorkArena Paper — per-category L1 results (Table 2)
# ──────────────────────────────────────────────────────────────────────
PAPER_L1_CATEGORIES = {
    # category: (n_tasks, {model: (score, std_err)})
    "Dashboard": (4, {
        "GPT-4o": (62.5, 6.8),
        "GPT-4o-V": (72.5, 6.0),
        "GPT-3.5": (20.0, 4.8),
        "Llama3-70B": (37.5, 6.0),
    }),
    "Form": (5, {
        "GPT-4o": (40.0, 5.9),
        "GPT-4o-V": (34.0, 4.8),
        "GPT-3.5": (2.0, 2.5),
        "Llama3-70B": (32.0, 4.6),
    }),
    "Knowledge": (1, {
        "GPT-4o": (80.0, 12.2),
        "GPT-4o-V": (70.0, 13.9),
        "GPT-3.5": (0.0, 4.3),
        "Llama3-70B": (30.0, 12.3),
    }),
    "List-filter": (6, {
        "GPT-4o": (0.0, 1.6),
        "GPT-4o-V": (0.0, 1.7),
        "GPT-3.5": (0.0, 1.6),
        "Llama3-70B": (0.0, 1.8),
    }),
    "List-sort": (6, {
        "GPT-4o": (10.0, 3.8),
        "GPT-4o-V": (13.3, 4.0),
        "GPT-3.5": (8.3, 3.7),
        "Llama3-70B": (1.7, 2.5),
    }),
    "Menu": (2, {
        "GPT-4o": (60.0, 8.0),
        "GPT-4o-V": (90.0, 6.0),
        "GPT-3.5": (5.0, 4.7),
        "Llama3-70B": (0.0, 2.9),
    }),
    "Service Catalog": (9, {
        "GPT-4o": (77.8, 3.2),
        "GPT-4o-V": (65.6, 3.6),
        "GPT-3.5": (5.6, 2.3),
        "Llama3-70B": (26.7, 3.4),
    }),
}

# Overall scores from paper
PAPER_L1_OVERALL = {
    "GPT-4o": (42.7, 1.5),
    "GPT-4o-V": (41.8, 1.7),
    "GPT-3.5": (6.1, 1.3),
    "Llama3-70B": (17.9, 1.5),
}

# All 33 L1 tasks from Table 6 of the paper
L1_TASK_METADATA = [
    # (category, task_name, n_instances, oracle_actions_mean, oracle_actions_std)
    ("List-filter", "FilterAssetList", 1000, 17.3, 6.5),
    ("List-filter", "FilterChangeRequestList", 1000, 18.7, 4.7),
    ("List-filter", "FilterHardwareList", 1000, 18.4, 5.6),
    ("List-filter", "FilterIncidentList", 1000, 16.2, 4.2),
    ("List-filter", "FilterServiceCatalogItemList", 1000, 19.9, 5.9),
    ("List-filter", "FilterUserList", 1000, 12.7, 3.2),
    ("List-sort", "SortAssetList", 150, 7.4, 2.3),
    ("List-sort", "SortChangeRequestList", 150, 7.7, 1.6),
    ("List-sort", "SortHardwareList", 150, 8.0, 2.3),
    ("List-sort", "SortIncidentList", 150, 8.0, 2.7),
    ("List-sort", "SortServiceCatalogItemList", 150, 8.3, 2.5),
    ("List-sort", "SortUserList", 150, 7.7, 2.1),
    ("Form", "CreateChangeRequest", 1000, 21.5, 6.2),
    ("Form", "CreateIncident", 1000, 23.0, 7.9),
    ("Form", "CreateHardwareAsset", 1000, 47.1, 10.9),
    ("Form", "CreateProblem", 1000, 10.0, 3.4),
    ("Form", "CreateUser", 1000, 17.9, 5.2),
    ("Knowledge", "KnowledgeBaseSearch", 1000, 4.0, 0.0),
    ("Service Catalog", "OrderDeveloperLaptopMac", 1000, 8.7, 0.9),
    ("Service Catalog", "OrderIpadMini", 80, 6.0, 0.0),
    ("Service Catalog", "OrderIpadPro", 60, 6.0, 0.0),
    ("Service Catalog", "OrderSalesLaptop", 1000, 9.0, 0.8),
    ("Service Catalog", "OrderStandardLaptop", 1000, 8.0, 0.6),
    ("Service Catalog", "OrderAppleWatch", 10, 4.0, 0.0),
    ("Service Catalog", "OrderAppleMacBookPro15", 10, 4.0, 0.0),
    ("Service Catalog", "OrderDevelopmentLaptopPC", 40, 6.0, 0.0),
    ("Service Catalog", "OrderLoanerLaptop", 350, 8.0, 0.0),
    ("Menu", "AllMenu", 1000, 3.0, 0.0),
    ("Menu", "Impersonation", 600, 7.0, 0.0),
    ("Dashboard", "SingleChartValueRetrieval", 1000, 1.0, 0.0),
    ("Dashboard", "SingleChartMinMaxRetrieval", 346, 1.0, 0.0),
    ("Dashboard", "MultiChartValueRetrieval", 444, 2.0, 0.0),
    ("Dashboard", "MultiChartMinMaxRetrieval", 72, 2.0, 0.0),
]


def build_paper_data():
    """Save paper data to raw/ and build per-category matrix."""
    print("\n[Source 2] Building WorkArena paper (ICML 2024) data...")

    # Save raw paper data
    paper_raw = {
        "source": "WorkArena: How Capable Are Web Agents at Solving Common Knowledge "
                  "Work Tasks? (ICML 2024, arXiv:2403.07718)",
        "table": "Table 2 — Success Rates",
        "categories": {},
        "overall": PAPER_L1_OVERALL,
        "task_metadata": [
            {
                "category": t[0], "task": t[1], "n_instances": t[2],
                "oracle_actions_mean": t[3], "oracle_actions_std": t[4],
            }
            for t in L1_TASK_METADATA
        ],
    }
    for cat, (n_tasks, scores) in PAPER_L1_CATEGORIES.items():
        paper_raw["categories"][cat] = {
            "n_tasks": n_tasks,
            "scores": {m: {"score": s, "std_err": se} for m, (s, se) in scores.items()},
        }

    raw_path = RAW_DIR / "paper_l1_results.json"
    with open(raw_path, "w") as f:
        json.dump(paper_raw, f, indent=2)

    # Build per-category response matrix (models x categories)
    models = ["GPT-4o", "GPT-4o-V", "GPT-3.5", "Llama3-70B"]
    categories = list(PAPER_L1_CATEGORIES.keys())

    data = {}
    for model in models:
        row = {}
        for cat in categories:
            row[cat] = PAPER_L1_CATEGORIES[cat][1][model][0]
        row["Overall"] = PAPER_L1_OVERALL[model][0]
        data[model] = row

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Model"

    out_path = PROCESSED_DIR / "response_matrix_paper_l1_categories.csv"
    df.to_csv(out_path)

    # Save task metadata
    meta_df = pd.DataFrame(L1_TASK_METADATA, columns=[
        "category", "task_name", "n_instances",
        "oracle_actions_mean", "oracle_actions_std",
    ])
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    return df, meta_df


# ──────────────────────────────────────────────────────────────────────
# Source 3: AgentRewardBench — per-task functional results
# ──────────────────────────────────────────────────────────────────────
ARB_AGENTS = [
    "GenericAgent-gpt-4o-2024-11-20",
    "GenericAgent-anthropic_claude-3.7-sonnet",
    "GenericAgent-meta-llama_Llama-3.3-70B-Instruct",
    "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct",
]


def fetch_arb_task_list():
    """Fetch the list of WorkArena tasks from AgentRewardBench functional results."""
    print("\n[Source 3] Fetching AgentRewardBench task list...")

    # We use the HF API to list files in the dataset
    api_url = (
        "https://huggingface.co/api/datasets/McGill-NLP/agent-reward-bench/tree/main"
        "/judgments/workarena/GenericAgent-gpt-4o-2024-11-20/functional"
    )

    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            files = r.json()
            task_files = [
                f["path"].split("/")[-1]
                for f in files
                if f["path"].endswith(".json")
            ]
            print(f"  Found {len(task_files)} task files for GPT-4o reference agent")
            return task_files
        else:
            print(f"  API returned {r.status_code}, trying known task list")
            return None
    except Exception as e:
        print(f"  Error fetching task list: {e}")
        return None


def fetch_arb_data(task_files):
    """Download per-task functional results from AgentRewardBench."""
    print("\n  Downloading per-task results...")
    out_dir = RAW_DIR / "agentrewardbench"
    out_dir.mkdir(exist_ok=True)

    all_results = []
    for agent in ARB_AGENTS:
        agent_dir = out_dir / agent
        agent_dir.mkdir(exist_ok=True)

        fetched = 0
        skipped = 0
        errors = 0

        for task_file in task_files:
            out_file = agent_dir / task_file
            task_name = task_file.replace(".json", "")

            if out_file.exists():
                try:
                    with open(out_file) as f:
                        data = json.load(f)
                    # Extract success info
                    reward = data.get("trajectory_info", {}).get(
                        "summary_info", {}
                    ).get("cum_reward", None)
                    all_results.append({
                        "agent": agent,
                        "task": task_name,
                        "reward": reward,
                    })
                    skipped += 1
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass  # re-fetch

            url = (
                f"{HF_ARB_BASE}/judgments/workarena/{agent}/functional/{task_file}"
            )
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    with open(out_file, "w") as f:
                        json.dump(data, f, indent=2)
                    reward = data.get("trajectory_info", {}).get(
                        "summary_info", {}
                    ).get("cum_reward", None)
                    all_results.append({
                        "agent": agent,
                        "task": task_name,
                        "reward": reward,
                    })
                    fetched += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

            time.sleep(0.05)  # rate limiting

        print(f"  {agent}: fetched={fetched}, cached={skipped}, errors={errors}")

    return all_results


def build_arb_matrix(all_results):
    """Build agents x tasks binary response matrix from AgentRewardBench data."""
    if not all_results:
        print("  No AgentRewardBench results to process.")
        return None

    df = pd.DataFrame(all_results)

    # Pivot to agents x tasks
    matrix = df.pivot_table(
        index="agent", columns="task", values="reward", aggfunc="first"
    )
    matrix.index.name = "Agent"

    # Sort columns
    matrix = matrix[sorted(matrix.columns)]

    out_path = PROCESSED_DIR / "response_matrix_agentrewardbench.csv"
    matrix.to_csv(out_path)

    # Also save as the canonical response_matrix.csv (primary output)
    canonical_path = PROCESSED_DIR / "response_matrix.csv"
    matrix.to_csv(canonical_path)

    return matrix


# ──────────────────────────────────────────────────────────────────────
# Summary Statistics
# ──────────────────────────────────────────────────────────────────────
def print_matrix_stats(df, label):
    """Print comprehensive statistics for a response matrix."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    n_nan = int(df.isna().sum().sum())
    n_filled = total_cells - n_nan
    fill_rate = n_filled / total_cells if total_cells > 0 else 0

    print(f"  Rows (agents/models): {n_rows}")
    print(f"  Columns (tasks/cats): {n_cols}")
    print(f"  Matrix dimensions:    {n_rows} x {n_cols}")
    print(f"  Total cells:          {total_cells:,}")
    print(f"  Filled cells:         {n_filled:,} ({fill_rate*100:.1f}%)")
    print(f"  NaN cells:            {n_nan:,} ({(1-fill_rate)*100:.1f}%)")

    # Value statistics (ignoring NaN)
    vals = df.values.flatten()
    vals = vals[~np.isnan(vals.astype(float))]
    if len(vals) > 0:
        print(f"\n  Value statistics:")
        print(f"    Mean:   {np.mean(vals):.3f}")
        print(f"    Median: {np.median(vals):.3f}")
        print(f"    Std:    {np.std(vals):.3f}")
        print(f"    Min:    {np.min(vals):.3f}")
        print(f"    Max:    {np.max(vals):.3f}")

        # For binary data
        unique_vals = np.unique(vals)
        if set(unique_vals).issubset({0.0, 1.0}):
            n_success = int((vals == 1.0).sum())
            n_fail = int((vals == 0.0).sum())
            print(f"\n  Binary outcome distribution:")
            print(f"    Success (1): {n_success:,} ({n_success/len(vals)*100:.1f}%)")
            print(f"    Failure (0): {n_fail:,} ({n_fail/len(vals)*100:.1f}%)")

    # Per-row stats
    row_means = df.mean(axis=1)
    if len(row_means) > 0:
        print(f"\n  Per-row (agent/model) mean:")
        print(f"    Min:    {row_means.min():.3f} ({row_means.idxmin()})")
        print(f"    Max:    {row_means.max():.3f} ({row_means.idxmax()})")
        print(f"    Median: {row_means.median():.3f}")

    # Per-column stats
    col_means = df.mean(axis=0)
    if len(col_means) > 0:
        print(f"\n  Per-column (task/category) mean:")
        print(f"    Min:    {col_means.min():.3f} ({col_means.idxmin()})")
        print(f"    Max:    {col_means.max():.3f} ({col_means.idxmax()})")
        print(f"    Median: {col_means.median():.3f}")

        # Task difficulty distribution (for binary data)
        if set(unique_vals).issubset({0.0, 1.0}):
            unsolved = int((col_means == 0).sum())
            easy = int((col_means > 0.75).sum())
            hard = int((col_means < 0.25).sum())
            print(f"\n  Task difficulty distribution:")
            print(f"    Unsolved (0%):  {unsolved}")
            print(f"    Hard (<25%):    {hard}")
            print(f"    Easy (>75%):    {easy}")


def write_data_availability_report(
    leaderboard_df, paper_df, meta_df, arb_matrix
):
    """Write a comprehensive report on data availability and limitations."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("WorkArena / WorkArena++ Data Availability Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    report_lines.append("BENCHMARK OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append("WorkArena (L1): 33 tasks, 19,912 instances — atomic ServiceNow tasks")
    report_lines.append("WorkArena++ (L2): 682 compositional tasks — multi-step workflows")
    report_lines.append("WorkArena++ (L3): Hardest — ticket-like, context-rich tasks")
    report_lines.append("Papers: ICML 2024 (arXiv:2403.07718), NeurIPS 2024 (arXiv:2407.05291)")
    report_lines.append("GitHub: github.com/ServiceNow/WorkArena")
    report_lines.append("Leaderboard: huggingface.co/spaces/ServiceNow/browsergym-leaderboard")
    report_lines.append("")

    report_lines.append("DATA SOURCES COLLECTED")
    report_lines.append("-" * 40)
    report_lines.append("")

    report_lines.append("1. BrowserGym Leaderboard (aggregate scores)")
    if leaderboard_df is not None:
        report_lines.append(
            f"   - {leaderboard_df.shape[0]} agents x "
            f"{leaderboard_df.shape[1]} columns (L1/L2/L3 + std_err)"
        )
        report_lines.append(f"   - File: processed/response_matrix_leaderboard.csv")
        report_lines.append(f"   - Granularity: ONE score per agent per benchmark level")
        report_lines.append(f"   - Limitation: No per-task breakdown")
    report_lines.append("")

    report_lines.append("2. WorkArena Paper Table 2 (per-category L1 results)")
    if paper_df is not None:
        report_lines.append(
            f"   - {paper_df.shape[0]} models x "
            f"{paper_df.shape[1]} categories"
        )
        report_lines.append(f"   - File: processed/response_matrix_paper_l1_categories.csv")
        report_lines.append(f"   - Granularity: Per-category (7 categories for 33 tasks)")
        report_lines.append(f"   - Limitation: Only 4 models; category-level, not task-level")
    report_lines.append("")

    report_lines.append("3. AgentRewardBench (per-task functional results)")
    if arb_matrix is not None:
        report_lines.append(
            f"   - {arb_matrix.shape[0]} agents x "
            f"{arb_matrix.shape[1]} tasks"
        )
        report_lines.append(f"   - File: processed/response_matrix_agentrewardbench.csv")
        report_lines.append(f"   - Granularity: Binary success/failure per task instance")
        report_lines.append(f"   - Limitation: Only 4 agents; mix of L1 and L2 tasks")
    report_lines.append("")

    report_lines.append("4. Task Metadata (all 33 L1 tasks)")
    if meta_df is not None:
        report_lines.append(f"   - {len(meta_df)} tasks across 7 categories")
        report_lines.append(f"   - File: processed/task_metadata.csv")
        report_lines.append(f"   - Fields: category, task_name, n_instances, "
                            f"oracle_actions_mean/std")
    report_lines.append("")

    report_lines.append("WHY PER-ITEM DATA IS LIMITED")
    report_lines.append("-" * 40)
    report_lines.append(
        "WorkArena is a LIVE ENVIRONMENT benchmark. Agents interact with a running"
    )
    report_lines.append(
        "ServiceNow instance in real time (not a static dataset). Key implications:"
    )
    report_lines.append(
        "  - Each run uses randomly sampled task configurations from a pool"
    )
    report_lines.append(
        "  - The BrowserGym leaderboard stores only aggregate success rates"
    )
    report_lines.append(
        "  - Full per-instance trajectories are not published (too large)"
    )
    report_lines.append(
        "  - The AgentRewardBench dataset provides the most granular public data,"
    )
    report_lines.append(
        "    but only for 4 agents on a subset of tasks"
    )
    report_lines.append("")

    report_lines.append("OPTIONS FOR MORE GRANULAR DATA")
    report_lines.append("-" * 40)
    report_lines.append(
        "1. Run AgentLab yourself (github.com/ServiceNow/AgentLab) to generate"
    )
    report_lines.append(
        "   per-instance results for any agent on WorkArena tasks"
    )
    report_lines.append(
        "2. Contact ServiceNow Research to request per-run result logs"
    )
    report_lines.append(
        "3. Use the AgentRewardBench per-task data (best available public source)"
    )
    report_lines.append(
        "4. Extract category-level data from the paper and impute per-task within"
    )
    report_lines.append(
        "   categories (assuming uniform performance within category)"
    )
    report_lines.append("")

    report_path = PROCESSED_DIR / "data_availability_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Saved: {report_path}")
    return report_lines


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("WorkArena / WorkArena++ Response Matrix Builder")
    print("=" * 70)

    # ── Source 1: Leaderboard ──
    leaderboard_results = fetch_leaderboard_data()
    leaderboard_df = build_leaderboard_matrix(leaderboard_results)
    print_matrix_stats(leaderboard_df, "BrowserGym Leaderboard (Agents x Benchmarks)")

    # Print leaderboard table
    print("\n  Agent scores (L1 / L2 / L3):")
    for agent in leaderboard_df.index:
        l1 = leaderboard_df.loc[agent].get("WorkArena-L1", np.nan)
        l2 = leaderboard_df.loc[agent].get("WorkArena-L2", np.nan)
        l3 = leaderboard_df.loc[agent].get("WorkArena-L3", np.nan)
        l1s = f"{l1:.1f}%" if not np.isnan(l1) else "N/A"
        l2s = f"{l2:.1f}%" if not np.isnan(l2) else "N/A"
        l3s = f"{l3:.1f}%" if not np.isnan(l3) else "N/A"
        print(f"    {agent:45s}  L1={l1s:>7s}  L2={l2s:>7s}  L3={l3s:>7s}")

    # ── Source 2: Paper data ──
    paper_df, meta_df = build_paper_data()
    print_matrix_stats(
        paper_df, "WorkArena Paper Table 2 (Models x L1 Categories)"
    )

    # Print paper table
    print("\n  Per-category scores:")
    for model in paper_df.index:
        scores = "  ".join(
            f"{cat}={paper_df.loc[model, cat]:.1f}%"
            for cat in paper_df.columns
        )
        print(f"    {model:15s}: {scores}")

    # Print task metadata summary
    print(f"\n  Task metadata ({len(meta_df)} tasks across "
          f"{meta_df['category'].nunique()} categories):")
    for cat, grp in meta_df.groupby("category"):
        total_inst = grp["n_instances"].sum()
        print(f"    {cat:20s}: {len(grp)} tasks, {total_inst:,} instances")
    print(f"    {'TOTAL':20s}: {len(meta_df)} tasks, "
          f"{meta_df['n_instances'].sum():,} instances")

    # ── Source 3: AgentRewardBench ──
    task_files = fetch_arb_task_list()
    arb_matrix = None
    if task_files:
        arb_results = fetch_arb_data(task_files)
        arb_matrix = build_arb_matrix(arb_results)
        if arb_matrix is not None:
            print_matrix_stats(
                arb_matrix,
                "AgentRewardBench (Agents x Tasks, binary success/failure)",
            )

            # Show per-agent summary
            print("\n  Per-agent success rates:")
            for agent in arb_matrix.index:
                row = arb_matrix.loc[agent]
                n_tasks = row.notna().sum()
                n_success = int((row == 1.0).sum())
                rate = row.mean()
                print(f"    {agent:55s}: {n_success}/{n_tasks} "
                      f"({rate*100:.1f}%)")
    else:
        print("\n  Could not fetch AgentRewardBench task list. "
              "Skipping per-task matrix.")

    # ── Data availability report ──
    write_data_availability_report(leaderboard_df, paper_df, meta_df, arb_matrix)

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Output files in {PROCESSED_DIR}:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = PROCESSED_DIR / f
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:50s}  {size_kb:.1f} KB")

    print(f"\n  Raw data in {RAW_DIR}:")
    for root, dirs, files in os.walk(RAW_DIR):
        level = root.replace(str(RAW_DIR), "").count(os.sep)
        indent = "    " + "  " * level
        subdir = os.path.basename(root)
        if level == 0:
            subdir = "raw/"
        n_files = len(files)
        print(f"{indent}{subdir}/ ({n_files} files)")

    print(f"\n  PRIMARY response matrix: response_matrix.csv")
    if arb_matrix is not None:
        print(f"    Source: AgentRewardBench per-task functional results")
        print(f"    Dimensions: {arb_matrix.shape[0]} agents x "
              f"{arb_matrix.shape[1]} tasks")
        fill = arb_matrix.notna().sum().sum() / (
            arb_matrix.shape[0] * arb_matrix.shape[1]
        ) * 100
        print(f"    Fill rate:  {fill:.1f}%")
        mean_sr = arb_matrix.mean().mean() * 100
        print(f"    Mean success rate: {mean_sr:.1f}%")
    else:
        print("    (not available — AgentRewardBench fetch failed)")
        print("    Fallback: response_matrix_leaderboard.csv")
        print(f"    Dimensions: {leaderboard_df.shape[0]} agents x "
              f"{leaderboard_df.shape[1]} columns")


if __name__ == "__main__":
    main()
