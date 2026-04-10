"""
Build response matrix for DPAI Arena from TeamCity API.

Source: https://dpaia.dev / https://dpaia.teamcity.com
Downloads per-task score reports for each agent, then builds a response matrix.

Agents: junie_cli (3 models), claude_code (3 configs), codex_cli, gemini_cli (2 models)

Outputs:
  - response_matrix_total_score.csv : agents (rows) x tasks (columns) -> total score
  - item_content.csv                : task_id, task description
"""

import csv
import json
import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

AGENTS = {
    "junie_cli_claude_opus_4.5": "10697",
    "junie_cli_sonnet_4.5": "10696",
    "junie_cli_gemini_3.0": "10698",
    "claude_code_opus_4.5": "10699",
    "claude_code_sonnet_4.5_auto": "10700",
    "claude_code_sonnet_4.5_explicit": "10701",
    "codex_cli_gpt5_codex": "10702",
    "gemini_cli_gemini_3.0_preview": "10703",
    "gemini_cli_gemini_2.5_pro": "10704",
}


def download():
    """Download DPAI score reports from TeamCity API."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading DPAI score reports from TeamCity API...")
    for agent_name, build_id in AGENTS.items():
        out_file = RAW_DIR / f"{agent_name}_score_report.json"
        if out_file.exists():
            print(f"  {agent_name}: already downloaded")
            continue
        url = (f"https://dpaia.teamcity.com/guestAuth/app/rest/builds/id:{build_id}"
               f"/artifacts/content/aggregated_total_score_report.json")
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            data = urllib.request.urlopen(req, timeout=30).read()
            out_file.write_bytes(data)
            print(f"  {agent_name}: downloaded")
        except Exception as e:
            print(f"  {agent_name}: FAILED ({e})")


def main():
    download()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build response matrix from score reports
    rows = {}
    for f in sorted(RAW_DIR.glob("*_score_report.json")):
        agent = f.stem.replace("_score_report", "")
        with open(f) as fh:
            data = json.load(fh)
        # Data is a dict with "task_scores" list, not a flat list
        tasks = data.get("task_scores", data) if isinstance(data, dict) else data
        if not isinstance(tasks, list):
            print(f"  WARNING: unexpected format in {f.name}, skipping")
            continue
        for task in tasks:
            tid = task.get("task_id", task.get("instance_id", ""))
            total_score = task.get("total_score", {})
            if isinstance(total_score, dict):
                score = total_score.get("final_score", total_score.get("score", 0))
            else:
                score = total_score
            rows.setdefault(tid, {})[agent] = score

    agents_list = sorted({a for r in rows.values() for a in r})
    tasks_sorted = sorted(rows.keys())

    # Save response matrix (agents as rows, tasks as columns)
    matrix = pd.DataFrame(index=agents_list, columns=tasks_sorted, dtype=float)
    matrix.index.name = "model"
    for tid, scores in rows.items():
        for agent, score in scores.items():
            matrix.loc[agent, tid] = score

    matrix.to_csv(OUTPUT_DIR / "response_matrix_total_score.csv")
    fill = matrix.notna().sum().sum()
    total = matrix.shape[0] * matrix.shape[1]
    print(f"Saved response_matrix_total_score.csv "
          f"({len(agents_list)} agents x {len(tasks_sorted)} tasks, "
          f"{fill}/{total} = {fill/total*100:.0f}% fill)")

    # Save item_content.csv (task IDs as content — no richer description available)
    items = pd.DataFrame({"item_id": tasks_sorted, "content": tasks_sorted})
    items.to_csv(OUTPUT_DIR / "item_content.csv", index=False)
    print(f"Saved item_content.csv ({len(items)} items)")


if __name__ == "__main__":
    main()
