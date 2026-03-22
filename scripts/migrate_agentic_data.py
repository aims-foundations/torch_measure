#!/usr/bin/env python3
# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Migrate agentic benchmark data from aims-foundation/eval_response_matrix to torch-measure-data.

Downloads binary response matrices and continuous scores from the HAL
(Holistic Agent Leaderboard) agentic benchmarks, converts them to the
standardized .pt dict format, and uploads to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxx  # token with read access to aims-foundation, write to torch-measure-data
    python scripts/migrate_agentic_data.py

Source data layout (aims-foundation/eval_response_matrix):
    pre-revision/{benchmark}/benchmark.csv          — binary response matrix (agents x items)
    pre-revision/{benchmark}/{score_type}.csv       — continuous score matrix
    post-revision/{benchmark}/resmat/resmat_{run}.csv — binary reruns
    post-revision/{benchmark}/scores/{type}_{run}.csv — continuous score reruns
    model_specs.csv                                  — model specifications (183 HELM models)

Each CSV has:
    - First column 'agent': model/agent identifier
    - Remaining columns '{benchmark}.{item_id}': response values

Destination .pt file format (consumed by torch_measure.datasets.load):
    {
        "data": torch.Tensor,             # (n_subjects, n_items), float32, NaN for missing
        "subject_ids": list[str],          # agent identifiers
        "item_ids": list[str],             # item identifiers
        "item_contents": list[str],        # (empty for agentic — task IDs are the identifiers)
        "subject_metadata": list[dict],    # structured agent metadata
    }
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_REPO = "aims-foundation/eval_response_matrix"
DST_REPO = "aims-foundation/torch-measure-data"
TMP_DIR = Path("/tmp/torch_measure_migration")

# Pre-revision benchmarks and their available score files.
PRE_REVISION_BENCHMARKS = {
    "assistantbench": [],
    "colbench_backend_programming": ["raw_score"],
    "corebench_hard": ["vision_score", "written_score"],
    "gaia": [],
    "online_mind2web": [],
    "scicode": [],
    "scienceagentbench": ["codebert_score", "success_rate", "valid_program"],
    "swebench_verified_mini": [],
    "taubench_airline": [],
    "usaco": [],
}

# Registry-friendly names (agentic/{short_name}).
BENCHMARK_NAME_MAP = {
    "assistantbench": "assistantbench",
    "colbench_backend_programming": "colbench",
    "corebench_hard": "corebench_hard",
    "gaia": "gaia",
    "online_mind2web": "mind2web",
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "swebench_verified_mini": "swebench",
    "taubench_airline": "taubench_airline",
    "usaco": "usaco",
}

# Post-revision benchmarks: benchmark → (run_prefix, run_ids).
POST_REVISION_BENCHMARKS: dict[str, list[str]] = {
    "colbench_backend_programming": (
        [f"moon{i}" for i in [0] + list(range(17, 41))]
        + [f"sun{i}" for i in range(12, 41)]
    ),
    "corebench_hard": [f"cloud{i}" for i in range(11)],
    "scicode": [f"beach{i}" for i in range(11)],
    "scienceagentbench": [f"sky{i}" for i in range(11)],
}

# Post-revision score types per benchmark.
POST_REVISION_SCORES: dict[str, list[str]] = {
    "colbench_backend_programming": ["raw_score"],
    "corebench_hard": ["vision_score", "written_score"],
    "scicode": ["subtask_score"],
    "scienceagentbench": ["codebert_score", "success_rate", "valid_program"],
}


# ---------------------------------------------------------------------------
# Agent metadata parsing
# ---------------------------------------------------------------------------

# Common org display names.
_ORG_DISPLAY = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "meta": "Meta",
    "gemini": "Google",
    "gpt": "OpenAI",
    "claude": "Anthropic",
    "deepseek": "DeepSeek",
    "o3": "OpenAI",
    "o4": "OpenAI",
}

# Regex to extract parameter count from model name.
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")


def _parse_agent_name(agent: str) -> dict:
    """Parse an agent identifier into structured metadata.

    Agent names follow patterns like:
        hal_generalist__claude_3_7_sonnet_20250219
        colbench_backend_programming_colbench_example_agent__gemini/gemini_20_flash
        core_agent__anthropic/claude_sonnet_4_5_high
        usaco_episodic__semantic__claude_3_7_sonnet_20250219_high

    Returns dict with keys: agent_framework, model, org, reasoning_effort.
    """
    # Split on '__' to separate framework from model
    parts = agent.split("__")
    if len(parts) >= 2:
        framework = parts[0]
        model_part = "__".join(parts[1:])
    else:
        framework = ""
        model_part = agent

    # Handle {benchmark}.{model} pattern in post-revision
    if "." in model_part and "/" not in model_part.split(".")[0]:
        dot_parts = model_part.split(".", 1)
        if not any(c.isdigit() for c in dot_parts[0]):
            model_part = dot_parts[1]

    # Extract org from "org/model" pattern
    if "/" in model_part:
        org_raw, model_name = model_part.split("/", 1)
    else:
        model_name = model_part
        org_raw = ""

    # Try to infer org from model name if not explicit
    if not org_raw:
        lower = model_name.lower()
        if "claude" in lower:
            org_raw = "anthropic"
        elif "gpt" in lower or lower.startswith("o3") or lower.startswith("o4"):
            org_raw = "openai"
        elif "gemini" in lower or "gemma" in lower:
            org_raw = "google"
        elif "llama" in lower:
            org_raw = "meta"
        elif "deepseek" in lower:
            org_raw = "deepseek"

    org = _ORG_DISPLAY.get(org_raw, org_raw)

    # Detect reasoning effort level (high/medium/low)
    reasoning = None
    for level in ("high", "medium", "low"):
        if model_name.endswith(f"_{level}"):
            reasoning = level
            model_name = model_name[: -(len(level) + 1)]
            break

    # Extract param count
    param_count = None
    match = _PARAM_RE.search(model_name)
    if match:
        num = match.group(1)
        if num.endswith(".0"):
            num = num[:-2]
        param_count = f"{num}B"

    return {
        "agent_framework": framework,
        "model": model_name,
        "org": org,
        "param_count": param_count,
        "reasoning_effort": reasoning,
    }


def build_subject_metadata(subject_ids: list[str]) -> list[dict]:
    """Build structured metadata for each agent."""
    return [_parse_agent_name(sid) for sid in subject_ids]


# ---------------------------------------------------------------------------
# CSV loading utilities
# ---------------------------------------------------------------------------


def load_csv(hf_path: str) -> pd.DataFrame:
    """Download and load a CSV from the source HF repo."""
    local = hf_hub_download(SRC_REPO, hf_path, repo_type="dataset")
    return pd.read_csv(local)


def csv_to_payload(df: pd.DataFrame, benchmark: str) -> dict:
    """Convert a benchmark CSV (agents x items) to a .pt payload dict.

    The first column is 'agent'; remaining columns are item responses.
    Column names are '{benchmark}.{item_id}'.
    """
    subject_ids = df.iloc[:, 0].astype(str).tolist()
    # Coerce data columns to numeric (handles empty CSVs with object dtype)
    numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    data = torch.tensor(numeric_df.values, dtype=torch.float32)

    # Extract item IDs from column names: "benchmark.item_id" → "item_id"
    raw_cols = df.columns[1:].tolist()
    item_ids = []
    for col in raw_cols:
        # Strip benchmark prefix if present
        if col.startswith(f"{benchmark}."):
            item_ids.append(col[len(benchmark) + 1 :])
        else:
            item_ids.append(col)

    # No textual item content for agentic benchmarks (tasks identified by ID)
    item_contents = [""] * len(item_ids)

    subject_metadata = build_subject_metadata(subject_ids)

    return {
        "data": data,
        "subject_ids": subject_ids,
        "item_ids": item_ids,
        "item_contents": item_contents,
        "subject_metadata": subject_metadata,
    }


# ---------------------------------------------------------------------------
# Pre-revision processing
# ---------------------------------------------------------------------------


def process_pre_revision() -> dict[str, dict]:
    """Process all pre-revision benchmark.csv files.

    Returns dict mapping registry name → payload.
    """
    payloads: dict[str, dict] = {}

    for bench, score_types in PRE_REVISION_BENCHMARKS.items():
        short = BENCHMARK_NAME_MAP[bench]
        print(f"\n--- Pre-revision: {bench} → agentic/{short} ---")

        # Binary response matrix
        df = load_csv(f"pre-revision/{bench}/benchmark.csv")
        payload = csv_to_payload(df, bench)
        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  benchmark.csv: {n_sub} agents x {n_items} items, {nan_pct:.1%} missing")
        payloads[f"agentic/{short}"] = payload

        # Continuous score variants
        for score_type in score_types:
            df_score = load_csv(f"pre-revision/{bench}/{score_type}.csv")
            score_payload = csv_to_payload(df_score, bench)
            n_sub_s, n_items_s = score_payload["data"].shape
            uniq = score_payload["data"][~torch.isnan(score_payload["data"])].unique()
            is_binary = set(uniq.tolist()).issubset({0.0, 1.0})
            dtype_str = "binary" if is_binary else f"continuous ({len(uniq)} unique values)"
            print(f"  {score_type}.csv: {n_sub_s} x {n_items_s}, {dtype_str}")
            payloads[f"agentic/{short}_{score_type}"] = score_payload

    return payloads


# ---------------------------------------------------------------------------
# Post-revision processing (multi-run data)
# ---------------------------------------------------------------------------


def process_post_revision() -> dict[str, dict]:
    """Process post-revision multi-run data.

    For each benchmark, loads all run CSVs and averages across runs
    to produce a continuous response matrix (pass rate per item).
    """
    payloads: dict[str, dict] = {}

    for bench, run_ids in POST_REVISION_BENCHMARKS.items():
        short = BENCHMARK_NAME_MAP[bench]
        print(f"\n--- Post-revision: {bench} → agentic/{short}_rerun ---")

        # Load all resmat runs (skip empty CSVs)
        all_dfs = []
        for run_id in run_ids:
            try:
                df = load_csv(f"post-revision/{bench}/resmat/resmat_{run_id}.csv")
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                print(f"  Warning: skipping {run_id}: {e}")

        if not all_dfs:
            print(f"  No runs loaded for {bench}, skipping")
            continue

        print(f"  Loaded {len(all_dfs)} runs")

        # Find union of all agents and items across runs
        all_agents = sorted(set().union(*(set(df.iloc[:, 0]) for df in all_dfs)))
        all_items = sorted(set().union(*(set(df.columns[1:]) for df in all_dfs)))

        # Build 3D tensor: (n_runs, n_agents, n_items) then average
        agent_to_idx = {a: i for i, a in enumerate(all_agents)}
        item_to_idx = {it: i for i, it in enumerate(all_items)}

        accumulator = torch.zeros(len(all_agents), len(all_items))
        count = torch.zeros(len(all_agents), len(all_items))

        for df in all_dfs:
            agents = df.iloc[:, 0].tolist()
            items = df.columns[1:].tolist()
            numeric = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            data = torch.tensor(numeric.values, dtype=torch.float32)
            for i, agent in enumerate(agents):
                ai = agent_to_idx[agent]
                for j, item in enumerate(items):
                    ji = item_to_idx[item]
                    val = data[i, j].item()
                    if not pd.isna(val):
                        accumulator[ai, ji] += val
                        count[ai, ji] += 1

        # Average: where count > 0, else NaN
        avg_data = torch.where(count > 0, accumulator / count, torch.tensor(float("nan")))

        # Build item_ids
        item_ids = []
        for col in all_items:
            if col.startswith(f"{bench}."):
                item_ids.append(col[len(bench) + 1 :])
            else:
                item_ids.append(col)

        payload = {
            "data": avg_data,
            "subject_ids": all_agents,
            "item_ids": item_ids,
            "item_contents": [""] * len(item_ids),
            "subject_metadata": build_subject_metadata(all_agents),
        }

        n_sub, n_items = avg_data.shape
        nan_pct = torch.isnan(avg_data).float().mean().item()
        print(f"  Averaged: {n_sub} agents x {n_items} items, {nan_pct:.1%} missing")
        payloads[f"agentic/{short}_rerun"] = payload

        # Also process score reruns
        for score_type in POST_REVISION_SCORES.get(bench, []):
            score_dfs = []
            for run_id in run_ids:
                try:
                    df_s = load_csv(f"post-revision/{bench}/scores/{score_type}_{run_id}.csv")
                    if len(df_s) > 0:
                        score_dfs.append(df_s)
                except Exception:
                    pass

            if not score_dfs:
                continue

            # Same averaging approach
            s_agents = sorted(set().union(*(set(df.iloc[:, 0]) for df in score_dfs)))
            s_items = sorted(set().union(*(set(df.columns[1:]) for df in score_dfs)))
            s_agent_idx = {a: i for i, a in enumerate(s_agents)}
            s_item_idx = {it: i for i, it in enumerate(s_items)}

            s_acc = torch.zeros(len(s_agents), len(s_items))
            s_cnt = torch.zeros(len(s_agents), len(s_items))

            for df_s in score_dfs:
                agents = df_s.iloc[:, 0].tolist()
                items = df_s.columns[1:].tolist()
                numeric = df_s.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
                data = torch.tensor(numeric.values, dtype=torch.float32)
                for i, agent in enumerate(agents):
                    ai = s_agent_idx[agent]
                    for j, item in enumerate(items):
                        ji = s_item_idx[item]
                        val = data[i, j].item()
                        if not pd.isna(val):
                            s_acc[ai, ji] += val
                            s_cnt[ai, ji] += 1

            s_avg = torch.where(s_cnt > 0, s_acc / s_cnt, torch.tensor(float("nan")))

            s_item_ids = []
            for col in s_items:
                if col.startswith(f"{bench}."):
                    s_item_ids.append(col[len(bench) + 1 :])
                else:
                    s_item_ids.append(col)

            s_payload = {
                "data": s_avg,
                "subject_ids": s_agents,
                "item_ids": s_item_ids,
                "item_contents": [""] * len(s_item_ids),
                "subject_metadata": build_subject_metadata(s_agents),
            }

            n_s, n_i = s_avg.shape
            print(f"  {score_type} (avg of {len(score_dfs)} runs): {n_s} x {n_i}")
            payloads[f"agentic/{short}_rerun_{score_type}"] = s_payload

    return payloads


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def make_aggregate(payloads: dict[str, dict]) -> dict | None:
    """Create aggregated response matrix from all pre-revision binary benchmarks."""
    # Only include the main binary benchmarks (not scores, not reruns)
    main_names = [f"agentic/{BENCHMARK_NAME_MAP[b]}" for b in PRE_REVISION_BENCHMARKS]
    main_payloads = {k: v for k, v in payloads.items() if k in main_names}

    if not main_payloads:
        return None

    # Find union of all agents
    all_agents = sorted(set().union(*(set(p["subject_ids"]) for p in main_payloads.values())))
    agent_to_idx = {a: i for i, a in enumerate(all_agents)}

    # Concatenate items across benchmarks
    all_data_cols = []
    all_item_ids = []
    all_item_contents = []

    for name in sorted(main_payloads.keys()):
        payload = main_payloads[name]
        short = name.split("/")[1]  # e.g., "swebench"
        n_items = payload["data"].shape[1]

        # Map this benchmark's agents into the global agent order
        col_data = torch.full((len(all_agents), n_items), float("nan"))
        for i, sid in enumerate(payload["subject_ids"]):
            global_idx = agent_to_idx[sid]
            col_data[global_idx] = payload["data"][i]

        all_data_cols.append(col_data)
        all_item_ids.extend([f"{short}:{iid}" for iid in payload["item_ids"]])
        all_item_contents.extend(payload["item_contents"])

    data = torch.cat(all_data_cols, dim=1)
    return {
        "data": data,
        "subject_ids": all_agents,
        "item_ids": all_item_ids,
        "item_contents": all_item_contents,
        "subject_metadata": build_subject_metadata(all_agents),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def ensure_repo(api: HfApi) -> None:
    """Create the destination repo if it doesn't exist."""
    try:
        api.repo_info(DST_REPO, repo_type="dataset")
        print(f"Destination repo {DST_REPO} already exists.")
    except Exception:
        print(f"Creating dataset repo {DST_REPO}...")
        api.create_repo(DST_REPO, repo_type="dataset", private=False)


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    ensure_repo(api)

    # Step 1: Process pre-revision data
    print("=" * 60)
    print("Processing pre-revision benchmarks...")
    print("=" * 60)
    payloads = process_pre_revision()

    # Step 2: Process post-revision multi-run data
    print("\n" + "=" * 60)
    print("Processing post-revision reruns...")
    print("=" * 60)
    rerun_payloads = process_post_revision()
    payloads.update(rerun_payloads)

    # Step 3: Create aggregate
    print("\n" + "=" * 60)
    print("Creating aggregate...")
    print("=" * 60)
    agg = make_aggregate(payloads)
    if agg is not None:
        n_sub, n_items = agg["data"].shape
        nan_pct = torch.isnan(agg["data"]).float().mean().item()
        print(f"  agentic/all: {n_sub} agents x {n_items} items, {nan_pct:.1%} missing")
        payloads["agentic/all"] = agg

    # Step 4: Save and upload
    print("\n" + "=" * 60)
    print("Saving and uploading...")
    print("=" * 60)
    for name, payload in sorted(payloads.items()):
        filename = f"{name}.pt"
        local_path = TMP_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)

        n_sub, n_items = payload["data"].shape
        nan_pct = torch.isnan(payload["data"]).float().mean().item()
        print(f"  {filename}: {n_sub}x{n_items}, {nan_pct:.0%} missing")

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=DST_REPO,
            repo_type="dataset",
        )

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Source: {SRC_REPO}")
    print(f"  Destination: {DST_REPO}")
    print(f"  Datasets uploaded: {len(payloads)}")
    print("\nDataset dimensions (for agentic.py registry):")
    for name, payload in sorted(payloads.items()):
        n_sub, n_items = payload["data"].shape
        print(f"  {name}: n_subjects={n_sub}, n_items={n_items}")
    print(f"\nVerify at: https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    main()
