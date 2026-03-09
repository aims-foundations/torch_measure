#!/usr/bin/env python3
"""
Build TAU-bench response matrices from multiple data sources.

IMPORTANT NOTES ON TASK ALIGNMENT:
- Airline: v1 (tau-bench) and v2 (tau2-bench) have DIFFERENT tasks with the same
  ID numbering (0-49). They are output as separate matrices. HAL uses the v1 task set.
- Retail: v1 and v2 share the SAME tasks (v2 just removes persona prefix from
  instructions). They are merged into a single matrix.
- Telecom: Only exists in v2 (tau2-bench). No v1 data.

Data sources:
  1. tau-bench v1 historical trajectories (gpt-4o, claude-3.5-sonnet-v2;
     airline 50 tasks, retail 115 tasks; 4/8 trials)
  2. tau2-bench result trajectories (claude-3.7-sonnet, gpt-4.1, gpt-4.1-mini,
     o4-mini; airline 50 tasks, retail 114 tasks, telecom 114 tasks; 4 trials)
  3. HAL leaderboard heatmap (26 agents; airline 50 tasks from v1; 1 trial)
  4. tau2-bench leaderboard submissions (17 models; aggregate pass@k only)

Outputs (all in processed/):
  response_matrix_v1_airline.csv       50 tasks x (2 v1 models + 26 HAL agents)
  response_matrix_v2_airline.csv       50 tasks x 4 v2 models
  response_matrix_retail.csv          115 tasks x 6 models (v1+v2 merged)
  response_matrix_telecom.csv         114 tasks x 4 v2 models
  response_matrix_hal_airline.csv      50 tasks x 26 HAL agents
  response_matrix_combined.csv         All domains stacked, all trajectory models
  task_metadata.csv                    Task descriptions, user IDs, metadata
  leaderboard_summary.csv             Aggregate pass@k from 17 leaderboard models
  summary_stats.json                  Dimensions, fill rates, model lists
"""

import json
import os
import csv
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Parse tau-bench v1 trajectories
# ---------------------------------------------------------------------------
def parse_tau_v1_trajectories():
    """Parse tau-bench v1 historical trajectory files."""
    files = {
        ("gpt-4o", "airline"): "gpt-4o-airline.json",
        ("gpt-4o", "retail"): "gpt-4o-retail.json",
        ("claude-3.5-sonnet-v2", "airline"): "sonnet-35-new-airline.json",
        ("claude-3.5-sonnet-v2", "retail"): "sonnet-35-new-retail.json",
    }
    records = []
    task_meta = {}
    for (model, domain), fname in files.items():
        fpath = RAW_DIR / fname
        if not fpath.exists():
            print(f"  [SKIP] {fpath} not found")
            continue
        with open(fpath) as f:
            data = json.load(f)
        for entry in data:
            tid = entry["task_id"]
            records.append({
                "domain": domain,
                "task_id": str(tid),
                "model": model,
                "trial": entry["trial"],
                "reward": entry["reward"],
                "source": "tau-bench-v1",
            })
            meta_key = (domain, str(tid), "v1")
            if meta_key not in task_meta:
                task_info = entry["info"]["task"]
                task_meta[meta_key] = {
                    "domain": domain,
                    "task_id": str(tid),
                    "benchmark": "tau-bench-v1",
                    "user_id": task_info.get("user_id", ""),
                    "instruction": task_info.get("instruction", "")[:500],
                    "num_required_actions": len(task_info.get("actions", [])),
                    "has_expected_outputs": len(task_info.get("outputs", [])) > 0,
                    "purpose": "",
                }
    print(f"  tau-bench v1: {len(records)} records from {len(files)} files")
    return records, task_meta


# ---------------------------------------------------------------------------
# 2. Parse tau2-bench trajectory results
# ---------------------------------------------------------------------------
def parse_tau2_trajectories():
    """Parse tau2-bench result trajectory files."""
    results_dir = RAW_DIR / "tau2_results"
    if not results_dir.exists():
        print(f"  [SKIP] {results_dir} not found")
        return [], {}

    records = []
    task_meta = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = results_dir / fname
        with open(fpath) as f:
            data = json.load(f)

        parts = fname.replace(".json", "").split("_")
        domain = None
        for d in ("airline", "retail", "telecom"):
            if d in parts:
                domain = d
                break
        if domain is None:
            print(f"  [SKIP] Cannot determine domain for {fname}")
            continue

        domain_idx = parts.index(domain)
        model_name = "_".join(parts[:domain_idx])
        mode = parts[domain_idx + 1] if domain_idx + 1 < len(parts) else "default"
        model_label = model_name
        if mode not in ("default",):
            model_label += f"_{mode}"

        # Parse task metadata
        for task in data.get("tasks", []):
            meta_key = (domain, str(task["id"]), "v2")
            if meta_key not in task_meta:
                desc = task.get("description", {})
                purpose = ""
                if isinstance(desc, dict):
                    purpose = str(desc.get("purpose", "") or "")[:500]
                elif isinstance(desc, str):
                    purpose = desc[:500]

                # Extract user scenario info
                scenario = task.get("user_scenario", {})
                reason = ""
                if isinstance(scenario, dict):
                    instr = scenario.get("instructions", {})
                    if isinstance(instr, dict):
                        reason = str(instr.get("reason_for_call", ""))[:500]

                # Count evaluation criteria actions
                eval_crit = task.get("evaluation_criteria", {})
                num_actions = 0
                if isinstance(eval_crit, dict):
                    actions = eval_crit.get("actions", [])
                    if isinstance(actions, list):
                        num_actions = len(actions)

                task_meta[meta_key] = {
                    "domain": domain,
                    "task_id": str(task["id"]),
                    "benchmark": "tau2-bench",
                    "user_id": "",
                    "instruction": reason if reason else purpose,
                    "num_required_actions": num_actions,
                    "has_expected_outputs": False,
                    "purpose": purpose,
                }

        for sim in data.get("simulations", []):
            reward = sim.get("reward_info", {}).get("reward", 0.0)
            records.append({
                "domain": domain,
                "task_id": str(sim["task_id"]),
                "model": model_label,
                "trial": sim["trial"],
                "reward": reward,
                "source": "tau2-bench",
            })

        print(f"  {fname}: {len(data.get('simulations', []))} sims, "
              f"model={model_label}, domain={domain}")

    print(f"  tau2-bench: {len(records)} total records")
    return records, task_meta


# ---------------------------------------------------------------------------
# 3. Parse HAL leaderboard heatmap (airline only, v1 tasks)
# ---------------------------------------------------------------------------
def parse_hal_heatmap():
    """Parse HAL leaderboard heatmap for per-task per-model pass/fail."""
    fpath = RAW_DIR / "hal_airline_heatmap.json"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return [], {}

    with open(fpath) as f:
        data = json.load(f)

    heatmap = data["data"][0]
    models_raw = heatmap["y"]
    task_ids = heatmap["x"]
    z_matrix = heatmap["z"]

    records = []
    for i, model_name in enumerate(models_raw):
        if "<b>" in model_name or "Tasks Solved" in model_name:
            continue
        clean_name = model_name.strip()
        for j, tid in enumerate(task_ids):
            records.append({
                "domain": "airline",
                "task_id": str(tid),
                "model": clean_name,
                "trial": 0,
                "reward": z_matrix[i][j],
                "source": "hal-leaderboard",
            })

    num_models = sum(1 for m in models_raw
                     if "<b>" not in m and "Tasks Solved" not in m)
    print(f"  HAL heatmap: {len(records)} records, "
          f"{num_models} models, {len(task_ids)} tasks")
    return records, {}


# ---------------------------------------------------------------------------
# 4. Parse tau2 leaderboard submissions
# ---------------------------------------------------------------------------
def parse_tau2_submissions():
    """Parse tau2-bench leaderboard submission files for aggregate pass@k."""
    submissions_dir = RAW_DIR / "tau2_submissions"
    if not submissions_dir.exists():
        print(f"  [SKIP] {submissions_dir} not found")
        return []

    rows = []
    for fname in sorted(os.listdir(submissions_dir)):
        if not fname.endswith(".json"):
            continue
        with open(submissions_dir / fname) as f:
            data = json.load(f)

        model_name = data.get("model_name", fname.replace(".json", ""))
        org = data.get("model_organization", "")
        sub_date = data.get("submission_date", "")
        results = data.get("results", {})
        traj_avail = data.get("trajectories_available", False)

        for domain in ("airline", "retail", "telecom"):
            if domain not in results:
                continue
            r = results[domain]
            rows.append({
                "model": model_name,
                "organization": org,
                "domain": domain,
                "pass_1": r.get("pass_1"),
                "pass_2": r.get("pass_2"),
                "pass_3": r.get("pass_3"),
                "pass_4": r.get("pass_4"),
                "submission_date": sub_date,
                "trajectories_available": traj_avail,
                "source_file": fname,
            })

    print(f"  Leaderboard submissions: {len(rows)} domain-model entries "
          f"from {len(os.listdir(submissions_dir))} files")
    return rows


# ---------------------------------------------------------------------------
# Matrix building utilities
# ---------------------------------------------------------------------------
def build_mean_matrix(records):
    """Build a task x model matrix averaging reward across trials.

    Returns: (matrix dict, sorted task_ids, sorted model names)
    """
    if not records:
        return None, [], []

    models = sorted(set(r["model"] for r in records))
    task_ids_raw = set(r["task_id"] for r in records)
    try:
        task_ids = sorted(task_ids_raw, key=lambda x: int(x))
    except (ValueError, TypeError):
        task_ids = sorted(task_ids_raw, key=str)

    agg = defaultdict(list)
    for r in records:
        agg[(r["task_id"], r["model"])].append(r["reward"])

    matrix = {}
    for tid in task_ids:
        row = {}
        for model in models:
            vals = agg.get((tid, model), [])
            if vals:
                row[model] = round(sum(vals) / len(vals), 4)
            else:
                row[model] = None
        matrix[tid] = row

    return matrix, task_ids, models


def write_matrix_csv(filepath, matrix, task_ids, col_names, domain, extra_cols=None):
    """Write response matrix to CSV."""
    header = ["domain", "task_id"]
    if extra_cols:
        header += extra_cols
    header += col_names
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for tid in task_ids:
            row_data = matrix[tid]
            base = [domain, tid]
            if extra_cols:
                base += [row_data.get(f"_extra_{c}", "") for c in extra_cols]
            vals = []
            for c in col_names:
                v = row_data.get(c)
                vals.append("" if v is None else v)
            writer.writerow(base + vals)
    print(f"  Wrote {filepath.name} ({len(task_ids)} tasks x {len(col_names)} models)")


def compute_fill_rate(matrix, task_ids, models):
    """Compute fill rate of matrix."""
    filled = 0
    total = len(task_ids) * len(models)
    for tid in task_ids:
        for m in models:
            if matrix[tid].get(m) is not None:
                filled += 1
    return filled, total


def compute_model_means(matrix, task_ids, models):
    """Compute per-model mean score."""
    means = {}
    for m in models:
        vals = [matrix[tid][m] for tid in task_ids if matrix[tid].get(m) is not None]
        means[m] = round(sum(vals) / len(vals), 4) if vals else None
    return means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TAU-bench Response Matrix Builder")
    print("=" * 70)

    # ---- Parse all sources ----
    print("\n[1] Parsing tau-bench v1 trajectories...")
    v1_records, v1_meta = parse_tau_v1_trajectories()

    print("\n[2] Parsing tau2-bench trajectories...")
    v2_records, v2_meta = parse_tau2_trajectories()

    print("\n[3] Parsing HAL leaderboard heatmap (v1 airline tasks)...")
    hal_records, _ = parse_hal_heatmap()

    print("\n[4] Parsing tau2-bench leaderboard submissions...")
    leaderboard_rows = parse_tau2_submissions()

    # ---- Build separate matrices ----

    # == V1 AIRLINE (+ HAL) ==
    # v1 airline tasks are DIFFERENT from v2 airline tasks.
    # HAL uses v1 task set.
    print("\n[5] Building v1 airline matrix (v1 models + HAL agents)...")
    v1_airline = [r for r in v1_records if r["domain"] == "airline"]
    v1_airline_combined = v1_airline + hal_records
    mat, tids, models = build_mean_matrix(v1_airline_combined)
    if mat:
        write_matrix_csv(OUT_DIR / "response_matrix_v1_airline.csv",
                         mat, tids, models, "airline_v1")
        filled, total = compute_fill_rate(mat, tids, models)
        means = compute_model_means(mat, tids, models)
        print(f"    Fill rate: {filled}/{total} ({filled/total:.1%})")
        print(f"    Model means: { {m: f'{v:.2%}' for m, v in means.items()} }")

    # == V2 AIRLINE ==
    print("\n[6] Building v2 airline matrix...")
    v2_airline = [r for r in v2_records if r["domain"] == "airline"]
    mat, tids, models = build_mean_matrix(v2_airline)
    if mat:
        write_matrix_csv(OUT_DIR / "response_matrix_v2_airline.csv",
                         mat, tids, models, "airline_v2")
        filled, total = compute_fill_rate(mat, tids, models)
        means = compute_model_means(mat, tids, models)
        print(f"    Fill rate: {filled}/{total} ({filled/total:.1%})")
        print(f"    Model means: { {m: f'{v:.2%}' for m, v in means.items()} }")

    # == RETAIL (v1 + v2 merged, same tasks) ==
    print("\n[7] Building retail matrix (v1 + v2 merged)...")
    retail_records = ([r for r in v1_records if r["domain"] == "retail"]
                      + [r for r in v2_records if r["domain"] == "retail"])
    mat, tids, models = build_mean_matrix(retail_records)
    if mat:
        write_matrix_csv(OUT_DIR / "response_matrix_retail.csv",
                         mat, tids, models, "retail")
        filled, total = compute_fill_rate(mat, tids, models)
        means = compute_model_means(mat, tids, models)
        print(f"    Fill rate: {filled}/{total} ({filled/total:.1%})")
        print(f"    Model means: { {m: f'{v:.2%}' for m, v in means.items()} }")

    # == TELECOM (v2 only) ==
    print("\n[8] Building telecom matrix (v2 only)...")
    telecom_records = [r for r in v2_records if r["domain"] == "telecom"]
    mat, tids, models = build_mean_matrix(telecom_records)
    if mat:
        write_matrix_csv(OUT_DIR / "response_matrix_telecom.csv",
                         mat, tids, models, "telecom")
        filled, total = compute_fill_rate(mat, tids, models)
        means = compute_model_means(mat, tids, models)
        print(f"    Fill rate: {filled}/{total} ({filled/total:.1%})")
        print(f"    Model means: { {m: f'{v:.2%}' for m, v in means.items()} }")

    # == HAL airline only (standalone) ==
    print("\n[9] Building HAL-only airline matrix...")
    mat, tids, models = build_mean_matrix(hal_records)
    if mat:
        write_matrix_csv(OUT_DIR / "response_matrix_hal_airline.csv",
                         mat, tids, models, "airline_v1_hal")
        filled, total = compute_fill_rate(mat, tids, models)
        print(f"    Fill rate: {filled}/{total} ({filled/total:.1%})")

    # == COMBINED (all trajectory sources, domain-prefixed task IDs) ==
    print("\n[10] Building combined response matrix...")
    combined_records = []
    # v1 airline -> airline_v1
    for r in v1_records:
        cr = dict(r)
        if r["domain"] == "airline":
            cr["combined_domain"] = "airline_v1"
        else:
            cr["combined_domain"] = r["domain"]
        cr["combined_task_id"] = f"{cr['combined_domain']}_{r['task_id']}"
        combined_records.append(cr)
    # v2
    for r in v2_records:
        cr = dict(r)
        if r["domain"] == "airline":
            cr["combined_domain"] = "airline_v2"
        else:
            cr["combined_domain"] = r["domain"]
        cr["combined_task_id"] = f"{cr['combined_domain']}_{r['task_id']}"
        combined_records.append(cr)
    # HAL
    for r in hal_records:
        cr = dict(r)
        cr["combined_domain"] = "airline_v1"
        cr["combined_task_id"] = f"airline_v1_{r['task_id']}"
        combined_records.append(cr)

    all_models = sorted(set(r["model"] for r in combined_records))
    all_task_ids = sorted(set(r["combined_task_id"] for r in combined_records))

    agg = defaultdict(list)
    task_domain_map = {}
    for r in combined_records:
        agg[(r["combined_task_id"], r["model"])].append(r["reward"])
        task_domain_map[r["combined_task_id"]] = r["combined_domain"]

    filepath = OUT_DIR / "response_matrix_combined.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "task_id"] + all_models)
        for ctid in all_task_ids:
            domain = task_domain_map[ctid]
            row = [domain, ctid]
            for m in all_models:
                vals = agg.get((ctid, m), [])
                if vals:
                    row.append(round(sum(vals) / len(vals), 4))
                else:
                    row.append("")
            writer.writerow(row)
    # Compute fill
    filled_combined = sum(1 for ctid in all_task_ids for m in all_models
                          if agg.get((ctid, m)))
    total_combined = len(all_task_ids) * len(all_models)
    print(f"  Wrote response_matrix_combined.csv "
          f"({len(all_task_ids)} tasks x {len(all_models)} models)")
    print(f"    Fill rate: {filled_combined}/{total_combined} "
          f"({filled_combined/total_combined:.1%})")

    # ---- Task metadata ----
    print("\n[11] Writing task metadata...")
    task_meta = {**v1_meta, **v2_meta}
    if task_meta:
        rows = sorted(task_meta.values(),
                      key=lambda x: (x["benchmark"], x["domain"], str(x["task_id"])))
        fieldnames = ["benchmark", "domain", "task_id", "user_id", "purpose",
                      "instruction", "num_required_actions", "has_expected_outputs"]
        with open(OUT_DIR / "task_metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote task_metadata.csv ({len(rows)} tasks)")

    # ---- Leaderboard summary ----
    print("\n[12] Writing leaderboard summary...")
    if leaderboard_rows:
        fieldnames = ["model", "organization", "domain", "pass_1", "pass_2",
                      "pass_3", "pass_4", "submission_date",
                      "trajectories_available", "source_file"]
        with open(OUT_DIR / "leaderboard_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(leaderboard_rows)
        print(f"  Wrote leaderboard_summary.csv ({len(leaderboard_rows)} rows)")

    # ---- Summary statistics ----
    print("\n[13] Computing summary statistics...")
    stats = {
        "score_type": "binary (0.0 or 1.0 pass/fail per trial)",
        "aggregation": "mean reward across trials per (task, model) cell",
        "note_airline_v1_v2": (
            "Airline tasks differ between v1 (tau-bench) and v2 (tau2-bench). "
            "They use the same task_id numbering (0-49) but are completely "
            "different scenarios. Separate matrices are provided."
        ),
        "note_retail_merged": (
            "Retail tasks are the same between v1 and v2 (v2 removes persona "
            "prefix). They are merged into a single matrix."
        ),
        "note_hal": (
            "HAL leaderboard runs the original tau-bench v1 airline tasks. "
            "Results are single-trial pass/fail."
        ),
    }

    # Per-matrix stats
    matrix_specs = {
        "v1_airline": {
            "file": "response_matrix_v1_airline.csv",
            "sources": "tau-bench v1 trajectories + HAL leaderboard",
        },
        "v2_airline": {
            "file": "response_matrix_v2_airline.csv",
            "sources": "tau2-bench trajectories",
        },
        "retail": {
            "file": "response_matrix_retail.csv",
            "sources": "tau-bench v1 + tau2-bench trajectories (merged)",
        },
        "telecom": {
            "file": "response_matrix_telecom.csv",
            "sources": "tau2-bench trajectories",
        },
        "hal_airline": {
            "file": "response_matrix_hal_airline.csv",
            "sources": "HAL leaderboard heatmap",
        },
        "combined": {
            "file": "response_matrix_combined.csv",
            "sources": "All trajectory sources combined",
        },
    }

    for key, spec in matrix_specs.items():
        fpath = OUT_DIR / spec["file"]
        if fpath.exists():
            with open(fpath) as f:
                reader = csv.reader(f)
                header = next(reader)
                rows_data = list(reader)
            model_cols = header[2:]  # skip domain, task_id
            n_tasks = len(rows_data)
            n_models = len(model_cols)
            filled = sum(1 for row in rows_data for v in row[2:] if v != "")
            total = n_tasks * n_models
            stats[f"matrix_{key}"] = {
                "file": spec["file"],
                "sources": spec["sources"],
                "num_tasks": n_tasks,
                "num_models": n_models,
                "model_names": model_cols,
                "total_cells": total,
                "filled_cells": filled,
                "fill_rate": round(filled / total, 4) if total else 0,
            }

    # Source record counts
    all_recs = v1_records + v2_records + hal_records
    stats["total_trajectory_records"] = len(all_recs)
    stats["total_unique_models"] = len(set(r["model"] for r in all_recs))
    stats["leaderboard_models_count"] = len(set(r["model"] for r in leaderboard_rows))
    stats["leaderboard_models"] = sorted(set(r["model"] for r in leaderboard_rows))
    stats["task_metadata_entries"] = len(task_meta)

    # Per-source
    src_counts = defaultdict(int)
    src_models = defaultdict(set)
    for r in all_recs:
        src_counts[r["source"]] += 1
        src_models[r["source"]].add(r["model"])
    stats["per_source"] = {
        src: {"records": src_counts[src], "models": sorted(src_models[src])}
        for src in src_counts
    }

    stats_path = OUT_DIR / "summary_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote summary_stats.json")

    # ---- Print final report ----
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    print(f"\nScore type: {stats['score_type']}")
    print(f"Aggregation: {stats['aggregation']}")
    print(f"Total trajectory records: {stats['total_trajectory_records']}")
    print(f"Total unique models/agents (trajectory): {stats['total_unique_models']}")
    print(f"Leaderboard models (aggregate only): {stats['leaderboard_models_count']}")
    print(f"Task metadata entries: {stats['task_metadata_entries']}")

    print("\n--- Response Matrix Dimensions ---")
    for key in matrix_specs:
        mkey = f"matrix_{key}"
        if mkey in stats:
            m = stats[mkey]
            print(f"\n  {m['file']}:")
            print(f"    Dimensions: {m['num_tasks']} tasks x {m['num_models']} models")
            print(f"    Fill rate: {m['filled_cells']}/{m['total_cells']} "
                  f"({m['fill_rate']:.1%})")
            print(f"    Sources: {m['sources']}")
            print(f"    Models: {', '.join(m['model_names'])}")

    print("\n--- Per-source record counts ---")
    for src, info in stats["per_source"].items():
        print(f"  {src}: {info['records']} records, models: {', '.join(info['models'])}")

    print("\n--- Leaderboard Models (aggregate pass@k only, no per-task data) ---")
    for m in stats["leaderboard_models"]:
        print(f"  {m}")

    print("\n--- Output files ---")
    for p in sorted(OUT_DIR.glob("*")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:45s} ({size_kb:7.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
