"""
Build a response matrix for VisualWebArena (VWA) multimodal web agent benchmark.

This script constructs a task x model response matrix with per-task pass/fail results.
Rows = 910 VWA tasks, Columns = model/agent configurations.

Data sources:
1. AgentRewardBench (McGill-NLP/agent-reward-bench on HuggingFace):
   - Per-task automated evaluation rewards (cum_reward) from BrowserGym trajectories
   - 3 models: GPT-4o, Claude 3.7 Sonnet, Qwen2.5-VL-72B
   - 100 tasks per model (same 100 tasks across all 3 models)

2. AgentRewardBench human annotations (annotations.csv):
   - Human-verified trajectory_success labels
   - Same 3 models, ~200 unique VWA tasks with human review

3. VisualWebArena paper (arXiv:2401.13649) + Google Sheets leaderboard:
   - Aggregate success rates for 36 model configurations
   - Per-domain (classifieds/reddit/shopping) breakdowns for original paper models

4. BrowserGym task configs (test_raw.json):
   - Complete 910 task metadata with global task IDs (0-909)
   - Task intent, domain, eval type, difficulty

Output:
- processed/response_matrix.csv: 910 rows x N model columns (1=pass, 0=fail, NaN=not evaluated)
- processed/task_metadata.csv: 910 rows with task details
- processed/leaderboard_aggregate.csv: All models with aggregate success rates
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 1: Load task metadata from BrowserGym's test_raw.json
# ============================================================
def load_task_metadata():
    """Load the 910 VWA task configs with global IDs (0-909)."""
    # Try installed package first
    json_path = None
    for candidate in [
        "/lfs/local/0/sttruong/miniconda3/lib/python3.12/site-packages/visualwebarena/test_raw.json",
        str(RAW_DIR / "test_raw.json"),
    ]:
        if os.path.exists(candidate):
            json_path = candidate
            break

    if json_path is None:
        # Download from BrowserGym GitHub
        print("Downloading test_raw.json from BrowserGym...")
        url = ("https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/"
               "browsergym/visualwebarena/src/browsergym/visualwebarena/test_raw.json")
        try:
            urllib.request.urlretrieve(url, str(RAW_DIR / "test_raw.json"))
            json_path = str(RAW_DIR / "test_raw.json")
        except Exception:
            # Fall back to libvisualwebarena package path
            try:
                import importlib.resources
                json_path = str(
                    importlib.resources.files("visualwebarena").joinpath("test_raw.json")
                )
            except Exception:
                print("ERROR: Cannot find test_raw.json. Install browsergym-visualwebarena.")
                sys.exit(1)

    with open(json_path) as f:
        all_configs = json.load(f)

    print(f"Loaded {len(all_configs)} VWA tasks from {json_path}")
    return all_configs


def save_task_metadata(all_configs):
    """Save task metadata CSV."""
    rows = []
    for c in all_configs:
        primary_domain = c["sites"][0] if c["sites"] else "unknown"
        all_sites = "+".join(c["sites"])
        rows.append({
            "task_id": c["task_id"],
            "domain": primary_domain,
            "all_sites": all_sites,
            "intent": c["intent"],
            "eval_types": ",".join(c["eval"]["eval_types"]),
            "reasoning_difficulty": c.get("reasoning_difficulty", ""),
            "visual_difficulty": c.get("visual_difficulty", ""),
            "overall_difficulty": c.get("overall_difficulty", ""),
            "require_login": c.get("require_login", False),
            "require_reset": c.get("require_reset", False),
        })

    out_path = PROCESSED_DIR / "task_metadata.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved task metadata: {out_path} ({len(rows)} tasks)")
    return rows


# ============================================================
# Step 2: Load AgentRewardBench per-task automated rewards
# ============================================================
def load_agentrewardbench_rewards():
    """Load per-task cum_reward from AgentRewardBench trajectory JSONs."""
    cache_path = RAW_DIR / "agentrewardbench_per_task_rewards.json"

    if cache_path.exists():
        print(f"Loading cached AgentRewardBench rewards from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print("Downloading AgentRewardBench per-task rewards (this may take a minute)...")

    models = {
        "GenericAgent-gpt-4o-2024-11-20": (
            "cleaned/visualwebarena/GenericAgent-gpt-4o-2024-11-20/"
            "GenericAgent-gpt-4o-2024-11-20_on_visualwebarena"
        ),
        "GenericAgent-claude-3.7-sonnet": (
            "cleaned/visualwebarena/GenericAgent-anthropic_claude-3.7-sonnet/"
            "GenericAgent-anthropic_claude-3.7-sonnet_on_visualwebarena.resized"
        ),
        "GenericAgent-qwen2.5-vl-72b": (
            "cleaned/visualwebarena/GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct/"
            "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct_on_visualwebarena.resized"
        ),
    }

    all_results = {}
    for model_name, path in models.items():
        print(f"  Processing {model_name}...")
        # List files via HuggingFace API
        api_url = (
            f"https://huggingface.co/api/datasets/McGill-NLP/agent-reward-bench/"
            f"tree/main/{path}"
        )
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req) as resp:
            files = json.loads(resp.read())
            json_files = [f for f in files if f["type"] == "file"
                         and f["path"].endswith(".json")]

        model_results = {}
        for file_info in json_files:
            filepath = file_info["path"]
            filename = filepath.split("/")[-1]
            match = re.search(r"\.(\d+)\.json$", filename)
            if not match:
                continue
            task_id = int(match.group(1))

            download_url = (
                f"https://huggingface.co/datasets/McGill-NLP/agent-reward-bench/"
                f"resolve/main/{filepath}"
            )
            req = urllib.request.Request(
                download_url, headers={"Range": "bytes=0-5000"}
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    content = resp.read().decode("utf-8", errors="ignore")
                    reward_match = re.search(r'"cum_reward":\s*([\d.]+)', content)
                    if reward_match:
                        model_results[task_id] = float(reward_match.group(1))
            except Exception as e:
                print(f"    WARNING: Failed to download {filename}: {e}")
            time.sleep(0.05)

        all_results[model_name] = model_results
        passed = sum(1 for v in model_results.values() if v > 0)
        print(f"    {len(model_results)} tasks, {passed} passed "
              f"({100 * passed / max(len(model_results), 1):.1f}%)")

    with open(cache_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Cached to {cache_path}")
    return all_results


# ============================================================
# Step 3: Load AgentRewardBench human annotations
# ============================================================
def load_human_annotations():
    """Load human-verified per-task pass/fail from annotations.csv."""
    csv_path = RAW_DIR / "annotations.csv"

    if not csv_path.exists():
        print("Downloading AgentRewardBench annotations.csv...")
        url = ("https://huggingface.co/datasets/McGill-NLP/agent-reward-bench/"
               "raw/main/data/annotations.csv")
        urllib.request.urlretrieve(url, str(csv_path))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        vwa_rows = [r for r in reader if r["benchmark"] == "visualwebarena"]

    print(f"Loaded {len(vwa_rows)} VWA human annotations")

    # Map model names to standardized names
    model_map = {
        "GenericAgent-gpt-4o-2024-11-20": "GenericAgent-gpt-4o-2024-11-20",
        "GenericAgent-anthropic_claude-3.7-sonnet": "GenericAgent-claude-3.7-sonnet",
        "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct": "GenericAgent-qwen2.5-vl-72b",
    }

    # Aggregate annotations per (model, task_id)
    # Multiple annotators may have reviewed the same trajectory
    results = defaultdict(lambda: defaultdict(list))
    for row in vwa_rows:
        model = model_map.get(row["model_name"], row["model_name"])
        # Extract numeric task_id from e.g. "visualwebarena.resized.398"
        match = re.search(r"\.(\d+)$", row["task_id"])
        if match:
            task_id = int(match.group(1))
            success = 1 if row["trajectory_success"] == "Successful" else 0
            results[model][task_id].append(success)

    # Use majority vote across annotators
    final_results = {}
    for model in results:
        model_data = {}
        for task_id, votes in results[model].items():
            model_data[task_id] = 1 if sum(votes) > len(votes) / 2 else 0
        final_results[model] = model_data
        passed = sum(model_data.values())
        print(f"  {model}: {len(model_data)} tasks, {passed} passed "
              f"({100 * passed / max(len(model_data), 1):.1f}%)")

    return final_results


# ============================================================
# Step 4: Parse leaderboard data
# ============================================================
def load_leaderboard():
    """Load aggregate success rates from the VWA Google Sheets leaderboard."""
    csv_path = RAW_DIR / "leaderboard.csv"

    if not csv_path.exists():
        print("Downloading VWA leaderboard from Google Sheets...")
        url = ("https://docs.google.com/spreadsheets/d/"
               "1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ/"
               "gviz/tq?tqx=out:csv&gid=2044883967")
        urllib.request.urlretrieve(url, str(csv_path))

    entries = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 5 or not row[4].strip():
                continue
            try:
                sr = float(row[4].strip())
            except ValueError:
                continue

            release_date = row[0].strip() if row[0].strip() else "N/A"
            model_type = row[1].strip() if len(row) > 1 else ""
            model = row[2].strip() if len(row) > 2 else ""
            inputs = row[3].strip() if len(row) > 3 else ""
            source = row[5].strip() if len(row) > 5 else ""

            if not model:
                # Human performance row
                model = "Human"
                model_type = "Human"

            entries.append({
                "release_date": release_date,
                "model_type": model_type,
                "model": model,
                "inputs": inputs,
                "success_rate_pct": sr,
                "result_source": source,
            })

    # Save leaderboard
    out_path = PROCESSED_DIR / "leaderboard_aggregate.csv"
    if entries:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entries[0].keys())
            writer.writeheader()
            writer.writerows(entries)

    print(f"Loaded {len(entries)} leaderboard entries, saved to {out_path}")
    return entries


# ============================================================
# Step 5: Load per-domain results from paper Table 3
# ============================================================
def get_paper_per_domain_results():
    """Return per-domain success rates from the original VWA paper (Table 3).
    These are aggregate per-domain, not per-task."""
    paper_results = [
        # (model_name, model_type, classifieds%, reddit%, shopping%, overall%)
        ("LLaMA-2-70B [text]", "Text-only", 0.43, 1.43, 1.29, 1.10),
        ("Mixtral-8x7B [text]", "Text-only", 1.71, 2.86, 1.29, 1.76),
        ("Gemini-Pro [text]", "Text-only", 0.85, 0.95, 3.43, 2.20),
        ("GPT-3.5 [text]", "Text-only", 0.43, 0.95, 3.65, 2.20),
        ("GPT-4 [text]", "Text-only", 5.56, 4.76, 9.23, 7.25),
        ("LLaMA-2-70B+BLIP2 [caption]", "Caption-augmented", 0.00, 0.95, 0.86, 0.66),
        ("Mixtral-8x7B+BLIP2 [caption]", "Caption-augmented", 1.28, 0.48, 2.79, 1.87),
        ("GPT-3.5+LLaVA [caption]", "Caption-augmented", 1.28, 1.43, 4.08, 2.75),
        ("GPT-3.5+BLIP2 [caption]", "Caption-augmented", 0.85, 1.43, 4.72, 2.97),
        ("Gemini-Pro+BLIP2 [caption]", "Caption-augmented", 1.71, 1.43, 6.01, 3.85),
        ("GPT-4+BLIP2 [caption]", "Caption-augmented", 8.55, 8.57, 16.74, 12.75),
        ("IDEFICS-80B [multimodal]", "Multimodal", 0.43, 0.95, 0.86, 0.77),
        ("CogVLM [multimodal]", "Multimodal", 0.00, 0.48, 0.43, 0.33),
        ("Gemini-Pro [multimodal]", "Multimodal", 3.42, 4.29, 8.15, 6.04),
        ("GPT-4V [multimodal]", "Multimodal", 8.12, 12.38, 19.74, 15.05),
        ("IDEFICS-80B+SoM [multimodal]", "Multimodal (SoM)", 0.85, 0.95, 1.07, 0.99),
        ("CogVLM+SoM [multimodal]", "Multimodal (SoM)", 0.00, 0.48, 0.43, 0.33),
        ("Gemini-Pro+SoM [multimodal]", "Multimodal (SoM)", 3.42, 3.81, 7.73, 5.71),
        ("GPT-4V+SoM [multimodal]", "Multimodal (SoM)", 9.83, 17.14, 19.31, 16.37),
        ("LLaMA-3-70B+BLIP2 [caption]", "Caption-augmented", 7.69, 5.24, 12.88, 9.78),
        ("Gemini-Flash-1.5+SoM [multimodal]", "Multimodal (SoM)", 3.85, 4.76, 8.80, 6.59),
        ("Gemini-Pro-1.5+SoM [multimodal]", "Multimodal (SoM)", 5.98, 12.86, 14.59, 11.98),
        ("GPT-4o+SoM [multimodal]", "Multimodal (SoM)", 20.51, 16.67, 20.82, 19.78),
    ]
    return paper_results


# ============================================================
# Step 6: Build the response matrix
# ============================================================
def build_response_matrix(
    all_configs,
    arb_rewards,
    human_annotations,
    leaderboard,
    paper_results,
):
    """Build the final response matrix: 910 tasks x N models."""
    n_tasks = len(all_configs)

    # Column names for per-task data (from AgentRewardBench)
    arb_models = sorted(arb_rewards.keys())

    # For each model, create two columns: automated reward and human annotation
    columns = []
    model_data = {}

    # AgentRewardBench automated rewards
    for model in arb_models:
        col_auto = f"{model} [auto]"
        columns.append(col_auto)
        model_data[col_auto] = {}
        for task_id_str, reward in arb_rewards[model].items():
            task_id = int(task_id_str)
            model_data[col_auto][task_id] = 1 if reward > 0 else 0

    # Human annotations (may disagree with automated)
    for model in arb_models:
        if model in human_annotations and human_annotations[model]:
            col_human = f"{model} [human]"
            columns.append(col_human)
            model_data[col_human] = {}
            for task_id, success in human_annotations[model].items():
                model_data[col_human][task_id] = success

    # Build the matrix
    matrix_rows = []
    for config in all_configs:
        task_id = config["task_id"]
        row = {"task_id": task_id}
        for col in columns:
            row[col] = model_data[col].get(task_id, "")
        matrix_rows.append(row)

    # Write CSV
    out_path = PROCESSED_DIR / "response_matrix.csv"
    fieldnames = ["task_id"] + columns
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"\nSaved response matrix: {out_path}")
    return matrix_rows, columns


# ============================================================
# Step 7: Print summary statistics
# ============================================================
def print_summary(matrix_rows, columns, all_configs, leaderboard, paper_results):
    """Print comprehensive summary statistics."""
    n_tasks = len(matrix_rows)
    n_models = len(columns)

    print("\n" + "=" * 70)
    print("VISUALWEBARENA RESPONSE MATRIX - SUMMARY REPORT")
    print("=" * 70)

    print(f"\n--- Matrix Dimensions ---")
    print(f"  Rows (tasks): {n_tasks}")
    print(f"  Columns (model configs with per-task data): {n_models}")

    # Fill rate
    total_cells = n_tasks * n_models
    filled_cells = sum(
        1 for row in matrix_rows for col in columns if row[col] != ""
    )
    fill_rate = 100 * filled_cells / total_cells if total_cells > 0 else 0
    print(f"  Total cells: {total_cells}")
    print(f"  Filled cells: {filled_cells}")
    print(f"  Fill rate: {fill_rate:.1f}%")

    print(f"\n--- Score Type ---")
    print(f"  Binary: 1 = pass (task completed), 0 = fail (task not completed)")
    print(f"  Source [auto]: BrowserGym automated evaluation (cum_reward)")
    print(f"  Source [human]: AgentRewardBench human annotation (trajectory_success)")

    print(f"\n--- Model Names (per-task data available) ---")
    for col in columns:
        # Count filled and passed
        filled = sum(1 for row in matrix_rows if row[col] != "")
        passed = sum(1 for row in matrix_rows if row[col] == 1)
        sr = 100 * passed / filled if filled > 0 else 0
        print(f"  {col}: {filled}/{n_tasks} tasks evaluated, "
              f"{passed} passed ({sr:.1f}%)")

    print(f"\n--- Task Coverage ---")
    # Domain distribution
    domain_counts = Counter()
    for config in all_configs:
        domain_counts[config["sites"][0]] += 1
    print(f"  Tasks by domain:")
    for domain, count in domain_counts.most_common():
        print(f"    {domain}: {count} tasks ({100*count/n_tasks:.1f}%)")

    # Difficulty distribution
    diff_counts = Counter()
    for config in all_configs:
        diff_counts[config.get("overall_difficulty", "unknown")] += 1
    print(f"  Tasks by difficulty:")
    for diff, count in diff_counts.most_common():
        print(f"    {diff}: {count} tasks ({100*count/n_tasks:.1f}%)")

    # Eval type distribution
    eval_counts = Counter()
    for config in all_configs:
        for et in config["eval"]["eval_types"]:
            eval_counts[et] += 1
    print(f"  Eval types (tasks can have multiple):")
    for et, count in eval_counts.most_common():
        print(f"    {et}: {count}")

    print(f"\n--- Leaderboard Aggregate Results ({len(leaderboard)} entries) ---")
    print(f"  Top 5 models:")
    sorted_lb = sorted(leaderboard, key=lambda x: x["success_rate_pct"], reverse=True)
    for entry in sorted_lb[:5]:
        print(f"    {entry['model']} ({entry['model_type']}): "
              f"{entry['success_rate_pct']:.1f}%")
    print(f"  ...")
    print(f"  Lowest: {sorted_lb[-1]['model']} ({sorted_lb[-1]['model_type']}): "
          f"{sorted_lb[-1]['success_rate_pct']:.1f}%")
    print(f"  Human performance: 88.7%")

    print(f"\n--- Paper Per-Domain Breakdown (23 model configs) ---")
    print(f"  Available domains: classifieds, reddit, shopping")
    print(f"  Example: GPT-4V+SoM [multimodal]: "
          f"classifieds=9.83%, reddit=17.14%, shopping=19.31%, overall=16.37%")

    print(f"\n--- Data Sources ---")
    print(f"  1. AgentRewardBench (HuggingFace): per-task for 3 models x 100 tasks")
    print(f"  2. AgentRewardBench annotations: human labels for ~200 unique tasks")
    print(f"  3. VWA leaderboard (Google Sheets): aggregate for {len(leaderboard)} models")
    print(f"  4. VWA paper (arXiv:2401.13649): per-domain for 23 model configs")
    print(f"  5. BrowserGym test_raw.json: full 910 task metadata")

    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Building VisualWebArena Response Matrix")
    print("=" * 70)

    # Step 1: Task metadata
    print("\n[1/6] Loading task metadata...")
    all_configs = load_task_metadata()
    task_metadata = save_task_metadata(all_configs)

    # Step 2: AgentRewardBench automated rewards
    print("\n[2/6] Loading AgentRewardBench automated rewards...")
    arb_rewards = load_agentrewardbench_rewards()

    # Step 3: Human annotations
    print("\n[3/6] Loading human annotations...")
    human_annotations = load_human_annotations()

    # Step 4: Leaderboard
    print("\n[4/6] Loading leaderboard aggregate data...")
    leaderboard = load_leaderboard()

    # Step 5: Paper per-domain results
    print("\n[5/6] Loading paper per-domain results...")
    paper_results = get_paper_per_domain_results()
    print(f"  {len(paper_results)} model configs with per-domain breakdowns")

    # Step 6: Build response matrix
    print("\n[6/6] Building response matrix...")
    matrix_rows, columns = build_response_matrix(
        all_configs, arb_rewards, human_annotations, leaderboard, paper_results
    )

    # Summary
    print_summary(matrix_rows, columns, all_configs, leaderboard, paper_results)


if __name__ == "__main__":
    main()
