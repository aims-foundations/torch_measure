#!/usr/bin/env python3
"""
Build response matrix and item content for ClineBench.

Source: https://github.com/cline/cline-bench
ClineBench evaluates AI coding agents on 12 real-world software engineering tasks.

Models evaluated:
  - Oracle (reference solution)
  - Terminus (AI agent)
  - Cline (AI agent, Claude Sonnet 4)

Outputs:
  - response_matrix.csv   : models (rows) x tasks (columns) -> continuous scores [0, 1]
  - item_content.csv       : task_id, instruction text from instruction.md
  - task_metadata.json     : per-task structured metadata (difficulty, category, tests, etc.)
"""

INFO = {
    'description': 'Build response matrix and item content for ClineBench',
    'testing_condition': '',
    'paper_url': '',
    'data_source_url': 'https://github.com/cline/cline-bench',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'unknown',
    'citation': '@misc{clinebench,\n  title={Clinebench},\n  howpublished={\\url{https://github.com/cline/cline-bench}},\n}',
    'tags': ['agent'],
}


import sys
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Download ─────────────────────────────────────────────────────────────

def download():
    """Clone the cline-bench repo if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = RAW_DIR / "cline-bench"
    if repo_dir.exists():
        print("cline-bench repo already cloned, skipping download")
        return
    print("Cloning cline-bench repo...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/cline/cline-bench.git", str(repo_dir)],
        check=True,
    )
    print(f"Done. Raw files in {repo_dir}")


# ── Task metadata extraction ────────────────────────────────────────────

def _read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _count_tests(tests_dir):
    """Count test files and test functions."""
    if not os.path.isdir(tests_dir):
        return {"test_files": 0, "test_functions": 0}
    test_files = [f for f in os.listdir(tests_dir)
                  if f.endswith(".py") and f.startswith("test")]
    total_funcs = 0
    for tf in test_files:
        content = _read_file(os.path.join(tests_dir, tf))
        if content:
            total_funcs += len(re.findall(r"def test_", content))
    return {"test_files": len(test_files), "test_functions": total_funcs}


def extract_task_metadata(tasks_dir):
    """Extract structured metadata from all cline-bench tasks."""
    tasks = []
    for entry in sorted(os.listdir(tasks_dir)):
        task_dir = os.path.join(tasks_dir, entry)
        if not os.path.isdir(task_dir) or entry.startswith("."):
            continue
        try:
            toml_path = os.path.join(task_dir, "task.toml")
            with open(toml_path, "rb") as f:
                toml_data = tomllib.load(f)

            instruction = _read_file(os.path.join(task_dir, "instruction.md"))
            test_info = _count_tests(os.path.join(task_dir, "tests"))
            env_dir = os.path.join(task_dir, "environment")
            metadata = toml_data.get("metadata", {})
            parts = entry.split("-", 1)

            tasks.append({
                "task_id": parts[0],
                "short_name": parts[1] if len(parts) > 1 else entry,
                "difficulty": metadata.get("difficulty", "unknown"),
                "category": metadata.get("category", "unknown"),
                "tags": metadata.get("tags", []),
                "agent_timeout_sec": toml_data.get("agent", {}).get("timeout_sec", 0),
                "verifier_timeout_sec": toml_data.get("verifier", {}).get("timeout_sec", 0),
                "build_timeout_sec": toml_data.get("environment", {}).get("build_timeout_sec", 0),
                "cpus": toml_data.get("environment", {}).get("cpus", 0),
                "memory_mb": toml_data.get("environment", {}).get("memory_mb", 0),
                "storage_mb": toml_data.get("environment", {}).get("storage_mb", 0),
                "has_dockerfile": os.path.exists(os.path.join(env_dir, "Dockerfile")),
                "has_docker_compose": os.path.exists(os.path.join(env_dir, "docker-compose.yaml")),
                "has_solution": os.path.exists(os.path.join(task_dir, "solution", "solve.sh")),
                "instruction_length_chars": len(instruction) if instruction else 0,
                "test_files": test_info["test_files"],
                "test_functions": test_info["test_functions"],
                "custom_docker_compose": metadata.get("custom_docker_compose", False),
                "instruction": instruction or "",
            })
            print(f"  Extracted: {tasks[-1]['short_name']} ({tasks[-1]['difficulty']})")
        except Exception as e:
            print(f"  ERROR processing {entry}: {e}")

    return tasks


# ── Response matrix ──────────────────────────────────────────────────────

# Hardcoded results from the ClineBench blog post / README.
# Source: https://github.com/cline/cline-bench
RESULTS = {
    # task_id: {model: score}
    "01k6kr5hbv8za80v8vnze3at8h": {"oracle": 1.0, "terminus": 0.0, "cline": 0.815},
    "01k6n26zm27ffa7qqbcx0prrnw": {"oracle": 1.0, "terminus": 0.0, "cline": 1.0},
    "01k6rkmyfgbwpvf7h81gh4pdgd": {"oracle": 1.0, "terminus": 1.0},
    "01k6zz0nyj31znwsevx4sn6zb2": {"oracle": 1.0, "terminus": 0.71},
    "01k7a12sd1nk15j08e6x0x7v9e": {"oracle": 1.0, "terminus": 0.0},
    "01k7x8zyeg4nzx6ehdb0fg5gfx": {"oracle": 1.0, "cline": 0.80},
    "01k8251zmv88p0hztas8htr6hw": {"oracle": 1.0, "terminus": 0.0, "cline": 0.83},
    "01k8mwgj1z6kr0a7q59r6ek2ar": {"oracle": 1.0, "terminus": 0.875},
    "01k8tymr1s3ndn1rzsrzy6dnfm": {"oracle": 1.0, "terminus": 0.0},
    "01k8ywgx6x7swdcse588426wc5": {"oracle": 1.0, "terminus": 0.0},
    "01kavebeh9sq8w7veabgyqnksh": {"oracle": 1.0},
    "01kbb2wvw29szdjwcs76265t3k": {"oracle": 1.0, "terminus": 0.0},
}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    download()

    tasks_dir = RAW_DIR / "cline-bench" / "tasks"
    if not tasks_dir.exists():
        print(f"ERROR: Tasks directory not found: {tasks_dir}")
        return

    # Extract task metadata
    print("\nExtracting task metadata...")
    if tomllib is None:
        print("  WARNING: tomllib/tomli not available (needs Python 3.11+ or pip install tomli)")
        print("  Skipping TOML-based metadata extraction, using task directory names only.")
        tasks = []
        for entry in sorted(os.listdir(tasks_dir)):
            task_dir = tasks_dir / entry
            if task_dir.is_dir() and not entry.startswith("."):
                instruction = _read_file(str(task_dir / "instruction.md"))
                parts = entry.split("-", 1)
                tasks.append({
                    "task_id": parts[0],
                    "short_name": parts[1] if len(parts) > 1 else entry,
                    "difficulty": "unknown",
                    "category": "unknown",
                    "instruction": instruction or "",
                })
    else:
        tasks = extract_task_metadata(str(tasks_dir))

    # Save task metadata (without instruction text)
    meta_for_json = [{k: v for k, v in t.items() if k != "instruction"} for t in tasks]
    with open(OUTPUT_DIR / "task_metadata.json", "w") as f:
        json.dump(meta_for_json, f, indent=2)
    print(f"Saved task_metadata.json ({len(tasks)} tasks)")

    # Save item_content.csv
    content_rows = []
    for t in tasks:
        content_rows.append({
            "item_id": t["task_id"],
            "content": t["instruction"],
        })
    pd.DataFrame(content_rows).to_csv(OUTPUT_DIR / "item_content.csv", index=False)
    print(f"Saved item_content.csv ({len(content_rows)} items)")

    # Build response matrix (models as rows, tasks as columns)
    task_ids = [t["task_id"] for t in tasks]
    models = sorted({m for scores in RESULTS.values() for m in scores})

    matrix = pd.DataFrame(index=models, columns=task_ids, dtype=float)
    matrix.index.name = "model"
    for task_id, scores in RESULTS.items():
        for model, score in scores.items():
            if task_id in matrix.columns:
                matrix.loc[model, task_id] = score

    matrix.to_csv(OUTPUT_DIR / "response_matrix.csv")
    n_models, n_items = matrix.shape
    fill_rate = matrix.notna().sum().sum() / (n_models * n_items) * 100
    print(f"Saved response_matrix.csv ({n_models} models x {n_items} items, "
          f"{fill_rate:.0f}% fill rate)")

    # Summary
    print(f"\nDifficulty distribution:")
    for t in tasks:
        print(f"  {t['short_name']}: {t['difficulty']}")


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os, subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
