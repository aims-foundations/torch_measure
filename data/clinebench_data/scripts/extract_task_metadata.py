"""
Extract structured metadata from all cline-bench tasks.
Reads task.toml and instruction.md files, outputs a JSON summary.
"""

import os
import json
import re
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = str(_BENCHMARK_DIR / "raw" / "cline-bench" / "tasks")
OUTPUT_DIR = str(_BENCHMARK_DIR / "processed")

def parse_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def count_tests(tests_dir):
    """Count test files and try to count test functions."""
    if not os.path.isdir(tests_dir):
        return {"test_files": 0, "test_functions": 0}

    test_files = [f for f in os.listdir(tests_dir)
                  if f.endswith('.py') and f.startswith('test')]

    total_funcs = 0
    for tf in test_files:
        content = read_file(os.path.join(tests_dir, tf))
        if content:
            # Count def test_ functions
            total_funcs += len(re.findall(r'def test_', content))

    return {"test_files": len(test_files), "test_functions": total_funcs}

def extract_task(task_dir):
    task_id = os.path.basename(task_dir)

    # Parse task.toml
    toml_path = os.path.join(task_dir, "task.toml")
    toml_data = parse_toml(toml_path)

    # Read instruction.md
    instruction = read_file(os.path.join(task_dir, "instruction.md"))

    # Read README.md
    readme = read_file(os.path.join(task_dir, "README.md"))

    # Check solution
    solution_path = os.path.join(task_dir, "solution", "solve.sh")
    has_solution = os.path.exists(solution_path)
    solution_content = read_file(solution_path)

    # Count tests
    tests_dir = os.path.join(task_dir, "tests")
    test_info = count_tests(tests_dir)

    # Check environment
    env_dir = os.path.join(task_dir, "environment")
    has_dockerfile = os.path.exists(os.path.join(env_dir, "Dockerfile"))
    has_docker_compose = os.path.exists(os.path.join(env_dir, "docker-compose.yaml"))

    # Extract short name from task_id
    parts = task_id.split("-", 1)
    short_name = parts[1] if len(parts) > 1 else task_id

    metadata = toml_data.get("metadata", {})

    return {
        "task_id": task_id,
        "short_name": short_name,
        "difficulty": metadata.get("difficulty", "unknown"),
        "category": metadata.get("category", "unknown"),
        "tags": metadata.get("tags", []),
        "agent_timeout_sec": toml_data.get("agent", {}).get("timeout_sec", 0),
        "verifier_timeout_sec": toml_data.get("verifier", {}).get("timeout_sec", 0),
        "build_timeout_sec": toml_data.get("environment", {}).get("build_timeout_sec", 0),
        "cpus": toml_data.get("environment", {}).get("cpus", 0),
        "memory_mb": toml_data.get("environment", {}).get("memory_mb", 0),
        "storage_mb": toml_data.get("environment", {}).get("storage_mb", 0),
        "has_dockerfile": has_dockerfile,
        "has_docker_compose": has_docker_compose,
        "has_solution": has_solution,
        "instruction_length_chars": len(instruction) if instruction else 0,
        "test_files": test_info["test_files"],
        "test_functions": test_info["test_functions"],
        "custom_docker_compose": metadata.get("custom_docker_compose", False),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = []
    for entry in sorted(os.listdir(TASKS_DIR)):
        task_dir = os.path.join(TASKS_DIR, entry)
        if os.path.isdir(task_dir) and not entry.startswith('.'):
            try:
                task = extract_task(task_dir)
                tasks.append(task)
                print(f"  Extracted: {task['short_name']} ({task['difficulty']})")
            except Exception as e:
                print(f"  ERROR processing {entry}: {e}")

    # Save JSON
    output_path = os.path.join(OUTPUT_DIR, "task_metadata.json")
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"\nSaved {len(tasks)} tasks to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Short Name':<45} {'Difficulty':<10} {'Tests':<8} {'Agent TO':<10} {'Memory':<8}")
    print("=" * 100)
    for t in tasks:
        print(f"{t['short_name']:<45} {t['difficulty']:<10} {t['test_functions']:<8} "
              f"{t['agent_timeout_sec']:<10} {t['memory_mb']:<8}")

    # Difficulty distribution
    diff_counts = {}
    for t in tasks:
        d = t['difficulty']
        diff_counts[d] = diff_counts.get(d, 0) + 1
    print(f"\nDifficulty distribution: {diff_counts}")
    print(f"Total tasks: {len(tasks)}")

if __name__ == "__main__":
    main()
