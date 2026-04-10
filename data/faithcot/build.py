"""
build.py — Build FaithCoT-BENCH response matrix.

Source: https://github.com/se7esx/FaithCoT-BENCH

The repository ships a zip (faithcot.zip) containing per-model, per-item
chain-of-thought responses on four task suites (AQuA, LogiQA, TruthfulQA,
HLE_BIO). Each response JSON file has fields `question`, `label`,
`sample_0.parsed_final_answer`, and `unfaithfulness` ∈ {0, 1}. We load all
responses and build a per-item matrix.

Two response matrices are produced:
  1. response_matrix.csv                  — faithfulness (1 = faithful)
  2. response_matrix_correct.csv          — correctness  (1 = correct answer)

Both have:
  - rows = 4 models (gpt-4o-mini, gemini-2.5-flash, Qwen2.5-7B-Instruct,
                     llama-3.1-8b-instruct)
  - columns = items, identified as "{task}__{index}"
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download() -> Path:
    """Clone the FaithCoT-BENCH repo and extract the bundled zip."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = RAW_DIR / "FaithCoT-BENCH"
    if not repo_dir.exists():
        print("Cloning FaithCoT-BENCH repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/se7esx/FaithCoT-BENCH.git", str(repo_dir)],
            check=True,
        )

    # Extract zip if needed
    zip_path = repo_dir / "faithcot.zip"
    extract_dir = PROCESSED_DIR / "extracted"
    marker = extract_dir / ".extracted"
    if zip_path.exists() and not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        marker.touch()
    return extract_dir / "faithcot"


def load_responses(data_root: Path):
    """Load all per-task, per-model, per-item responses.

    Yields (task, model, item_index, {correct, faithful}).
    """
    if not data_root.exists():
        raise FileNotFoundError(f"FaithCoT data root not found: {data_root}")

    tasks = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    print(f"Tasks found: {tasks}")

    for task in tasks:
        task_dir = data_root / task
        models = sorted([m.name for m in task_dir.iterdir() if m.is_dir()])
        for model in models:
            model_dir = task_dir / model
            response_files = sorted(model_dir.glob("response_*.json"))
            for rf in response_files:
                try:
                    idx = int(rf.stem.split("_")[-1])
                except ValueError:
                    continue
                try:
                    with open(rf) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [WARN] Skipping {rf}: {e}")
                    continue

                unfaithful = data.get("unfaithfulness")
                # Some files use 'sample_0', some 'sample_1' - prefer sample_0
                parsed = None
                label = data.get("label")
                for key in ("sample_0", "sample_1"):
                    if key in data and isinstance(data[key], dict):
                        parsed = data[key].get("parsed_final_answer")
                        break

                correct = None
                if parsed is not None and label is not None:
                    correct = int(str(parsed).strip() == str(label).strip())

                faithful = None
                if unfaithful is not None:
                    try:
                        faithful = 1 - int(unfaithful)
                    except (TypeError, ValueError):
                        faithful = None

                yield task, model, idx, correct, faithful


def build_matrices(data_root: Path):
    """Build two response matrices: faithful and correct."""
    rows = list(load_responses(data_root))
    df = pd.DataFrame(rows, columns=["task", "model", "idx", "correct", "faithful"])
    df["item"] = df["task"] + "__" + df["idx"].astype(str)

    # Use items that appear in either matrix as the full column universe
    all_items = sorted(
        df["item"].unique(),
        key=lambda c: (c.split("__")[0], int(c.split("__")[1])),
    )

    # Response matrix: faithfulness
    rm_faith = (
        df.pivot_table(index="model", columns="item", values="faithful", aggfunc="first")
        .reindex(columns=all_items)
    )
    # Response matrix: correctness
    rm_correct = (
        df.pivot_table(index="model", columns="item", values="correct", aggfunc="first")
        .reindex(columns=all_items)
    )

    rm_faith.index.name = "model"
    rm_correct.index.name = "model"

    return rm_faith, rm_correct, df


def main():
    data_root = download()
    print(f"Data root: {data_root}")

    rm_faith, rm_correct, df = build_matrices(data_root)

    print("\n=== Faithfulness matrix ===")
    print(f"Shape: {rm_faith.shape}  (models x items)")
    print(rm_faith.mean(axis=1).to_string())

    print("\n=== Correctness matrix ===")
    print(f"Shape: {rm_correct.shape}  (models x items)")
    print(rm_correct.mean(axis=1).to_string())

    out_faith = PROCESSED_DIR / "response_matrix.csv"
    rm_faith.to_csv(out_faith)
    print(f"\nSaved faithfulness matrix to {out_faith}")

    out_correct = PROCESSED_DIR / "response_matrix_correct.csv"
    rm_correct.to_csv(out_correct)
    print(f"Saved correctness matrix to {out_correct}")

    # Summary
    summary = pd.DataFrame({
        "metric": ["n_models", "n_items",
                   "mean_faithful", "mean_correct",
                   "tasks"],
        "value": [
            rm_faith.shape[0], rm_faith.shape[1],
            rm_faith.values.mean() if rm_faith.size else 0,
            rm_correct.values.mean() if rm_correct.size else 0,
            ", ".join(sorted(df["task"].unique())),
        ],
    })
    summary.to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
