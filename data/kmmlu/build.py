#!/usr/bin/env python3
"""
Build KMMLU response matrix (models x items, binary correct/incorrect).

Data sources:
  1. HAERAE-HUB/KMMLU (HuggingFace)
     - 45 subject configs, test split (35,030 items total)
     - Columns: question, answer (int 0-3 mapping to A-D), A, B, C, D,
                Category, Human Accuracy

  2. daekeun-ml/evaluate-llm-on-korean-dataset (GitHub)
     - Per-item model predictions on the KMMLU test set
     - 30 result CSV files at results/[KMMLU] <model>.csv
     - Columns: category, answer, pred, response
     - Row order within each category matches the upstream HF test split
       (verified empirically across multiple models)

Outputs:
  - task_metadata.csv  : item_id, question, answer_key, category, config,
                         human_accuracy, split, source_dataset, language
  - item_content.csv   : item_id, full question + options text
  - model_summary.csv  : per-model aggregate statistics + human baseline
  - response_matrix.csv: models (rows) x items (columns), binary {0,1}
                         plus a 'human_baseline' pseudo-row
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_REPO_URL = "https://github.com/daekeun-ml/evaluate-llm-on-korean-dataset.git"
RESULTS_REPO_DIR = RAW_DIR / "evaluate-llm-on-korean-dataset"

KMMLU_CONFIGS = [
    "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance",
    "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
    "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics",
    "Education", "Electrical-Engineering", "Electronics-Engineering",
    "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing",
    "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
    "Information-Technology", "Interior-Architecture-and-Design", "Korean-History",
    "Law", "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering",
    "Marketing", "Materials-Engineering", "Mechanical-Engineering",
    "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology",
    "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering",
    "Real-Estate", "Refrigerating-Machinery", "Social-Welfare", "Taxation",
    "Telecommunications-and-Wireless-Technology", "Math",
]

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def clone_results_repo():
    """Clone the daekeun-ml eval repo that holds per-item prediction CSVs."""
    if RESULTS_REPO_DIR.exists():
        print(f"  Results repo already cloned: {RESULTS_REPO_DIR}")
        return RESULTS_REPO_DIR
    print(f"  Cloning {RESULTS_REPO_URL}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", RESULTS_REPO_URL, str(RESULTS_REPO_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}")
        sys.exit(1)
    return RESULTS_REPO_DIR


def load_kmmlu_test_rows():
    """Load KMMLU test split, produce item rows ordered by config then source order."""
    from datasets import load_dataset

    all_rows = []
    item_id = 0

    for cfg in KMMLU_CONFIGS:
        print(f"  Loading {cfg}...", end=" ", flush=True)
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", cfg, split="test")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        n_items = 0
        for row_idx, row in enumerate(ds):
            q = row["question"]
            a_key = ANSWER_MAP.get(row["answer"], str(row["answer"]))
            options_text = (
                f"A: {row['A']}\nB: {row['B']}\n"
                f"C: {row['C']}\nD: {row['D']}"
            )
            full_content = f"{q}\n\n{options_text}"
            human_acc = row.get("Human Accuracy", np.nan)
            if human_acc is None:
                human_acc = np.nan

            all_rows.append({
                "item_id": f"kmmlu_{item_id:06d}",
                "cfg_row_idx": row_idx,
                "config": cfg,
                "question_short": q[:200],
                "answer_key": a_key,
                "category": row.get("Category", cfg),
                "human_accuracy": human_acc,
                "split": "test",
                "source_dataset": "HAERAE-HUB/KMMLU",
                "language": "korean",
                "full_content": full_content,
            })
            item_id += 1
            n_items += 1

        print(f"{n_items} items")

    return all_rows


def build_category_canonical_map(cfgs):
    """Return a dict mapping category aliases (lower/snake) to canonical name."""
    mapping = {}
    for c in cfgs:
        mapping[c.lower().replace("-", "_")] = c
        mapping[c.lower()] = c
        mapping[c] = c
    return mapping


def load_model_results(results_dir: Path, test_rows: list[dict]) -> dict[str, list]:
    """Load per-item model predictions from the repo's results/ folder.

    Returns a dict {model_name: [0/1/NaN for each item in test_rows order]}.
    Models whose files don't cover all KMMLU items are skipped.
    """
    results_root = results_dir / "results"
    if not results_root.exists():
        print(f"  ERROR: results dir not found: {results_root}")
        return {}

    result_files = sorted(p for p in results_root.glob("[[]KMMLU[]] *.csv"))
    print(f"  Found {len(result_files)} KMMLU result files")

    # Build index: (config, rank_within_config) -> index in test_rows
    cfg_to_indices = {}
    for idx, row in enumerate(test_rows):
        cfg_to_indices.setdefault(row["config"], []).append(idx)

    expected_sizes = {c: len(v) for c, v in cfg_to_indices.items()}
    cat_canonical = build_category_canonical_map(KMMLU_CONFIGS)

    model_responses = {}
    n_total = len(test_rows)

    for f in result_files:
        model_name = f.stem.replace("[KMMLU] ", "").strip()
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"    SKIP {model_name}: read failed: {e}")
            continue

        if not {"category", "answer", "pred"}.issubset(df.columns):
            print(f"    SKIP {model_name}: missing columns")
            continue

        # Normalize category to canonical name
        df["category"] = df["category"].map(
            lambda c: cat_canonical.get(str(c).lower().replace("-", "_"), str(c))
        )
        bad_cats = set(df["category"].unique()) - set(KMMLU_CONFIGS)
        if bad_cats:
            print(f"    SKIP {model_name}: unknown cats {sorted(bad_cats)[:3]}")
            continue

        # Check per-category row counts
        sizes = df.groupby("category").size().to_dict()
        missing_cats = [c for c in KMMLU_CONFIGS if c not in sizes]
        if missing_cats:
            print(f"    SKIP {model_name}: missing cats {missing_cats[:3]}")
            continue
        bad_sizes = [
            (c, sizes[c], expected_sizes[c])
            for c in KMMLU_CONFIGS
            if sizes[c] != expected_sizes[c]
        ]
        if bad_sizes:
            # Some rows missing — skip to avoid misalignment.
            print(f"    SKIP {model_name}: size mismatch {bad_sizes[:2]}")
            continue

        # Build scores array aligned to test_rows order.
        scores = np.full(n_total, np.nan, dtype=np.float32)
        for cfg in KMMLU_CONFIGS:
            cfg_rows = df[df["category"] == cfg]
            cfg_indices = cfg_to_indices[cfg]
            for k, (_, r) in enumerate(cfg_rows.iterrows()):
                pred = str(r["pred"]).strip().upper() if pd.notna(r["pred"]) else ""
                gold = str(r["answer"]).strip().upper() if pd.notna(r["answer"]) else ""
                if pred and gold:
                    scores[cfg_indices[k]] = 1.0 if pred == gold else 0.0
                else:
                    scores[cfg_indices[k]] = 0.0

        n_ans = int(np.sum(~np.isnan(scores)))
        acc = float(np.nanmean(scores)) if n_ans > 0 else 0.0
        print(f"    {model_name}: acc={acc:.3f} ({n_ans}/{n_total})")
        model_responses[model_name] = scores.tolist()

    return model_responses


def main():
    print("=" * 70)
    print("KMMLU Response Matrix Builder")
    print("=" * 70)

    # Step 1: Load KMMLU test set from HuggingFace
    print("\nLoading KMMLU test set from HuggingFace...")
    test_rows = load_kmmlu_test_rows()
    if not test_rows:
        print("ERROR: No data loaded.")
        sys.exit(1)

    df = pd.DataFrame(test_rows)
    n_items = len(df)
    print(f"\nTotal test items: {n_items}")
    print(f"Configs: {df['config'].nunique()}")

    # Step 2: Clone daekeun-ml eval repo
    print("\nCloning Korean eval results repo...")
    results_repo = clone_results_repo()

    # Step 3: Extract per-item model predictions
    print("\nExtracting per-model predictions...")
    model_responses = load_model_results(results_repo, test_rows)

    if not model_responses:
        print("ERROR: No model responses could be loaded.")
        sys.exit(1)

    # Step 4: Write task_metadata.csv
    meta_cols = [
        "item_id", "question_short", "answer_key", "category", "config",
        "human_accuracy", "split", "source_dataset", "language",
    ]
    meta_df = df[meta_cols].copy()
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved task_metadata.csv: {meta_df.shape}")

    # Step 5: Write item_content.csv
    content_df = df[["item_id", "full_content"]].copy()
    content_df.columns = ["item_id", "content"]
    content_path = PROCESSED_DIR / "item_content.csv"
    content_df.to_csv(content_path, index=False)
    print(f"Saved item_content.csv: {content_df.shape}")

    # Step 6: Build response matrix (rows=models, cols=items)
    item_ids = df["item_id"].tolist()

    rows = {}
    for model_name, scores in model_responses.items():
        rows[model_name] = scores

    # Add human_baseline pseudo-row using per-item Human Accuracy
    if "human_accuracy" in df.columns:
        human = df["human_accuracy"].astype(float).tolist()
        rows["human_baseline"] = human

    rm_df = pd.DataFrame.from_dict(rows, orient="index", columns=item_ids)
    rm_df.index.name = "model"
    rm_path = PROCESSED_DIR / "response_matrix.csv"
    rm_df.to_csv(rm_path)
    print(f"Saved response_matrix.csv: {rm_df.shape}")

    # Step 7: Write model_summary.csv
    summary_rows = []
    for model_name in rows.keys():
        scores = np.array(rows[model_name], dtype=float)
        n_ans = int(np.sum(~np.isnan(scores)))
        acc = float(np.nanmean(scores)) if n_ans > 0 else 0.0
        summary_rows.append({
            "model": model_name,
            "source": (
                "HAERAE-HUB/KMMLU"
                if model_name == "human_baseline"
                else "daekeun-ml/evaluate-llm-on-korean-dataset"
            ),
            "overall_accuracy": acc,
            "n_items": n_ans,
            "notes": (
                "Per-item human accuracy from KMMLU dataset"
                if model_name == "human_baseline"
                else "Per-item predictions from public Korean eval repo"
            ),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model_summary.csv: {summary_df.shape}")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Total items: {n_items}")
    print(f"  Models:      {len(model_responses)} + human_baseline")
    print(f"  Language:    Korean")
    print("\nDone!")


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
