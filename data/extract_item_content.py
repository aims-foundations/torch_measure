#!/usr/bin/env python3
"""
Extract item_content.csv for benchmarks that have raw/processed data
but no standardized item content file.

Each benchmark's item_content.csv has two columns:
  - item_id: unique identifier for the item
  - content: the text content of the item (question, instruction, task description)

Usage:
  python extract_item_content.py                    # extract all
  python extract_item_content.py terminal_bench     # extract one benchmark
  python extract_item_content.py --list             # list available extractors

The script is idempotent: re-running will overwrite existing item_content.csv files.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

# Auto-detect data directory
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR.parent / "torch_measure").exists():
    # Running from torch_measure repo root
    DATA_DIR = SCRIPT_DIR.parent
elif (SCRIPT_DIR / "torch_measure").exists():
    DATA_DIR = SCRIPT_DIR / "torch_measure" / "data"
else:
    # Default: assume we're in the data directory
    DATA_DIR = SCRIPT_DIR

# Override with environment variable if set
import os
DATA_DIR = Path(os.environ.get("TORCH_MEASURE_DATA", DATA_DIR))


def save_item_content(bench_dir: str, items: list[dict]) -> int:
    """Save item_content.csv and return count."""
    out_path = DATA_DIR / bench_dir / "processed" / "item_content.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(items)
    df.to_csv(out_path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Extractors — one function per benchmark
# ---------------------------------------------------------------------------

def extract_terminal_bench() -> int:
    """Terminal-Bench: task instruction from complete metadata."""
    meta_path = DATA_DIR / "terminal_bench_data/processed/tasks_complete_metadata.csv"
    if not meta_path.exists():
        return 0
    meta = pd.read_csv(meta_path)
    items = [
        {"item_id": row["task_name"], "content": str(row["instruction"])[:2000]}
        for _, row in meta.iterrows()
        if pd.notna(row.get("instruction"))
    ]
    return save_item_content("terminal_bench_data", items)


def extract_livecodebench() -> int:
    """LiveCodeBench: question title + content from submission eval JSONs."""
    sub_dir = DATA_DIR / "livecodebench_data/raw/submissions"
    if not sub_dir.exists():
        return 0

    items = {}
    for model_dir in sorted(sub_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for ef in model_dir.glob("*.json"):
            try:
                with open(ef) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for entry in data:
                qid = entry.get("question_id", "")
                if qid and qid not in items:
                    title = entry.get("question_title", "")
                    content = entry.get("question_content", "")
                    text = f"{title}\n{content}"[:2000] if content else title
                    if len(text) > 10:
                        items[qid] = {"item_id": qid, "content": text}
        if len(items) >= 1200:
            break

    return save_item_content("livecodebench_data", list(items.values()))


def extract_alpacaeval() -> int:
    """AlpacaEval: instruction text from item_metadata.csv."""
    meta_path = DATA_DIR / "alpacaeval_data/processed/item_metadata.csv"
    if not meta_path.exists():
        return 0
    meta = pd.read_csv(meta_path)
    items = [
        {"item_id": str(row.get("item_idx", i)), "content": str(row["instruction"])[:2000]}
        for i, (_, row) in enumerate(meta.iterrows())
        if pd.notna(row.get("instruction"))
    ]
    return save_item_content("alpacaeval_data", items)


def extract_wildbench() -> int:
    """WildBench: intent + primary_tag from task_metadata.csv."""
    meta_path = DATA_DIR / "wildbench_data/raw/task_metadata.csv"
    if not meta_path.exists():
        return 0
    meta = pd.read_csv(meta_path)
    items = []
    for _, row in meta.iterrows():
        text = str(row.get("intent", ""))
        if row.get("primary_tag"):
            text = f"[{row['primary_tag']}] {text}"
        if len(text) > 10:
            items.append({
                "item_id": str(row.get("session_id", "")),
                "content": text,
            })
    return save_item_content("wildbench_data", items)


def extract_corebench() -> int:
    """CORE-Bench: capsule title + field + language from task_metadata.csv."""
    meta_path = DATA_DIR / "corebench_data/processed/task_metadata.csv"
    if not meta_path.exists():
        return 0
    meta = pd.read_csv(meta_path)
    items = []
    for _, row in meta.iterrows():
        parts = []
        if pd.notna(row.get("capsule_title")):
            parts.append(str(row["capsule_title"]))
        if pd.notna(row.get("field")):
            parts.append(f"Field: {row['field']}")
        if pd.notna(row.get("language")):
            parts.append(f"Language: {row['language']}")
        if parts:
            items.append({
                "item_id": str(row["task_id"]),
                "content": " | ".join(parts),
            })
    return save_item_content("corebench_data", items)


def extract_editbench() -> int:
    """EditBench: instruction preview + language from task_metadata.csv."""
    meta_path = DATA_DIR / "editbench_data/processed/task_metadata.csv"
    if not meta_path.exists():
        return 0
    meta = pd.read_csv(meta_path)
    items = []
    for _, row in meta.iterrows():
        text = str(row.get("instruction_preview", ""))
        if pd.notna(row.get("programming_language")):
            text = f"[{row['programming_language']}] {text}"
        if pd.notna(row.get("natural_language")) and str(row["natural_language"]).lower() != "english":
            text = f"({row['natural_language']}) {text}"
        if len(text) > 10:
            items.append({"item_id": str(row["task_id"]), "content": text})
    return save_item_content("editbench_data", items)


def extract_afrimedqa() -> int:
    """AfriMedQA: question text + answer options from raw CSV."""
    csv_path = DATA_DIR / "afrimedqa_data/raw/AfriMed-QA/data/afri_med_qa_15k_v2.5_phase_2_15275.csv"
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row.get("question_clean")):
            parts.append(str(row["question_clean"]))
        elif pd.notna(row.get("question")):
            parts.append(str(row["question"]))
        if pd.notna(row.get("answer_options")):
            parts.append(str(row["answer_options"])[:500])
        if parts:
            items.append({
                "item_id": str(row.get("sample_id", "")),
                "content": "\n".join(parts)[:2000],
            })
    return save_item_content("afrimedqa_data", items)


def extract_cybench() -> int:
    """CyBench: task names from leaderboard columns."""
    lb_path = DATA_DIR / "cybench_data/raw/leaderboard.csv"
    if not lb_path.exists():
        return 0
    lb = pd.read_csv(lb_path)
    # First column is model name, rest are task names
    items = [
        {"item_id": col, "content": f"CTF Challenge: {col}"}
        for col in lb.columns[1:]
    ]
    return save_item_content("cybench_data", items)


def extract_dpai() -> int:
    """DPAI Arena: task IDs from long-format results."""
    results_path = DATA_DIR / "dpai_data/processed/all_results_long_format.csv"
    if not results_path.exists():
        return 0
    df = pd.read_csv(results_path)
    if "task_id" not in df.columns:
        return 0
    task_ids = df["task_id"].unique()
    items = [
        {"item_id": str(tid), "content": f"DPAI Java SE Task: {tid}"}
        for tid in task_ids
    ]
    return save_item_content("dpai_data", items)


def extract_sib200() -> int:
    """SIB-200: topic classification text across 200+ languages."""
    annotated_dir = DATA_DIR / "sib200_data/raw/sib-200/data/annotated"
    if not annotated_dir.exists():
        return 0

    items = []
    for lang_dir in sorted(annotated_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        test_file = lang_dir / "test.tsv"
        if not test_file.exists():
            continue
        try:
            df = pd.read_csv(test_file, sep="\t")
            lang = lang_dir.name
            for _, row in df.iterrows():
                text = str(row.get("text", ""))
                cat = str(row.get("category", ""))
                idx = str(row.get("index_id", ""))
                if len(text) > 10:
                    items.append({
                        "item_id": lang + "_" + idx,
                        "content": "[" + lang + "] [" + cat + "] " + text[:500],
                    })
        except Exception:
            continue

    return save_item_content("sib200_data", items)


def extract_helm_multilingual() -> int:
    """HELM Multilingual: benchmark + subject + language from task_metadata.csv.

    Note: this is metadata-only (no actual question text). Embeddings will
    reflect category labels, not item semantics.
    """
    meta_path = DATA_DIR / "helm_multilingual_data/processed/task_metadata.csv"
    if not meta_path.exists():
        return 0
    meta = pd.read_csv(meta_path)
    items = []
    for _, row in meta.iterrows():
        parts = []
        if pd.notna(row.get("benchmark")):
            parts.append(str(row["benchmark"]))
        if pd.notna(row.get("subject")):
            parts.append(str(row["subject"]))
        if pd.notna(row.get("language")):
            parts.append("lang:" + str(row["language"]))
        content = " | ".join(parts)
        if len(content) > 5:
            items.append({"item_id": str(row["item_id"]), "content": content})
    return save_item_content("helm_multilingual_data", items)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "terminal_bench": extract_terminal_bench,
    "livecodebench": extract_livecodebench,
    "alpacaeval": extract_alpacaeval,
    "wildbench": extract_wildbench,
    "corebench": extract_corebench,
    "editbench": extract_editbench,
    "afrimedqa": extract_afrimedqa,
    "cybench": extract_cybench,
    "dpai": extract_dpai,
    "sib200": extract_sib200,
    "helm_multilingual": extract_helm_multilingual,
}


def main():
    parser = argparse.ArgumentParser(description="Extract item_content.csv for benchmarks")
    parser.add_argument("benchmarks", nargs="*", help="Benchmarks to extract (default: all)")
    parser.add_argument("--list", action="store_true", help="List available extractors")
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    args = parser.parse_args()

    if args.data_dir:
        global DATA_DIR
        DATA_DIR = Path(args.data_dir)

    if args.list:
        print("Available extractors:")
        for name in sorted(EXTRACTORS):
            print(f"  {name}")
        return

    targets = args.benchmarks if args.benchmarks else list(EXTRACTORS.keys())

    print(f"Data directory: {DATA_DIR}")
    print()

    total = 0
    for name in targets:
        if name not in EXTRACTORS:
            print(f"[ERROR] Unknown benchmark: {name}")
            print(f"  Available: {', '.join(sorted(EXTRACTORS))}")
            sys.exit(1)

        fn = EXTRACTORS[name]
        n = fn()
        status = f"{n:,} items" if n > 0 else "no data found"
        print(f"  {name:25s}: {status}")
        total += n

    print(f"\nTotal: {total:,} items extracted")


if __name__ == "__main__":
    main()
