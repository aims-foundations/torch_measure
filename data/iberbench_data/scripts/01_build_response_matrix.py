#!/usr/bin/env python3
"""
Build IberBench and Latin American Spanish NLP benchmark item collection.

Data sources:
  1. IberBench (iberbench/iberbench_all on HuggingFace):
     - IroSvA irony detection (2019): Spain, Mexico, Cuba varieties
     - TASS-2020 sentiment analysis: Spain, Mexico, Peru, Costa Rica, Uruguay
     - MEX-A3T aggressiveness detection (2019): Mexico
  2. HOMO-MEX (jhovany/Homomex2024): Track 1 LGBT+phobia detection, Mexico
  3. TASS-2019 (mrm8488/tass-2019): Sentiment analysis, Spain

Outputs:
  - task_metadata.csv  : Item-level metadata (item_id, text, label, task,
                         language, language_variety, source_dataset, split)
  - item_content.csv   : Item ID + full text content
  - model_summary.csv  : Placeholder (no per-model results available)
  - collection_summary.txt : Human-readable summary of what was collected

NOTE ON RESPONSE MATRIX:
  No publicly available per-model per-item evaluation results exist for these
  benchmarks. To generate a response_matrix.csv, run target models through
  lm-eval-harness using IberBench's task configurations:
    https://github.com/iberbench/iberbench
  Then score each model's predictions against gold labels to produce a binary
  {0, 1} matrix (items x models). The task_metadata.csv produced here provides
  the gold labels and item IDs needed for that alignment.
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# IberBench dataset configs
# ──────────────────────────────────────────────────────────────────────
IBERBENCH_REPO = "iberbench/iberbench_all"

IBERBENCH_CONFIGS = {
    # IroSvA irony detection (binary: 0=not ironic, 1=ironic)
    "irosva_spain": {
        "config": "iberlef-irosva-irony_detection-2019-spanish-spain",
        "task": "irony_detection",
        "language_variety": "spain",
    },
    "irosva_mexico": {
        "config": "iberlef-irosva-irony_detection-2019-spanish-mexico",
        "task": "irony_detection",
        "language_variety": "mexico",
    },
    "irosva_cuba": {
        "config": "iberlef-irosva-irony_detection-2019-spanish-cuba",
        "task": "irony_detection",
        "language_variety": "cuba",
    },
    # TASS-2020 sentiment analysis (3 classes: 0=N, 1=NEU, 2=P)
    "tass2020_spain": {
        "config": "tass-tass-sentiment_analysis-2020-spanish-spain",
        "task": "sentiment_analysis",
        "language_variety": "spain",
    },
    "tass2020_mexico": {
        "config": "tass-tass-sentiment_analysis-2020-spanish-mexico",
        "task": "sentiment_analysis",
        "language_variety": "mexico",
    },
    "tass2020_peru": {
        "config": "tass-tass-sentiment_analysis-2020-spanish-peru",
        "task": "sentiment_analysis",
        "language_variety": "peru",
    },
    "tass2020_costa_rica": {
        "config": "tass-tass-sentiment_analysis-2020-spanish-costa_rica",
        "task": "sentiment_analysis",
        "language_variety": "costa_rica",
    },
    "tass2020_uruguay": {
        "config": "tass-tass-sentiment_analysis-2020-spanish-uruguay",
        "task": "sentiment_analysis",
        "language_variety": "uruguay",
    },
    # MEX-A3T aggressiveness detection (binary: 0=not aggressive, 1=aggressive)
    "mex_a3t": {
        "config": "iberlef-mex_a3t-aggressiveness_detection-2019-spanish-mexico",
        "task": "aggressiveness_detection",
        "language_variety": "mexico",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Part 1: Download and process IberBench datasets
# ──────────────────────────────────────────────────────────────────────
def load_iberbench_datasets():
    """Download all IberBench configs and return a list of row dicts."""
    from datasets import load_dataset

    print("=" * 70)
    print("PART 1: Downloading IberBench datasets")
    print("=" * 70)

    all_rows = []
    item_counter = 0

    for name, info in IBERBENCH_CONFIGS.items():
        config = info["config"]
        task = info["task"]
        variety = info["language_variety"]

        print(f"\n  Loading {name} ({config})...", end=" ")
        ds = load_dataset(IBERBENCH_REPO, config, split="train")
        n = len(ds)
        print(f"{n} rows")

        # Save raw CSV
        raw_path = RAW_DIR / f"iberbench_{name}.csv"
        ds.to_csv(str(raw_path))
        print(f"    Saved raw: {raw_path.name}")

        for row in ds:
            all_rows.append({
                "item_id": f"iberbench_{name}_{item_counter}",
                "text": row["text"],
                "label": str(row["label"]),
                "task": task,
                "language": row.get("language", "spanish"),
                "language_variety": row.get("language_variation", variety),
                "source_dataset": f"iberbench_{name}",
                "split": "train",
            })
            item_counter += 1

    print(f"\n  Total IberBench items: {len(all_rows)}")
    return all_rows


# ──────────────────────────────────────────────────────────────────────
# Part 2: Download and process HOMO-MEX
# ──────────────────────────────────────────────────────────────────────
def load_homomex():
    """Download HOMO-MEX Track 1 (binary polarity P/NP) from HuggingFace."""
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("PART 2: Downloading HOMO-MEX (jhovany/Homomex2024)")
    print("=" * 70)

    ds = load_dataset(
        "jhovany/Homomex2024",
        data_files="track_1_dev.csv",
        split="train",
    )
    n = len(ds)
    print(f"  Loaded {n} rows")
    print(f"  Columns: {ds.column_names}")

    # Save raw
    raw_path = RAW_DIR / "homomex_track1.csv"
    ds.to_csv(str(raw_path))
    print(f"  Saved raw: {raw_path.name}")

    # The dataset has 3 labels: P (phobic), NP (not phobic), NR (not relevant)
    # For binary polarity as specified, we keep P and NP; NR items are retained
    # but flagged, since they are part of the original benchmark.
    all_rows = []
    for i, row in enumerate(ds):
        label = str(row["label"]).strip()
        all_rows.append({
            "item_id": f"homomex_{i}",
            "text": row["content"],
            "label": label,
            "task": "lgbtphobia_detection",
            "language": "spanish",
            "language_variety": "mexico",
            "source_dataset": "homomex_track1",
            "split": "dev",
        })

    print(f"  Total HOMO-MEX items: {len(all_rows)}")

    # Label distribution
    from collections import Counter
    label_dist = Counter(r["label"] for r in all_rows)
    print(f"  Label distribution: {dict(label_dist)}")

    return all_rows


# ──────────────────────────────────────────────────────────────────────
# Part 3: Download and process TASS-2019
# ──────────────────────────────────────────────────────────────────────
def load_tass2019():
    """Download TASS-2019 from mrm8488/tass-2019."""
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("PART 3: Downloading TASS-2019 (mrm8488/tass-2019)")
    print("=" * 70)

    all_rows = []
    item_counter = 0

    for split_name in ["train", "test"]:
        print(f"\n  Loading {split_name} split...", end=" ")
        ds = load_dataset("mrm8488/tass-2019", split=split_name)
        n = len(ds)
        print(f"{n} rows")
        print(f"  Columns: {ds.column_names}")

        # Save raw
        raw_path = RAW_DIR / f"tass2019_{split_name}.csv"
        ds.to_csv(str(raw_path))
        print(f"  Saved raw: {raw_path.name}")

        for row in ds:
            # Train split has 'sentiments' (N, P, NEU, NONE) and 'labels' (0-3)
            # Test split has labels=-1 and sentiments=None (no gold labels)
            sentiment = row.get("sentiments")
            label_int = row.get("labels")

            if sentiment is not None and sentiment != "":
                label = str(sentiment)
            elif label_int is not None and label_int >= 0:
                label_map = {0: "N", 1: "P", 2: "NEU", 3: "NONE"}
                label = label_map.get(label_int, str(label_int))
            else:
                label = "UNLABELED"

            all_rows.append({
                "item_id": f"tass2019_{item_counter}",
                "text": row["sentence"],
                "label": label,
                "task": "sentiment_analysis",
                "language": "spanish",
                "language_variety": "spain",
                "source_dataset": "tass2019",
                "split": split_name,
            })
            item_counter += 1

    print(f"\n  Total TASS-2019 items: {len(all_rows)}")

    from collections import Counter
    label_dist = Counter(r["label"] for r in all_rows)
    print(f"  Label distribution: {dict(label_dist)}")

    return all_rows


# ──────────────────────────────────────────────────────────────────────
# Build and save outputs
# ──────────────────────────────────────────────────────────────────────
def build_and_save(all_rows):
    """Build task_metadata.csv, item_content.csv, and model_summary.csv."""
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    df = pd.DataFrame(all_rows)

    # ── task_metadata.csv ──
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"\n  Saved task_metadata.csv: {meta_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # ── item_content.csv (matches mmlupro_data convention) ──
    item_content = df[["item_id", "text"]].rename(columns={"text": "content"})
    content_path = PROCESSED_DIR / "item_content.csv"
    item_content.to_csv(content_path, index=False)
    print(f"\n  Saved item_content.csv: {content_path}")
    print(f"    Shape: {item_content.shape}")

    # ── model_summary.csv (placeholder) ──
    model_summary = pd.DataFrame(columns=[
        "model", "source", "overall_accuracy", "n_items_evaluated",
        "notes",
    ])
    summary_path = PROCESSED_DIR / "model_summary.csv"
    model_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved model_summary.csv (placeholder): {summary_path}")

    return df


def print_collection_summary(df):
    """Print and save a human-readable summary of the collection."""
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)

    summary_lines = []

    def log(line=""):
        print(f"  {line}")
        summary_lines.append(line)

    log(f"Total items collected: {len(df)}")
    log()

    # By source dataset
    log("Items by source dataset:")
    for ds_name in sorted(df["source_dataset"].unique()):
        subset = df[df["source_dataset"] == ds_name]
        n = len(subset)
        labels = sorted(subset["label"].unique())
        variety = subset["language_variety"].iloc[0]
        task = subset["task"].iloc[0]
        log(f"  {ds_name:40s}  n={n:>5d}  variety={variety:<12s}  "
            f"task={task:<26s}  labels={labels}")

    log()

    # By task
    log("Items by task:")
    for task in sorted(df["task"].unique()):
        n = len(df[df["task"] == task])
        varieties = sorted(df[df["task"] == task]["language_variety"].unique())
        log(f"  {task:30s}  n={n:>5d}  varieties={varieties}")

    log()

    # By language variety
    log("Items by language variety:")
    for variety in sorted(df["language_variety"].unique()):
        n = len(df[df["language_variety"] == variety])
        tasks = sorted(df[df["language_variety"] == variety]["task"].unique())
        log(f"  {variety:15s}  n={n:>5d}  tasks={tasks}")

    log()

    # By split
    log("Items by split:")
    for split in sorted(df["split"].unique()):
        n = len(df[df["split"] == split])
        log(f"  {split:10s}  n={n:>5d}")

    log()
    log("NOTE: No response_matrix.csv generated. Per-model per-item results")
    log("are not publicly available. Run models through lm-eval-harness with")
    log("IberBench task configs to produce evaluation results, then build the")
    log("response matrix from those outputs.")

    # Save summary to file
    summary_path = PROCESSED_DIR / "collection_summary.txt"
    with open(summary_path, "w") as f:
        f.write("IberBench & Latin American Spanish NLP Benchmark Collection\n")
        f.write("=" * 60 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\n  Saved collection_summary.txt: {summary_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("IberBench & Latin American Spanish NLP Benchmark Builder")
    print("=" * 58 + "\n")

    # Part 1: IberBench datasets
    iberbench_rows = load_iberbench_datasets()

    # Part 2: HOMO-MEX
    homomex_rows = load_homomex()

    # Part 3: TASS-2019
    tass2019_rows = load_tass2019()

    # Combine all
    all_rows = iberbench_rows + homomex_rows + tass2019_rows
    print(f"\n  Combined total items: {len(all_rows)}")

    # Build and save
    df = build_and_save(all_rows)

    # Summary
    print_collection_summary(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
