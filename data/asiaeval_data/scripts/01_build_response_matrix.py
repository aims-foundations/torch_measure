#!/usr/bin/env python3
"""
Build task metadata and item content CSVs for Southeast Asian and South Asian
NLP benchmarks from HuggingFace.

Data sources:
  1. NusaX       — thonyyy/nusax_sentiment (12 Indonesian local languages, sentiment)
                    (mirror of indonlp/NusaX-senti without requiring trust_remote_code)
  2. Belebele    — facebook/belebele (reading comprehension, Global South subset)
  3. XCOPA       — xcopa (causal reasoning, 11 languages)
  4. Global-MMLU — CohereForAI/Global-MMLU (multilingual MMLU, subset)
  5. IndicCOPA   — ai4bharat/IndicCOPA (15+ Indic languages, causal reasoning)
                    (loaded via direct JSONL downloads from HuggingFace Hub)

Outputs (in processed/):
  - task_metadata.csv      : item_id, text (first 200 chars), label, task_type,
                              language, source_dataset, split
  - item_content.csv       : item_id, full content
  - summary_stats.csv      : per-language/dataset/split summary statistics
  - summary_by_language.csv: high-level per-language summary
"""

import os
import sys
import json
import warnings
import traceback
from pathlib import Path

import numpy as np
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

# Global accumulators
all_metadata_rows = []
all_content_rows = []
item_counter = 0


def next_item_id():
    global item_counter
    iid = f"item_{item_counter:06d}"
    item_counter += 1
    return iid


def truncate(text, max_len=200):
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ──────────────────────────────────────────────────────────────────────
# 1. NusaX — via thonyyy/nusax_sentiment (parquet mirror, no script)
# ──────────────────────────────────────────────────────────────────────
def load_nusax():
    print("=" * 70)
    print("DATASET 1: NusaX (thonyyy/nusax_sentiment)")
    print("  Mirror of indonlp/NusaX-senti without trust_remote_code")
    print("=" * 70)

    from datasets import load_dataset

    total_items = 0
    try:
        # This mirror has all languages in a single dataset with a 'language' column
        ds = load_dataset("thonyyy/nusax_sentiment")
        for split_name in ds.keys():
            split_ds = ds[split_name]
            n = len(split_ds)
            print(f"  {split_name}: {n} items")
            print(f"  Columns: {split_ds.column_names}")

            # Check first sample to understand structure
            if n > 0:
                sample = split_ds[0]
                print(f"  Sample: {sample}")

            for row in split_ds:
                iid = next_item_id()
                text = str(row.get("text", ""))
                label = row.get("label", "")
                language = str(row.get("language", "unknown"))
                orig_split = str(row.get("split", split_name))

                # Label is already string in this mirror: negative/neutral/positive
                label_str = str(label)

                all_metadata_rows.append({
                    "item_id": iid,
                    "text": truncate(text),
                    "label": label_str,
                    "task_type": "sentiment_analysis",
                    "language": language,
                    "source_dataset": "NusaX",
                    "split": orig_split,
                })
                all_content_rows.append({
                    "item_id": iid,
                    "content": text,
                })
                total_items += 1

    except Exception as e:
        print(f"  ERROR loading thonyyy/nusax_sentiment: {e}")
        traceback.print_exc()

    print(f"  Total NusaX items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 2. Belebele — facebook/belebele (streaming mode for reliability)
# ──────────────────────────────────────────────────────────────────────
def load_belebele():
    print("=" * 70)
    print("DATASET 2: Belebele (facebook/belebele)")
    print("  Using streaming mode to avoid brotli decoder errors")
    print("=" * 70)

    from datasets import load_dataset, get_dataset_config_names

    target_langs = [
        "hin_Deva", "ben_Beng", "tam_Taml", "tha_Thai", "vie_Latn",
        "ind_Latn", "tgl_Latn", "swa_Latn", "yor_Latn", "ara_Arab",
        "zho_Hans",
    ]

    try:
        configs = get_dataset_config_names("facebook/belebele")
        print(f"  Available configs ({len(configs)}): {configs[:10]}...")
        available_target = [l for l in target_langs if l in configs]
        missing = [l for l in target_langs if l not in configs]
        if missing:
            print(f"  Missing configs: {missing}")
        target_langs_final = available_target
    except Exception as e:
        print(f"  Could not get configs: {e}")
        target_langs_final = target_langs

    total_items = 0
    for lang in target_langs_final:
        try:
            # Use streaming to avoid brotli decoder issues
            ds_stream = load_dataset(
                "facebook/belebele", lang, split="test", streaming=True
            )
            items = list(ds_stream)
            n = len(items)
            print(f"  {lang}/test: {n} items")

            for row in items:
                iid = next_item_id()

                passage = str(row.get("flores_passage", ""))
                question = str(row.get("question", ""))
                mc1 = str(row.get("mc_answer1", ""))
                mc2 = str(row.get("mc_answer2", ""))
                mc3 = str(row.get("mc_answer3", ""))
                mc4 = str(row.get("mc_answer4", ""))
                correct = row.get("correct_answer_num", "")

                full_content = (
                    f"Passage: {passage}\n"
                    f"Question: {question}\n"
                    f"A: {mc1}\nB: {mc2}\nC: {mc3}\nD: {mc4}\n"
                    f"Correct: {correct}"
                )

                all_metadata_rows.append({
                    "item_id": iid,
                    "text": truncate(f"{question} | {passage}"),
                    "label": str(correct),
                    "task_type": "reading_comprehension",
                    "language": lang,
                    "source_dataset": "Belebele",
                    "split": "test",
                })
                all_content_rows.append({
                    "item_id": iid,
                    "content": full_content,
                })
                total_items += 1

        except Exception as e:
            print(f"  ERROR loading {lang}: {e}")

    print(f"  Total Belebele items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 3. XCOPA — xcopa (causal reasoning)
# ──────────────────────────────────────────────────────────────────────
def load_xcopa():
    print("=" * 70)
    print("DATASET 3: XCOPA (causal reasoning)")
    print("=" * 70)

    from datasets import load_dataset, get_dataset_config_names

    hf_names = ["xcopa", "cambridgeltl/xcopa"]

    configs = None
    chosen_name = None
    for hf_name in hf_names:
        try:
            configs = get_dataset_config_names(hf_name)
            chosen_name = hf_name
            print(f"  Using dataset: {hf_name}")
            print(f"  Available configs: {configs}")
            break
        except Exception as e:
            print(f"  Tried {hf_name}: {e}")

    if configs is None:
        print("  Could not load XCOPA from any source. Skipping.")
        return

    total_items = 0
    for lang in configs:
        try:
            ds = load_dataset(chosen_name, lang)
            for split_name in ds.keys():
                split_ds = ds[split_name]
                n = len(split_ds)
                print(f"  {lang}/{split_name}: {n} items")

                for row in split_ds:
                    iid = next_item_id()

                    premise = str(row.get("premise", ""))
                    choice1 = str(row.get("choice1", ""))
                    choice2 = str(row.get("choice2", ""))
                    question_type = str(row.get("question", ""))
                    label = row.get("label", "")

                    full_content = (
                        f"Premise: {premise}\n"
                        f"Question type: {question_type}\n"
                        f"Choice 1: {choice1}\n"
                        f"Choice 2: {choice2}\n"
                        f"Label: {label}"
                    )

                    all_metadata_rows.append({
                        "item_id": iid,
                        "text": truncate(f"{premise} | {choice1} | {choice2}"),
                        "label": str(label),
                        "task_type": "causal_reasoning",
                        "language": lang,
                        "source_dataset": "XCOPA",
                        "split": split_name,
                    })
                    all_content_rows.append({
                        "item_id": iid,
                        "content": full_content,
                    })
                    total_items += 1

        except Exception as e:
            print(f"  ERROR loading {lang}: {e}")

    print(f"  Total XCOPA items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 4. Global-MMLU — CohereForAI/Global-MMLU
# ──────────────────────────────────────────────────────────────────────
def load_global_mmlu():
    print("=" * 70)
    print("DATASET 4: Global-MMLU (CohereForAI/Global-MMLU)")
    print("=" * 70)

    from datasets import load_dataset, get_dataset_config_names

    target_langs = ["ar", "hi", "bn", "th", "vi", "id", "sw", "yo", "es", "pt", "zh"]

    hf_names = ["CohereForAI/Global-MMLU", "CohereLabs/Global-MMLU"]
    configs = None
    chosen_name = None

    for hf_name in hf_names:
        try:
            configs = get_dataset_config_names(hf_name)
            chosen_name = hf_name
            print(f"  Using dataset: {hf_name}")
            print(f"  Available configs ({len(configs)}): {configs[:20]}...")
            break
        except Exception as e:
            print(f"  Tried {hf_name}: {e}")

    if configs is None:
        print("  Could not load Global-MMLU. Skipping.")
        return

    available_target = [l for l in target_langs if l in configs]
    missing = [l for l in target_langs if l not in configs]
    if missing:
        print(f"  Missing language configs: {missing}")
    print(f"  Will load languages: {available_target}")

    total_items = 0
    for lang in available_target:
        try:
            print(f"  Loading {lang}...", end=" ", flush=True)
            ds = load_dataset(chosen_name, lang, split="test")
            n = len(ds)
            print(f"test: {n} items")

            # Cap at 5000 per language
            max_items = 5000
            indices = range(min(n, max_items))
            if n > max_items:
                print(f"    (sampling {max_items} of {n} items)")

            for idx in indices:
                row = ds[int(idx)]
                iid = next_item_id()

                question = str(row.get("question", ""))
                subject = str(row.get("subject", ""))
                answer = str(row.get("answer", ""))
                choices = row.get("choices", [])
                if isinstance(choices, list):
                    choices_str = " | ".join(str(c) for c in choices)
                else:
                    choices_str = str(choices)

                answer_map = {0: "A", 1: "B", 2: "C", 3: "D",
                              "0": "A", "1": "B", "2": "C", "3": "D",
                              "A": "A", "B": "B", "C": "C", "D": "D"}
                answer_str = answer_map.get(answer, str(answer))

                full_content = (
                    f"Subject: {subject}\n"
                    f"Question: {question}\n"
                    f"Choices: {choices_str}\n"
                    f"Answer: {answer_str}"
                )

                all_metadata_rows.append({
                    "item_id": iid,
                    "text": truncate(f"[{subject}] {question}"),
                    "label": answer_str,
                    "task_type": "multiple_choice_qa",
                    "language": lang,
                    "source_dataset": "Global-MMLU",
                    "split": "test",
                })
                all_content_rows.append({
                    "item_id": iid,
                    "content": full_content,
                })
                total_items += 1

        except Exception as e:
            print(f"\n  ERROR loading {lang}: {e}")
            traceback.print_exc()

    print(f"  Total Global-MMLU items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# 5. IndicCOPA — ai4bharat/IndicCOPA (direct JSONL download)
# ──────────────────────────────────────────────────────────────────────
def load_indiccopa():
    print("=" * 70)
    print("DATASET 5: IndicCOPA (ai4bharat/IndicCOPA)")
    print("  Loading via direct JSONL file downloads (dataset script unsupported)")
    print("=" * 70)

    from huggingface_hub import hf_hub_download, list_repo_files

    # Discover available language files
    try:
        all_files = list_repo_files("ai4bharat/IndicCOPA", repo_type="dataset")
        jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
        print(f"  Found {len(jsonl_files)} JSONL files: {jsonl_files}")
    except Exception as e:
        print(f"  ERROR listing repo files: {e}")
        return

    # Parse language codes from filenames like "data/test.hi.jsonl"
    lang_files = {}
    for f in jsonl_files:
        parts = f.split("/")[-1].replace(".jsonl", "").split(".")
        if len(parts) >= 2:
            split_name = parts[0]
            lang_code = parts[1]
            lang_files.setdefault(lang_code, []).append((split_name, f))

    print(f"  Languages found: {sorted(lang_files.keys())}")

    total_items = 0
    for lang in sorted(lang_files.keys()):
        for split_name, filepath in lang_files[lang]:
            try:
                local_path = hf_hub_download(
                    "ai4bharat/IndicCOPA", filepath, repo_type="dataset"
                )
                with open(local_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                n = len(lines)
                print(f"  {lang}/{split_name}: {n} items")

                for line in lines:
                    row = json.loads(line.strip())
                    iid = next_item_id()

                    premise = str(row.get("premise", ""))
                    choice1 = str(row.get("choice1", ""))
                    choice2 = str(row.get("choice2", ""))
                    question_type = str(row.get("question", ""))
                    label = row.get("label", "")

                    full_content = (
                        f"Premise: {premise}\n"
                        f"Question type: {question_type}\n"
                        f"Choice 1: {choice1}\n"
                        f"Choice 2: {choice2}\n"
                        f"Label: {label}"
                    )

                    all_metadata_rows.append({
                        "item_id": iid,
                        "text": truncate(f"{premise} | {choice1} | {choice2}"),
                        "label": str(label),
                        "task_type": "causal_reasoning",
                        "language": lang,
                        "source_dataset": "IndicCOPA",
                        "split": split_name,
                    })
                    all_content_rows.append({
                        "item_id": iid,
                        "content": full_content,
                    })
                    total_items += 1

            except Exception as e:
                print(f"  ERROR loading {lang}/{split_name}: {e}")
                traceback.print_exc()

    print(f"  Total IndicCOPA items: {total_items}\n")


# ──────────────────────────────────────────────────────────────────────
# Summary Statistics
# ──────────────────────────────────────────────────────────────────────
def build_summary_stats(metadata_df):
    """Build per-language, per-dataset summary statistics."""
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = metadata_df.groupby(
        ["source_dataset", "language", "task_type", "split"]
    ).agg(
        n_items=("item_id", "count"),
        n_labels=("label", "nunique"),
        label_distribution=("label", lambda x: dict(x.value_counts().head(5))),
    ).reset_index()

    high_level = metadata_df.groupby(
        ["source_dataset", "language"]
    ).agg(
        n_items=("item_id", "count"),
        n_splits=("split", "nunique"),
        task_types=("task_type", lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()

    print(f"\n  Datasets loaded: {metadata_df['source_dataset'].nunique()}")
    print(f"  Total items: {len(metadata_df)}")
    print(f"  Languages: {metadata_df['language'].nunique()}")
    print(f"  Task types: {sorted(metadata_df['task_type'].unique())}")

    print("\n  Items per dataset:")
    for ds, grp in metadata_df.groupby("source_dataset"):
        langs = sorted(grp["language"].unique())
        print(f"    {ds}: {len(grp)} items, {len(langs)} languages")
        print(f"      Languages: {', '.join(langs)}")

    print("\n  Items per language (across all datasets):")
    lang_counts = metadata_df.groupby("language")["item_id"].count().sort_values(
        ascending=False
    )
    for lang, count in lang_counts.items():
        datasets = metadata_df[metadata_df["language"] == lang][
            "source_dataset"
        ].unique()
        print(f"    {lang:15s}: {count:6d} items  ({', '.join(sorted(datasets))})")

    return summary, high_level


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("AsiaEval Response Matrix Builder")
    print("=" * 70)
    print("Building task metadata and item content for SE/South Asian benchmarks")
    print("=" * 70 + "\n")

    # Load each dataset
    load_nusax()
    load_belebele()
    load_xcopa()
    load_global_mmlu()
    load_indiccopa()

    # Build DataFrames
    print("\n" + "=" * 70)
    print("BUILDING OUTPUT FILES")
    print("=" * 70)

    metadata_df = pd.DataFrame(all_metadata_rows)
    content_df = pd.DataFrame(all_content_rows)

    if len(metadata_df) == 0:
        print("ERROR: No data was loaded. Exiting.")
        sys.exit(1)

    print(f"  Total metadata rows: {len(metadata_df)}")
    print(f"  Total content rows: {len(content_df)}")

    # Build summary stats
    summary, high_level = build_summary_stats(metadata_df)

    # ── Save outputs ──
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    out_meta = PROCESSED_DIR / "task_metadata.csv"
    metadata_df.to_csv(out_meta, index=False)
    print(f"  Saved: {out_meta} ({len(metadata_df)} rows)")

    out_content = PROCESSED_DIR / "item_content.csv"
    content_df.to_csv(out_content, index=False)
    print(f"  Saved: {out_content} ({len(content_df)} rows)")

    out_summary = PROCESSED_DIR / "summary_stats.csv"
    summary.to_csv(out_summary, index=False)
    print(f"  Saved: {out_summary}")

    out_high = PROCESSED_DIR / "summary_by_language.csv"
    high_level.to_csv(out_high, index=False)
    print(f"  Saved: {out_high}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Datasets: {metadata_df['source_dataset'].nunique()}")
    print(f"  Languages: {metadata_df['language'].nunique()}")
    print(f"  Total items: {len(metadata_df)}")
    print(f"  Task types: {sorted(metadata_df['task_type'].unique())}")
    print(f"\n  Output directory: {PROCESSED_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
