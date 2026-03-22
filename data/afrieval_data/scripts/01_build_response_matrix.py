#!/usr/bin/env python3
"""
Build task metadata and item content for African NLP benchmarks.

Data sources (all from HuggingFace, public, no auth needed):
  1. AfriSenti — masakhane/afrisenti (14 African languages, sentiment)
  2. MasakhaNER — whoisjones/masakhaner (20 African languages, NER)
  3. MasakhaNEWS — masakhane/masakhanews (16 African languages, news classification)
  4. SIB-200 — Davlan/sib200 (197 languages, topic classification — African subset)
  5. AfriQA — downloaded directly from GitHub JSON files (10 African languages, QA)

Outputs:
  - task_metadata.csv  : Item-level metadata (item_id, text, label, task_type, language,
                         language_variety, source_dataset, split, n_chars)
  - item_content.csv   : Full text content per item (item_id, content)
  - summary_stats.csv  : Summary statistics per dataset x language
"""

import os
import sys
import json
import warnings
import traceback
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.error import HTTPError

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

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# African language codes and SIB-200 config names
# ──────────────────────────────────────────────────────────────────────
# SIB-200 uses lang_Script format — these are the African-language configs
SIB200_AFRICAN_CONFIGS = [
    "afr_Latn", "aka_Latn", "amh_Ethi", "bam_Latn", "bem_Latn",
    "dyu_Latn", "ewe_Latn", "fon_Latn", "fuv_Latn", "gaz_Latn",
    "hau_Latn", "ibo_Latn", "kab_Latn", "kam_Latn", "kbp_Latn",
    "kea_Latn", "kik_Latn", "kin_Latn", "kmb_Latn", "knc_Arab",
    "knc_Latn", "kon_Latn", "lin_Latn", "lug_Latn", "luo_Latn",
    "mos_Latn", "nso_Latn", "nus_Latn", "nya_Latn", "plt_Latn",
    "run_Latn", "sag_Latn", "sna_Latn", "som_Latn", "sot_Latn",
    "ssw_Latn", "swh_Latn", "taq_Latn", "taq_Tfng", "tir_Ethi",
    "tsn_Latn", "tso_Latn", "tum_Latn", "twi_Latn", "tzm_Tfng",
    "umb_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zul_Latn",
]

# Mapping of language codes to full names (for metadata)
LANG_NAMES = {
    "afr": "Afrikaans", "amh": "Amharic", "bam": "Bambara", "ewe": "Ewe",
    "fon": "Fon", "ful": "Fulfulde", "hau": "Hausa", "ibo": "Igbo",
    "kin": "Kinyarwanda", "lin": "Lingala", "lug": "Luganda", "luo": "Luo",
    "mos": "Mossi", "nya": "Chichewa", "orm": "Oromo", "sna": "Shona",
    "som": "Somali", "sot": "Sesotho", "ssw": "Swati", "swa": "Swahili",
    "tir": "Tigrinya", "tsn": "Setswana", "tso": "Tsonga", "twi": "Twi",
    "wol": "Wolof", "xho": "Xhosa", "yor": "Yoruba", "zul": "Zulu",
    "bem": "Bemba", "aka": "Akan", "pcm": "Nigerian Pidgin", "run": "Kirundi",
    "gaz": "Oromo", "nso": "Sepedi", "plt": "Malagasy", "swh": "Swahili",
    "kmb": "Kimbundu", "umb": "Umbundu", "taq": "Tamasheq", "tzm": "Tamazight",
    "kea": "Kabuverdianu", "dyu": "Dyula", "fuv": "Fulfulde (Nigeria)",
    "kab": "Kabyle", "kam": "Kamba", "kbp": "Kabiye", "kik": "Kikuyu",
    "knc": "Kanuri", "kon": "Kongo", "nus": "Nuer", "sag": "Sango",
    "tum": "Tumbuka", "bbj": "Ghomala",
    "eng": "English", "fra": "French", "por": "Portuguese",
    "arq": "Algerian Arabic", "ary": "Moroccan Arabic",
}


def safe_str(val, max_chars=200):
    """Convert a value to string, truncated to max_chars."""
    if val is None:
        return ""
    s = str(val).replace("\n", " ").replace("\r", " ").strip()
    return s[:max_chars]


def get_dataset_configs(dataset_name):
    """Try to get available configs for a dataset."""
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name)
        return configs
    except Exception as e:
        print(f"  Warning: could not get configs for {dataset_name}: {e}")
        return None


def load_dataset_safe(dataset_name, config=None, split=None, **kwargs):
    """Safely load a dataset, handling errors gracefully."""
    from datasets import load_dataset
    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split, **kwargs)
        else:
            ds = load_dataset(dataset_name, split=split, **kwargs)
        return ds
    except Exception as e:
        print(f"  Warning: Failed to load {dataset_name} (config={config}): {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# 1. AfriSenti
# ──────────────────────────────────────────────────────────────────────

def collect_afrisenti(all_metadata, all_content):
    """Collect AfriSenti: Twitter sentiment analysis in 14 African languages."""
    print("\n" + "=" * 70)
    print("1. AfriSenti — masakhane/afrisenti")
    print("=" * 70)

    dataset_name = "masakhane/afrisenti"
    configs = get_dataset_configs(dataset_name)
    print(f"  Available configs: {configs}")

    if configs is None:
        print("  ERROR: Cannot retrieve configs. Skipping AfriSenti.")
        return

    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    total_items = 0
    for lang_config in configs:
        print(f"  Loading config: {lang_config} ...", end=" ", flush=True)
        try:
            ds = load_dataset_safe(dataset_name, lang_config)
            if ds is None:
                continue

            for split_name in ds.keys():
                split_ds = ds[split_name]
                print(f"({split_name}: {len(split_ds)})", end=" ", flush=True)

                columns = split_ds.column_names
                text_col = "tweet" if "tweet" in columns else "text" if "text" in columns else columns[0]
                label_col = "label" if "label" in columns else None

                for idx, row in enumerate(split_ds):
                    item_id = f"afrisenti_{lang_config}_{split_name}_{idx}"
                    text = str(row.get(text_col, ""))
                    label_val = row.get(label_col, None) if label_col else None

                    if isinstance(label_val, int) and label_val in label_map:
                        label_str = label_map[label_val]
                    elif isinstance(label_val, str):
                        label_str = label_val
                    else:
                        label_str = str(label_val) if label_val is not None else ""

                    all_metadata.append({
                        "item_id": item_id,
                        "text": safe_str(text),
                        "label": label_str,
                        "task_type": "sentiment_analysis",
                        "language": lang_config,
                        "language_variety": LANG_NAMES.get(lang_config, lang_config),
                        "source_dataset": "afrisenti",
                        "split": split_name,
                        "n_chars": len(text),
                    })
                    all_content.append({
                        "item_id": item_id,
                        "content": text,
                    })
                    total_items += 1
            print()
        except Exception as e:
            print(f"\n  Error processing {lang_config}: {e}")
            traceback.print_exc()

    print(f"  AfriSenti total items: {total_items}")


# ──────────────────────────────────────────────────────────────────────
# 2. MasakhaNER (via whoisjones/masakhaner which has parquet data)
# ──────────────────────────────────────────────────────────────────────

def collect_masakhaner(all_metadata, all_content):
    """Collect MasakhaNER: NER in 20 African languages.

    Uses whoisjones/masakhaner which has the data in loadable parquet format
    (the original masakhane/masakhaner2 uses a deprecated loading script).
    """
    print("\n" + "=" * 70)
    print("2. MasakhaNER — whoisjones/masakhaner")
    print("=" * 70)

    dataset_name = "whoisjones/masakhaner"
    configs = get_dataset_configs(dataset_name)
    print(f"  Available configs: {configs}")

    if configs is None:
        print("  ERROR: Cannot retrieve configs. Skipping MasakhaNER.")
        return

    # Load a representative subset (5 languages) plus any extra
    target_langs = ["hau", "yor", "swa", "ibo", "kin"]
    available_langs = [l for l in target_langs if l in configs]
    # Also add a few more if available
    extras = [l for l in configs if l not in available_langs][:5]
    available_langs.extend(extras)

    total_items = 0
    for lang_config in available_langs:
        print(f"  Loading config: {lang_config} ...", end=" ", flush=True)
        try:
            ds = load_dataset_safe(dataset_name, lang_config)
            if ds is None:
                continue

            for split_name in ds.keys():
                split_ds = ds[split_name]
                print(f"({split_name}: {len(split_ds)})", end=" ", flush=True)

                columns = split_ds.column_names

                for idx, row in enumerate(split_ds):
                    item_id = f"masakhaner_{lang_config}_{split_name}_{idx}"

                    # This dataset has: tokens, text, token_spans, char_spans
                    text = str(row.get("text", ""))
                    if not text and "tokens" in columns:
                        tokens = row.get("tokens", [])
                        text = " ".join([str(t) for t in tokens]) if isinstance(tokens, list) else str(tokens)

                    # Extract entity types from token_spans
                    token_spans = row.get("token_spans", [])
                    entity_types = set()
                    if isinstance(token_spans, list):
                        for span in token_spans:
                            if isinstance(span, dict) and "label" in span:
                                entity_types.add(span["label"])
                    label_str = ",".join(sorted(entity_types)) if entity_types else "O"

                    all_metadata.append({
                        "item_id": item_id,
                        "text": safe_str(text),
                        "label": label_str,
                        "task_type": "named_entity_recognition",
                        "language": lang_config,
                        "language_variety": LANG_NAMES.get(lang_config, lang_config),
                        "source_dataset": "masakhaner",
                        "split": split_name,
                        "n_chars": len(text),
                    })
                    all_content.append({
                        "item_id": item_id,
                        "content": text,
                    })
                    total_items += 1
            print()
        except Exception as e:
            print(f"\n  Error processing {lang_config}: {e}")
            traceback.print_exc()

    print(f"  MasakhaNER total items: {total_items}")


# ──────────────────────────────────────────────────────────────────────
# 3. MasakhaNEWS
# ──────────────────────────────────────────────────────────────────────

def collect_masakhanews(all_metadata, all_content):
    """Collect MasakhaNEWS: News topic classification in 16 African languages."""
    print("\n" + "=" * 70)
    print("3. MasakhaNEWS — masakhane/masakhanews")
    print("=" * 70)

    dataset_name = "masakhane/masakhanews"
    configs = get_dataset_configs(dataset_name)
    print(f"  Available configs: {configs}")

    if configs is None:
        print("  ERROR: Cannot retrieve configs. Skipping MasakhaNEWS.")
        return

    topic_label_map = {
        0: "business", 1: "entertainment", 2: "health",
        3: "politics", 4: "religion", 5: "sports", 6: "technology",
    }

    total_items = 0
    for lang_config in configs:
        print(f"  Loading config: {lang_config} ...", end=" ", flush=True)
        try:
            ds = load_dataset_safe(dataset_name, lang_config)
            if ds is None:
                continue

            for split_name in ds.keys():
                split_ds = ds[split_name]
                print(f"({split_name}: {len(split_ds)})", end=" ", flush=True)

                columns = split_ds.column_names
                text_col = "headline" if "headline" in columns else "text" if "text" in columns else columns[0]
                label_col = "label" if "label" in columns else None

                for idx, row in enumerate(split_ds):
                    item_id = f"masakhanews_{lang_config}_{split_name}_{idx}"
                    text = str(row.get(text_col, ""))
                    label_val = row.get(label_col, None) if label_col else None

                    if isinstance(label_val, int) and label_val in topic_label_map:
                        label_str = topic_label_map[label_val]
                    elif isinstance(label_val, str):
                        label_str = label_val
                    else:
                        label_str = str(label_val) if label_val is not None else ""

                    full_text = text
                    if "text" in columns and text_col != "text":
                        article = str(row.get("text", ""))
                        full_text = f"{text}\n{article}" if article else text

                    all_metadata.append({
                        "item_id": item_id,
                        "text": safe_str(text),
                        "label": label_str,
                        "task_type": "topic_classification",
                        "language": lang_config,
                        "language_variety": LANG_NAMES.get(lang_config, lang_config),
                        "source_dataset": "masakhanews",
                        "split": split_name,
                        "n_chars": len(full_text),
                    })
                    all_content.append({
                        "item_id": item_id,
                        "content": full_text,
                    })
                    total_items += 1
            print()
        except Exception as e:
            print(f"\n  Error processing {lang_config}: {e}")
            traceback.print_exc()

    print(f"  MasakhaNEWS total items: {total_items}")


# ──────────────────────────────────────────────────────────────────────
# 4. SIB-200 (African subset)
# ──────────────────────────────────────────────────────────────────────

def collect_sib200(all_metadata, all_content):
    """Collect SIB-200: Topic classification across 197 languages (African subset)."""
    print("\n" + "=" * 70)
    print("4. SIB-200 — Davlan/sib200 (African subset)")
    print("=" * 70)

    dataset_name = "Davlan/sib200"
    configs = get_dataset_configs(dataset_name)
    if configs:
        print(f"  Total configs: {len(configs)}")
    else:
        print("  ERROR: Cannot retrieve configs. Skipping SIB-200.")
        return

    # Filter to African configs
    african_configs = [c for c in SIB200_AFRICAN_CONFIGS if c in configs]
    print(f"  African configs to load ({len(african_configs)}): {african_configs}")

    total_items = 0
    for lang_config in african_configs:
        print(f"  Loading config: {lang_config} ...", end=" ", flush=True)
        try:
            ds = load_dataset_safe(dataset_name, lang_config)
            if ds is None:
                continue

            # Extract short language code from config (e.g., "afr_Latn" -> "afr")
            lang_code = lang_config.split("_")[0]

            for split_name in ds.keys():
                split_ds = ds[split_name]
                print(f"({split_name}: {len(split_ds)})", end=" ", flush=True)

                columns = split_ds.column_names
                text_col = "text" if "text" in columns else columns[0]
                label_col = "category" if "category" in columns else "label" if "label" in columns else None

                for idx, row in enumerate(split_ds):
                    item_id = f"sib200_{lang_config}_{split_name}_{idx}"
                    text = str(row.get(text_col, ""))
                    label_val = row.get(label_col, None) if label_col else None
                    label_str = str(label_val) if label_val is not None else ""

                    all_metadata.append({
                        "item_id": item_id,
                        "text": safe_str(text),
                        "label": label_str,
                        "task_type": "topic_classification",
                        "language": lang_code,
                        "language_variety": LANG_NAMES.get(lang_code, lang_config),
                        "source_dataset": "sib200",
                        "split": split_name,
                        "n_chars": len(text),
                    })
                    all_content.append({
                        "item_id": item_id,
                        "content": text,
                    })
                    total_items += 1
            print()
        except Exception as e:
            print(f"\n  Error processing {lang_config}: {e}")
            traceback.print_exc()

    print(f"  SIB-200 total African items: {total_items}")


# ──────────────────────────────────────────────────────────────────────
# 5. AfriQA (from GitHub JSON files, since HF loading script is broken)
# ──────────────────────────────────────────────────────────────────────

AFRIQA_LANG_2_PIVOT = {
    "bem": "en", "fon": "fr", "hau": "en", "ibo": "en", "kin": "en",
    "swa": "en", "twi": "en", "wol": "fr", "yor": "en", "zul": "en",
}

AFRIQA_BASE_URL = "https://raw.githubusercontent.com/masakhane-io/afriqa/main/data/queries"


def download_jsonl(url):
    """Download and parse a JSONL file from a URL (one JSON object per line)."""
    try:
        req = Request(url, headers={"User-Agent": "Python/datasets"})
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        records = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception as e:
        print(f"[download error: {e}]", end=" ")
        return None


def collect_afriqa(all_metadata, all_content):
    """Collect AfriQA: QA in 10 African languages (from GitHub JSONL files)."""
    print("\n" + "=" * 70)
    print("5. AfriQA — masakhane/afriqa (from GitHub JSONL)")
    print("=" * 70)

    total_items = 0
    for lang, pivot in AFRIQA_LANG_2_PIVOT.items():
        print(f"  Loading language: {lang} (pivot={pivot}) ...", end=" ", flush=True)
        lang_items = 0

        for split_name, split_file in [("train", "train"), ("dev", "dev"), ("test", "test")]:
            url = f"{AFRIQA_BASE_URL}/{lang}/queries.afriqa.{lang}.{pivot}.{split_file}.json"
            records = download_jsonl(url)

            if records is None or len(records) == 0:
                print(f"[{split_name}: MISSING]", end=" ", flush=True)
                continue

            print(f"({split_name}: {len(records)})", end=" ", flush=True)

            for idx, record in enumerate(records):
                item_id = f"afriqa_{lang}_{split_name}_{idx}"

                # Extract question and answer
                question = str(record.get("question", record.get("query", "")))
                # answers field is stored as a string repr of a list
                answers_raw = record.get("answers", "")
                if isinstance(answers_raw, str):
                    # Parse string like "['2021']"
                    try:
                        import ast
                        answers_list = ast.literal_eval(answers_raw)
                        answer = str(answers_list[0]) if answers_list else ""
                    except (ValueError, SyntaxError):
                        answer = answers_raw
                elif isinstance(answers_raw, list):
                    answer = str(answers_raw[0]) if answers_raw else ""
                else:
                    answer = str(answers_raw) if answers_raw else ""

                text = question
                full_text = f"Q: {question}"

                all_metadata.append({
                    "item_id": item_id,
                    "text": safe_str(text),
                    "label": safe_str(answer),
                    "task_type": "question_answering",
                    "language": lang,
                    "language_variety": LANG_NAMES.get(lang, lang),
                    "source_dataset": "afriqa",
                    "split": split_name,
                    "n_chars": len(full_text),
                })
                all_content.append({
                    "item_id": item_id,
                    "content": full_text,
                })
                total_items += 1
                lang_items += 1

        print(f" [{lang_items} items]")

    print(f"  AfriQA total items: {total_items}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Building African NLP Benchmark Collection")
    print("=" * 70)

    all_metadata = []
    all_content = []

    # Collect each dataset
    collect_afrisenti(all_metadata, all_content)
    collect_masakhaner(all_metadata, all_content)
    collect_masakhanews(all_metadata, all_content)
    collect_sib200(all_metadata, all_content)
    collect_afriqa(all_metadata, all_content)

    # ──────────────────────────────────────────────────────────────────
    # Build DataFrames and save
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving processed data")
    print("=" * 70)

    if not all_metadata:
        print("ERROR: No data collected. Check dataset availability.")
        sys.exit(1)

    # task_metadata.csv
    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(PROCESSED_DIR / "task_metadata.csv", index=False)
    print(f"  task_metadata.csv: {len(meta_df)} rows")

    # item_content.csv
    content_df = pd.DataFrame(all_content)
    content_df.to_csv(PROCESSED_DIR / "item_content.csv", index=False)
    print(f"  item_content.csv: {len(content_df)} rows")

    # ──────────────────────────────────────────────────────────────────
    # Summary statistics
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Per dataset
    print("\nPer dataset:")
    for ds_name in meta_df["source_dataset"].unique():
        subset = meta_df[meta_df["source_dataset"] == ds_name]
        n_langs = subset["language"].nunique()
        n_items = len(subset)
        splits = subset["split"].unique().tolist()
        print(f"  {ds_name}: {n_items} items, {n_langs} languages, splits={splits}")

    # Per language
    print("\nPer language (top 25 by item count):")
    lang_counts = meta_df.groupby("language").size().sort_values(ascending=False)
    for lang, count in lang_counts.head(25).items():
        name = LANG_NAMES.get(lang, lang)
        print(f"  {lang} ({name}): {count} items")

    # Per task type
    print("\nPer task type:")
    for task_type in meta_df["task_type"].unique():
        subset = meta_df[meta_df["task_type"] == task_type]
        print(f"  {task_type}: {len(subset)} items")

    # Save summary
    summary_rows = []
    for (ds, lang), group in meta_df.groupby(["source_dataset", "language"]):
        summary_rows.append({
            "source_dataset": ds,
            "language": lang,
            "language_name": LANG_NAMES.get(lang, lang),
            "n_items": len(group),
            "n_train": len(group[group["split"] == "train"]),
            "n_validation": len(group[group["split"].isin(["validation", "dev"])]),
            "n_test": len(group[group["split"] == "test"]),
            "task_type": group["task_type"].iloc[0],
            "n_labels": group["label"].nunique(),
            "avg_n_chars": group["n_chars"].mean(),
            "median_n_chars": group["n_chars"].median(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(PROCESSED_DIR / "summary_stats.csv", index=False)
    print(f"\n  summary_stats.csv: {len(summary_df)} rows")

    # Overall stats
    print(f"\n  Total items: {len(meta_df)}")
    print(f"  Total languages: {meta_df['language'].nunique()}")
    print(f"  Total datasets: {meta_df['source_dataset'].nunique()}")
    print(f"  Total task types: {meta_df['task_type'].nunique()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
