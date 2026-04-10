#!/usr/bin/env python3
"""
Build task metadata, item content, and response matrices for African NLP benchmarks.

Part 1 — Item collection (from HuggingFace and GitHub):
  Data sources:
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

Part 2 — Response-matrix mining (from GitHub per-item predictions):
  Sources mined (all publicly available on GitHub):
    1. MasakhaNER v1.0 — entity_analysis/ directory
       Repository: github.com/masakhane-io/masakhane-ner
       Models: XLM-R, mBERT, biLSTM_CRF, freeze_XLM-R_BiLSTM, freeze_mBERT_BiLSTM
       Languages: amh, hau, ibo, kin, lug, luo, pcm, swa, wol, yor (10 languages)
       Format: CoNLL per-token predictions on test sets

    2. MasakhaNER v2.0 — baseline_models_results/ directory
       Repository: github.com/masakhane-io/masakhane-ner (MasakhaNER2.0/)
       Models: afriberta, afroxlmr, mbert, mdeberta, rembert, xlmrbase, xlmrlarge
       Languages: bam, bbj, ewe, fon, hau, ibo, kin, lug, mos, nya, pcm, sna,
                  swa, tsn, twi, wol, xho, yor, zul (19 languages)
       Format: CoNLL per-token predictions (test_predictions{1..7}.txt)

  Outputs:
    - response_matrix_masakhaner_v1_sentence.csv  (sentence-level: all entities correct?)
    - response_matrix_masakhaner_v2_sentence.csv  (sentence-level: all entities correct?)
    - response_matrix_masakhaner_v1_token.csv     (token-level: each token correct?)
    - response_matrix_masakhaner_v2_token.csv     (token-level: each token correct?)
    - mining_report.txt                            (summary of what was found)

Response matrix convention (matching torch_measure):
  - Rows: item_id
  - Columns: model names
  - Values: 1.0 (correct), 0.0 (incorrect), empty (not evaluated)
"""

import os
import sys
import io
import json
import warnings
import traceback
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent
BASE_DIR = _BENCHMARK_DIR
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main"

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
# Merged from both item-collection and prediction-mining sources
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


# ======================================================================
# Part 1 — Item collection helpers
# ======================================================================

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


# ======================================================================
# Part 2 — Response-matrix mining helpers
# ======================================================================

# ──────────────────────────────────────────────────────────────────────
# Utility: download text from URL
# ──────────────────────────────────────────────────────────────────────
def download_text(url, timeout=30):
    """Download text content from a URL. Returns None on failure."""
    try:
        req = Request(url, headers={"User-Agent": "Python/torch_measure"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            return None
        print(f"    [HTTP {e.code}] {url}")
        return None
    except (URLError, Exception) as e:
        print(f"    [Error] {url}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Parse CoNLL-format prediction files
# ──────────────────────────────────────────────────────────────────────
def parse_conll_predictions(text, n_cols_expected=None):
    """Parse CoNLL-format NER predictions.

    Handles:
      - 2-column: token predicted_tag  (prediction-only files)
      - 3-column: token gold_tag predicted_tag  (biLSTM_CRF files)

    Returns list of sentences, where each sentence is a list of dicts:
      [{"token": str, "gold": str or None, "predicted": str}, ...]

    Blank lines separate sentences.
    """
    sentences = []
    current_sentence = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue

        parts = line.split()
        if len(parts) == 3:
            # token gold predicted
            current_sentence.append({
                "token": parts[0],
                "gold": parts[1],
                "predicted": parts[2],
            })
        elif len(parts) == 2:
            # token predicted (or token gold -- depends on context)
            current_sentence.append({
                "token": parts[0],
                "gold": None,
                "predicted": parts[1],
            })
        elif len(parts) == 1:
            # Just a token with no tag (skip or treat as O)
            current_sentence.append({
                "token": parts[0],
                "gold": None,
                "predicted": "O",
            })
        # Lines with more parts: try to recover
        elif len(parts) > 3:
            current_sentence.append({
                "token": parts[0],
                "gold": parts[-2] if len(parts) >= 3 else None,
                "predicted": parts[-1],
            })

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def parse_conll_gold(text):
    """Parse CoNLL-format gold standard file (2-column: token gold_tag)."""
    sentences = []
    current_sentence = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue

        parts = line.split()
        if len(parts) >= 2:
            current_sentence.append({
                "token": parts[0],
                "gold": parts[1],
            })
        elif len(parts) == 1:
            current_sentence.append({
                "token": parts[0],
                "gold": "O",
            })

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def align_predictions_with_gold(pred_sentences, gold_sentences, strict=False):
    """Align prediction-only files (2-col) with gold files (2-col).

    Returns (aligned_sentences, skipped_indices) where:
      - aligned_sentences: list of aligned sentences with gold + predicted tags
      - skipped_indices: set of sentence indices that could not be aligned

    If strict=True, returns (None, None) on any mismatch.
    If strict=False, skips sentences with token count mismatches.
    """
    if len(pred_sentences) != len(gold_sentences):
        if strict:
            return None, None
        # Use min length
        n = min(len(pred_sentences), len(gold_sentences))
    else:
        n = len(pred_sentences)

    aligned = []
    skipped = set()

    for i in range(n):
        pred_sent = pred_sentences[i]
        gold_sent = gold_sentences[i]

        if len(pred_sent) != len(gold_sent):
            skipped.add(i)
            # Insert a placeholder so indices stay aligned
            aligned.append(None)
            continue

        aligned_sent = []
        for pred_tok, gold_tok in zip(pred_sent, gold_sent):
            aligned_sent.append({
                "token": gold_tok["token"],
                "gold": gold_tok["gold"],
                "predicted": pred_tok["predicted"],
            })
        aligned.append(aligned_sent)

    return aligned, skipped


# ──────────────────────────────────────────────────────────────────────
# Scoring functions
# ──────────────────────────────────────────────────────────────────────
def score_sentence_level(sentences):
    """Score at sentence level: 1 if all tokens in sentence are correctly predicted.

    Returns list of (sentence_idx, correct_bool)
    """
    results = []
    for idx, sent in enumerate(sentences):
        all_correct = all(
            tok["gold"] == tok["predicted"]
            for tok in sent
            if tok["gold"] is not None
        )
        results.append((idx, 1.0 if all_correct else 0.0))
    return results


def score_token_level(sentences):
    """Score at token level: 1 if token tag matches gold.

    Returns list of (sentence_idx, token_idx, correct_bool)
    """
    results = []
    for sent_idx, sent in enumerate(sentences):
        for tok_idx, tok in enumerate(sent):
            if tok["gold"] is not None:
                correct = 1.0 if tok["gold"] == tok["predicted"] else 0.0
                results.append((sent_idx, tok_idx, correct))
    return results


# ──────────────────────────────────────────────────────────────────────
# Source 1: MasakhaNER v1.0 — entity_analysis/
# ──────────────────────────────────────────────────────────────────────
MASAKHANER_V1_MODELS = {
    "XLM-R": {
        "dir": "entity_analysis/XLM-R",
        "pattern": "{lang}_xlmr_test_predictions.txt",
        "format": "2col",  # token, predicted
    },
    "mBERT": {
        "dir": "entity_analysis/mBERT",
        "pattern": "{lang}_bert_test_predictions.txt",
        "format": "2col",
    },
    "biLSTM_CRF": {
        "dir": "entity_analysis/biLSTM_CRF",
        "pattern": "test.{lang}_model",
        "format": "3col",  # token, gold, predicted
    },
    "freeze_XLM-R_BiLSTM": {
        "dir": "entity_analysis/freeze_XLM-R_BiLSTM",
        "pattern": "{lang}_freezexlmr_test_predictions.txt",
        "format": "2col",
    },
    "freeze_mBERT_BiLSTM": {
        "dir": "entity_analysis/freeze_mBERT_BiLSTM",
        "pattern": "{lang}_freezembert_test_predictions.txt",
        "format": "2col",
    },
}

MASAKHANER_V1_LANGUAGES = [
    "amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"
]


def collect_masakhaner_v1(report_lines):
    """Collect MasakhaNER v1.0 predictions from entity_analysis/."""
    print("\n" + "=" * 70)
    print("Source 1: MasakhaNER v1.0 — entity_analysis/")
    print("=" * 70)
    report_lines.append("\n=== MasakhaNER v1.0 (entity_analysis/) ===")

    # First download gold standard test data for alignment
    gold_data = {}  # lang -> list of gold sentences
    for lang in MASAKHANER_V1_LANGUAGES:
        url = f"{GITHUB_RAW_BASE}/data/{lang}/test.txt"
        print(f"  Downloading gold data for {lang}...", end=" ", flush=True)
        text = download_text(url)
        if text:
            gold_sentences = parse_conll_gold(text)
            gold_data[lang] = gold_sentences
            print(f"({len(gold_sentences)} sentences)")
        else:
            print("MISSING")

    # Now download and process each model's predictions
    # sentence_results[item_id][model] = score
    sentence_results = defaultdict(dict)
    token_results = defaultdict(dict)

    total_files = 0
    total_sentences = 0
    model_lang_counts = {}

    for model_name, model_info in MASAKHANER_V1_MODELS.items():
        print(f"\n  Model: {model_name}")
        model_lang_counts[model_name] = 0

        for lang in MASAKHANER_V1_LANGUAGES:
            filename = model_info["pattern"].format(lang=lang)
            url = f"{GITHUB_RAW_BASE}/{model_info['dir']}/{filename}"

            print(f"    {lang}: ", end="", flush=True)
            text = download_text(url)
            if text is None:
                print("MISSING")
                continue

            total_files += 1

            skipped = set()
            if model_info["format"] == "3col":
                # biLSTM_CRF: token gold predicted
                sentences = parse_conll_predictions(text)
                # In 3-col files, gold is already embedded
            else:
                # 2-col files: token predicted -- need gold alignment
                pred_sentences = parse_conll_predictions(text)
                if lang not in gold_data:
                    print(f"no gold data for alignment")
                    continue

                sentences, skipped = align_predictions_with_gold(
                    pred_sentences, gold_data[lang]
                )
                if sentences is None:
                    print(f"alignment FAILED (pred={len(pred_sentences)}, "
                          f"gold={len(gold_data[lang])} sentences)")
                    report_lines.append(
                        f"  ALIGNMENT FAILED: {model_name}/{lang} "
                        f"(pred={len(pred_sentences)}, gold={len(gold_data[lang])})"
                    )
                    continue

            # Score sentences (skip None placeholders from alignment)
            valid_sentences = [
                (i, s) for i, s in enumerate(sentences)
                if s is not None
            ]
            n_skipped = len(skipped) if skipped else 0
            n_total = len(valid_sentences)
            n_correct = 0

            for sent_idx, sent in valid_sentences:
                all_correct = all(
                    tok["gold"] == tok["predicted"]
                    for tok in sent
                    if tok["gold"] is not None
                )
                score = 1.0 if all_correct else 0.0
                if score == 1.0:
                    n_correct += 1

                item_id = f"masakhaner_v1_{lang}_test_{sent_idx}"
                sentence_results[item_id][model_name] = score

                # Token-level
                for tok_idx, tok in enumerate(sent):
                    if tok["gold"] is not None:
                        tok_score = 1.0 if tok["gold"] == tok["predicted"] else 0.0
                        tok_item_id = f"masakhaner_v1_{lang}_test_{sent_idx}_tok{tok_idx}"
                        token_results[tok_item_id][model_name] = tok_score

            accuracy = n_correct / n_total if n_total > 0 else 0

            skip_msg = f" ({n_skipped} skipped)" if n_skipped > 0 else ""
            print(f"{n_total} sentences{skip_msg}, "
                  f"sentence-acc={accuracy:.3f}")

            model_lang_counts[model_name] += 1
            total_sentences += n_total

    # Summary
    print(f"\n  Total files downloaded: {total_files}")
    print(f"  Total sentences scored: {total_sentences}")
    for m, c in model_lang_counts.items():
        print(f"    {m}: {c} languages")

    report_lines.append(f"  Files downloaded: {total_files}")
    report_lines.append(f"  Total sentences scored: {total_sentences}")
    for m, c in model_lang_counts.items():
        report_lines.append(f"    {m}: {c} languages")

    return sentence_results, token_results


# ──────────────────────────────────────────────────────────────────────
# Source 2: MasakhaNER v2.0 — baseline_models_results/
# ──────────────────────────────────────────────────────────────────────
MASAKHANER_V2_MODELS = [
    "afriberta", "afroxlmr", "mbert", "mdeberta",
    "rembert", "xlmrbase", "xlmrlarge"
]

MASAKHANER_V2_LANGUAGES = [
    "bam", "bbj", "ewe", "fon", "hau", "ibo", "kin", "lug", "mos",
    "nya", "pcm", "sna", "swa", "tsn", "twi", "wol", "xho", "yor", "zul"
]

# V2.0 has 7 runs per model-language pair (test_predictions1..7)
# We use the first run (test_predictions1) as the canonical prediction
V2_RUN_IDX = 1


def collect_masakhaner_v2(report_lines):
    """Collect MasakhaNER v2.0 predictions from baseline_models_results/."""
    print("\n" + "=" * 70)
    print("Source 2: MasakhaNER v2.0 — baseline_models_results/")
    print("=" * 70)
    report_lines.append("\n=== MasakhaNER v2.0 (baseline_models_results/) ===")

    # Download gold standard test data for v2.0
    gold_data = {}
    for lang in MASAKHANER_V2_LANGUAGES:
        url = f"{GITHUB_RAW_BASE}/MasakhaNER2.0/data/{lang}/test.txt"
        print(f"  Downloading v2 gold data for {lang}...", end=" ", flush=True)
        text = download_text(url)
        if text:
            gold_sentences = parse_conll_gold(text)
            gold_data[lang] = gold_sentences
            print(f"({len(gold_sentences)} sentences)")
        else:
            print("MISSING")

    sentence_results = defaultdict(dict)
    token_results = defaultdict(dict)

    total_files = 0
    total_sentences = 0
    model_lang_counts = defaultdict(int)

    for model_name in MASAKHANER_V2_MODELS:
        print(f"\n  Model: {model_name}")

        for lang in MASAKHANER_V2_LANGUAGES:
            dirname = f"{lang}_{model_name}"
            filename = f"test_predictions{V2_RUN_IDX}.txt"
            url = (f"{GITHUB_RAW_BASE}/MasakhaNER2.0/"
                   f"baseline_models_results/{dirname}/{filename}")

            print(f"    {lang}: ", end="", flush=True)
            text = download_text(url)
            if text is None:
                print("MISSING")
                continue

            total_files += 1

            # V2.0 predictions are 2-col (token, predicted)
            pred_sentences = parse_conll_predictions(text)

            if lang not in gold_data:
                print(f"no gold data")
                continue

            sentences, skipped = align_predictions_with_gold(
                pred_sentences, gold_data[lang]
            )
            if sentences is None:
                print(f"alignment FAILED (pred={len(pred_sentences)}, "
                      f"gold={len(gold_data[lang])} sentences)")
                report_lines.append(
                    f"  ALIGNMENT FAILED: v2/{model_name}/{lang} "
                    f"(pred={len(pred_sentences)}, gold={len(gold_data[lang])})"
                )
                continue

            # Score sentences (skip None placeholders from alignment)
            valid_sentences = [
                (i, s) for i, s in enumerate(sentences)
                if s is not None
            ]
            n_skipped = len(skipped) if skipped else 0
            n_total = len(valid_sentences)
            n_correct = 0

            # Full model name with v2 prefix
            full_model_name = f"v2_{model_name}"

            for sent_idx, sent in valid_sentences:
                all_correct = all(
                    tok["gold"] == tok["predicted"]
                    for tok in sent
                    if tok["gold"] is not None
                )
                score = 1.0 if all_correct else 0.0
                if score == 1.0:
                    n_correct += 1

                item_id = f"masakhaner_v2_{lang}_test_{sent_idx}"
                sentence_results[item_id][full_model_name] = score

                # Token-level
                for tok_idx, tok in enumerate(sent):
                    if tok["gold"] is not None:
                        tok_score = 1.0 if tok["gold"] == tok["predicted"] else 0.0
                        tok_item_id = f"masakhaner_v2_{lang}_test_{sent_idx}_tok{tok_idx}"
                        token_results[tok_item_id][full_model_name] = tok_score

            accuracy = n_correct / n_total if n_total > 0 else 0

            skip_msg = f" ({n_skipped} skipped)" if n_skipped > 0 else ""
            print(f"{n_total} sentences{skip_msg}, sentence-acc={accuracy:.3f}")

            model_lang_counts[model_name] += 1
            total_sentences += n_total

    print(f"\n  Total files downloaded: {total_files}")
    print(f"  Total sentences scored: {total_sentences}")
    for m, c in model_lang_counts.items():
        print(f"    {m}: {c} languages")

    report_lines.append(f"  Files downloaded: {total_files}")
    report_lines.append(f"  Total sentences scored: {total_sentences}")
    for m, c in model_lang_counts.items():
        report_lines.append(f"    {m}: {c} languages")

    return sentence_results, token_results


# ──────────────────────────────────────────────────────────────────────
# Build response matrix DataFrame from results dict
# ──────────────────────────────────────────────────────────────────────
def build_response_matrix(results_dict):
    """Convert {item_id: {model: score}} into a DataFrame.

    Returns DataFrame with item_id as first column, then one column per model.
    """
    if not results_dict:
        return pd.DataFrame(columns=["item_id"])

    # Get all models
    all_models = set()
    for scores in results_dict.values():
        all_models.update(scores.keys())
    all_models = sorted(all_models)

    rows = []
    for item_id in sorted(results_dict.keys()):
        row = {"item_id": item_id}
        for model in all_models:
            row[model] = results_dict[item_id].get(model, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ──────────────────────────────────────────────────────────────────────
# Build item metadata for mined predictions
# ──────────────────────────────────────────────────────────────────────
def build_item_metadata(sentence_results, version_prefix, gold_data_map):
    """Build metadata for items in the response matrix.

    Returns DataFrame with: item_id, language, language_name, source_dataset,
    task_type, split, n_tokens
    """
    rows = []
    for item_id in sorted(sentence_results.keys()):
        # Parse item_id: masakhaner_{version}_{lang}_test_{idx}
        parts = item_id.split("_")
        # e.g., masakhaner_v1_hau_test_42
        if len(parts) >= 5:
            lang = parts[2]
            sent_idx = int(parts[-1])
        else:
            lang = "unknown"
            sent_idx = 0

        n_tokens = 0
        if lang in gold_data_map and sent_idx < len(gold_data_map[lang]):
            n_tokens = len(gold_data_map[lang][sent_idx])

        rows.append({
            "item_id": item_id,
            "language": lang,
            "language_name": LANG_NAMES.get(lang, lang),
            "source_dataset": f"masakhaner_{version_prefix}",
            "task_type": "named_entity_recognition",
            "split": "test",
            "n_tokens": n_tokens,
        })

    return pd.DataFrame(rows)


# ======================================================================
# Main
# ======================================================================

def main():
    # ==================================================================
    # Part 1: Collect items from HuggingFace / GitHub
    # ==================================================================
    print("=" * 70)
    print("Part 1: Building African NLP Benchmark Collection")
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

    print("\nPart 1 done!")

    # ==================================================================
    # Part 2: Mine per-item predictions and build response matrices
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("Part 2: Mining Per-Item Model Results for African NLP Benchmarks")
    print("=" * 70)
    print()
    print("This step downloads published per-item predictions from GitHub")
    print("and builds response matrices in torch_measure format.")
    print()

    report_lines = [
        "Mining Report: Per-Item Model Results for African NLP Benchmarks",
        "=" * 60,
        "",
        "Sources searched:",
        "  1. AfroBench (McGill-NLP) — AGGREGATE ONLY, no per-item data",
        "  2. AfriSenti SemEval 2023 participants — no usable item IDs",
        "  3. MasakhaNER v1.0 entity_analysis/ — PER-TOKEN predictions found",
        "  4. MasakhaNER v2.0 baseline_models_results/ — PER-TOKEN predictions found",
        "  5. MasakhaNEWS — no per-item predictions published",
        "  6. IrokoBench — dataset only, no per-item model outputs",
        "  7. Sahara — no per-item results in repo",
        "  8. Bridging-the-Gap — files in Git LFS, not directly accessible",
        "  9. HuggingFace model cards — no per-item results found",
        "",
    ]

    # ── Source 1: MasakhaNER v1.0 ──
    v1_sent, v1_tok = collect_masakhaner_v1(report_lines)

    # ── Source 2: MasakhaNER v2.0 ──
    v2_sent, v2_tok = collect_masakhaner_v2(report_lines)

    # ──────────────────────────────────────────────────────────────────
    # Build and save response matrices
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Building Response Matrices")
    print("=" * 70)

    matrices = {}

    # V1 sentence-level
    if v1_sent:
        df = build_response_matrix(v1_sent)
        out_path = PROCESSED_DIR / "response_matrix_masakhaner_v1_sentence.csv"
        df.to_csv(out_path, index=False)
        matrices["v1_sentence"] = df
        n_items = len(df)
        n_models = len(df.columns) - 1
        print(f"  response_matrix_masakhaner_v1_sentence.csv: "
              f"{n_items} items x {n_models} models")
        report_lines.append(f"\nOutput: response_matrix_masakhaner_v1_sentence.csv")
        report_lines.append(f"  Items: {n_items}, Models: {n_models}")
        report_lines.append(f"  Models: {', '.join(df.columns[1:])}")
    else:
        print("  WARNING: No MasakhaNER v1 sentence results collected")

    # V1 token-level
    if v1_tok:
        df = build_response_matrix(v1_tok)
        out_path = PROCESSED_DIR / "response_matrix_masakhaner_v1_token.csv"
        df.to_csv(out_path, index=False)
        matrices["v1_token"] = df
        n_items = len(df)
        n_models = len(df.columns) - 1
        print(f"  response_matrix_masakhaner_v1_token.csv: "
              f"{n_items} items x {n_models} models")
        report_lines.append(f"\nOutput: response_matrix_masakhaner_v1_token.csv")
        report_lines.append(f"  Items: {n_items}, Models: {n_models}")
    else:
        print("  WARNING: No MasakhaNER v1 token results collected")

    # V2 sentence-level
    if v2_sent:
        df = build_response_matrix(v2_sent)
        out_path = PROCESSED_DIR / "response_matrix_masakhaner_v2_sentence.csv"
        df.to_csv(out_path, index=False)
        matrices["v2_sentence"] = df
        n_items = len(df)
        n_models = len(df.columns) - 1
        print(f"  response_matrix_masakhaner_v2_sentence.csv: "
              f"{n_items} items x {n_models} models")
        report_lines.append(f"\nOutput: response_matrix_masakhaner_v2_sentence.csv")
        report_lines.append(f"  Items: {n_items}, Models: {n_models}")
        report_lines.append(f"  Models: {', '.join(df.columns[1:])}")
    else:
        print("  WARNING: No MasakhaNER v2 sentence results collected")

    # V2 token-level (can be very large; only save if manageable)
    if v2_tok:
        n_items = len(v2_tok)
        if n_items > 2_000_000:
            print(f"  Skipping v2 token-level: {n_items} items (too large)")
            report_lines.append(
                f"\nSkipped: response_matrix_masakhaner_v2_token.csv "
                f"({n_items} items, too large)"
            )
        else:
            df = build_response_matrix(v2_tok)
            out_path = PROCESSED_DIR / "response_matrix_masakhaner_v2_token.csv"
            df.to_csv(out_path, index=False)
            matrices["v2_token"] = df
            n_models = len(df.columns) - 1
            print(f"  response_matrix_masakhaner_v2_token.csv: "
                  f"{n_items} items x {n_models} models")
            report_lines.append(
                f"\nOutput: response_matrix_masakhaner_v2_token.csv"
            )
            report_lines.append(f"  Items: {n_items}, Models: {n_models}")

    # ──────────────────────────────────────────────────────────────────
    # Summary statistics per matrix
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    report_lines.append("\n\n=== Summary Statistics ===")

    for name, df in matrices.items():
        model_cols = [c for c in df.columns if c != "item_id"]
        print(f"\n  {name}:")
        report_lines.append(f"\n  {name}:")

        for model in model_cols:
            valid = df[model].dropna()
            n_valid = len(valid)
            if n_valid > 0:
                mean_score = valid.mean()
                line = f"    {model}: {n_valid} items, mean={mean_score:.4f}"
            else:
                line = f"    {model}: 0 valid items"
            print(line)
            report_lines.append(line)

        # Language breakdown for sentence-level matrices
        if "sentence" in name:
            print(f"\n    Per-language breakdown:")
            report_lines.append(f"    Per-language breakdown:")
            # Extract language from item_id
            df_copy = df.copy()
            df_copy["language"] = df_copy["item_id"].apply(
                lambda x: x.split("_")[2] if len(x.split("_")) >= 3 else "unk"
            )
            for lang in sorted(df_copy["language"].unique()):
                lang_df = df_copy[df_copy["language"] == lang]
                n_items = len(lang_df)
                # Average score across all models
                model_scores = []
                for model in model_cols:
                    valid = lang_df[model].dropna()
                    if len(valid) > 0:
                        model_scores.append(valid.mean())
                avg = np.mean(model_scores) if model_scores else 0
                lang_name = LANG_NAMES.get(lang, lang)
                line = (f"      {lang} ({lang_name}): {n_items} sentences, "
                        f"avg_accuracy={avg:.3f}")
                print(line)
                report_lines.append(line)

    # ──────────────────────────────────────────────────────────────────
    # Save mining report
    # ──────────────────────────────────────────────────────────────────
    report_path = PROCESSED_DIR / "mining_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Mining report saved to: {report_path}")

    # ──────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_items = sum(len(df) for df in matrices.values())
    total_models = len(set(
        col for df in matrices.values()
        for col in df.columns if col != "item_id"
    ))

    print(f"  Total response matrices: {len(matrices)}")
    print(f"  Total unique items: {total_items}")
    print(f"  Total unique models: {total_models}")
    print(f"  Output directory: {PROCESSED_DIR}")
    print()

    print("Sources with NO per-item data (aggregate only):")
    print("  - AfroBench: per-language aggregate CSV scores only")
    print("  - AfriSenti SemEval 2023 (NLP-UMUTeam): predictions.csv has")
    print("    (index, y_pred, y_real) but no item_id or language mapping")
    print("  - MasakhaNEWS: no model output files in repository")
    print("  - IrokoBench: dataset files only (test.tsv), no model outputs")
    print("  - Sahara: evaluation scripts only, no published results")
    print("  - Bridging-the-Gap: files stored in Git LFS (67GB), not accessible")
    print("  - HuggingFace model cards: no per-item results found")
    print()
    print("Done!")


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
