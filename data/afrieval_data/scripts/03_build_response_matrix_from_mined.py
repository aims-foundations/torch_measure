#!/usr/bin/env python3
"""
Build response matrices from mined per-item model evaluation results.

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

import pandas as pd
import numpy as np

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

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main"

# ──────────────────────────────────────────────────────────────────────
# Language names
# ──────────────────────────────────────────────────────────────────────
LANG_NAMES = {
    "afr": "Afrikaans", "amh": "Amharic", "bam": "Bambara", "bbj": "Ghomala",
    "ewe": "Ewe", "fon": "Fon", "hau": "Hausa", "ibo": "Igbo",
    "kin": "Kinyarwanda", "lug": "Luganda", "luo": "Luo", "mos": "Mossi",
    "nya": "Chichewa", "pcm": "Nigerian Pidgin", "sna": "Shona",
    "swa": "Swahili", "tsn": "Setswana", "twi": "Twi", "wol": "Wolof",
    "xho": "Xhosa", "yor": "Yoruba", "zul": "Zulu",
}


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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Mining Per-Item Model Results for African NLP Benchmarks")
    print("=" * 70)
    print()
    print("This script downloads published per-item predictions from GitHub")
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
