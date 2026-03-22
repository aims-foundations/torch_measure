#!/usr/bin/env python3
"""
Build SIB-200 response matrix from published LLM per-item predictions.

Data source:
  SIB-200 GitHub repo (https://github.com/dadelani/sib-200)
  - llm-results/output_gpt4/*.tsv   : GPT-4 per-item predictions (205 languages)
  - llm-results/output_gpt3.5/*.tsv : GPT-3.5 per-item predictions (205 languages)

  Each TSV has columns: [pandas_idx], index_id, category (ground truth), text, model_response
  The task is topic classification into 7 categories:
    science/technology, travel, politics, sports, health, entertainment, geography

  Model responses are free-text; we use fuzzy matching to extract predicted categories.

Outputs (in ../processed/):
  - response_matrix.csv  : Items x models binary matrix (1=correct, 0=incorrect)
  - task_metadata.csv    : Item-level metadata (item_id, language, language_script, category, text)
  - model_summary.csv    : Per-model accuracy overall and per-language
"""

import os
import re
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
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
REPO_DIR = RAW_DIR / "sib-200"
LLM_RESULTS_DIR = REPO_DIR / "llm-results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# SIB-200 category labels (ground truth)
# ──────────────────────────────────────────────────────────────────────
VALID_CATEGORIES = {
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
}

# Mapping from various model output phrasings to canonical category labels.
# GPT models often produce variations like "Science/Technology", "science_technology",
# "Science and Technology", etc.
CATEGORY_ALIASES = {}

def _build_alias_map():
    """Build a comprehensive alias map for fuzzy category matching."""
    aliases = {}
    for cat in VALID_CATEGORIES:
        # Exact lowercase
        aliases[cat.lower()] = cat
        # Without slash
        aliases[cat.lower().replace("/", " ")] = cat
        aliases[cat.lower().replace("/", "_")] = cat
        aliases[cat.lower().replace("/", " and ")] = cat
        aliases[cat.lower().replace("/", " & ")] = cat
        # Each word alone (for multi-word categories)
        for word in cat.lower().replace("/", " ").split():
            if word not in ("and", "&"):
                aliases[word] = cat

    # Extra common GPT aliases
    aliases["tech"] = "science/technology"
    aliases["sci"] = "science/technology"
    aliases["sci/tech"] = "science/technology"
    aliases["science"] = "science/technology"
    aliases["technology"] = "science/technology"
    aliases["science and technology"] = "science/technology"
    aliases["science & technology"] = "science/technology"
    aliases["science_technology"] = "science/technology"
    aliases["sport"] = "sports"
    aliases["politic"] = "politics"
    aliases["political"] = "politics"
    aliases["geographic"] = "geography"
    aliases["geographical"] = "geography"
    aliases["healthy"] = "health"
    aliases["medical"] = "health"
    aliases["medicine"] = "health"
    aliases["entertain"] = "entertainment"
    aliases["traveling"] = "travel"
    aliases["travelling"] = "travel"
    aliases["tourism"] = "travel"

    return aliases

CATEGORY_ALIASES = _build_alias_map()


def extract_category_from_response(response_text):
    """
    Extract the predicted category from a model's free-text response.

    Strategy:
      1. Direct match: check if any canonical category appears in the response.
      2. Quoted match: look for text in quotes that matches a category.
      3. Fuzzy match: normalize and search for alias matches.
      4. Fallback: return None (unmatched).

    Returns (predicted_category, match_method) or (None, "unmatched").
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return None, "empty"

    text = response_text.strip()
    text_lower = text.lower()

    # Strategy 1: Direct substring match on canonical categories
    # Check longer categories first to avoid partial matches
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if cat.lower() in text_lower:
            return cat, "direct"

    # Strategy 2: Look for quoted text
    quoted = re.findall(r'"([^"]+)"', text)
    quoted += re.findall(r"'([^']+)'", text)
    for q in quoted:
        q_lower = q.strip().lower()
        if q_lower in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[q_lower], "quoted"
        # Check if any canonical category is in the quoted text
        for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
            if cat.lower() in q_lower:
                return cat, "quoted_substring"

    # Strategy 3: Fuzzy alias matching on the full response
    # Check multi-word aliases first
    multi_word_aliases = {k: v for k, v in CATEGORY_ALIASES.items() if " " in k or "/" in k}
    for alias, cat in sorted(multi_word_aliases.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in text_lower:
            return cat, "alias"

    # Then single-word aliases (only match as whole words)
    single_word_aliases = {k: v for k, v in CATEGORY_ALIASES.items() if " " not in k and "/" not in k}
    for alias, cat in sorted(single_word_aliases.items(), key=lambda x: len(x[0]), reverse=True):
        # Match as whole word to avoid false positives (e.g., "port" matching "sports")
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower):
            # Avoid matching 'sport' inside 'transport', etc.
            return cat, "word_alias"

    return None, "unmatched"


# ──────────────────────────────────────────────────────────────────────
# Step 1: Clone repo if needed
# ──────────────────────────────────────────────────────────────────────
def ensure_repo():
    """Clone the SIB-200 repo if not already present."""
    if REPO_DIR.exists() and (LLM_RESULTS_DIR).exists():
        print(f"  Repo already present at {REPO_DIR}")
        return True

    print("  Cloning SIB-200 repo...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/dadelani/sib-200.git", str(REPO_DIR)],
            check=True, capture_output=True, text=True,
        )
        print(f"  Cloned to {REPO_DIR}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR cloning repo: {e.stderr}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Step 2: Parse all TSV files
# ──────────────────────────────────────────────────────────────────────
def parse_model_results(model_dir, model_name):
    """
    Parse all TSV files for one model.

    Returns:
      results: list of dicts with keys: language, index_id, category, text, prediction, predicted_cat, match_method
      match_stats: Counter of match methods
    """
    tsv_files = sorted(model_dir.glob("*.tsv"))
    print(f"  Found {len(tsv_files)} language files for {model_name}")

    results = []
    match_stats = Counter()
    error_files = []

    for tsv_path in tsv_files:
        # Language code from filename: e.g., eng_Latn.tsv -> eng_Latn
        lang_code = tsv_path.stem  # e.g., "eng_Latn"

        try:
            # Read TSV -- first column is pandas index, so use index_col=0
            df = pd.read_csv(tsv_path, sep="\t", index_col=0, dtype=str, on_bad_lines="skip")

            # Identify the prediction column (last column, named gpt-4 or gpt-3)
            pred_col = df.columns[-1]

            for _, row in df.iterrows():
                index_id = str(row.get("index_id", "")).strip()
                category = str(row.get("category", "")).strip().lower()
                text = str(row.get("text", "")).strip()
                prediction = str(row.get(pred_col, "")).strip()

                # Extract predicted category
                pred_cat, method = extract_category_from_response(prediction)
                match_stats[method] += 1

                results.append({
                    "language": lang_code,
                    "index_id": index_id,
                    "category": category,
                    "text": text,
                    "prediction_raw": prediction,
                    "predicted_category": pred_cat,
                    "match_method": method,
                })

        except Exception as e:
            error_files.append((tsv_path.name, str(e)))

    if error_files:
        print(f"    WARNING: {len(error_files)} files had errors: {error_files[:5]}")

    return results, match_stats


# ──────────────────────────────────────────────────────────────────────
# Step 3: Build response matrix
# ──────────────────────────────────────────────────────────────────────
def build_response_matrix(all_results):
    """
    Build a binary response matrix: items x models.

    Item ID format: "{language}_{index_id}"
    Values: 1 (correct), 0 (incorrect), NaN (prediction could not be parsed)
    """
    print("\n" + "=" * 70)
    print("BUILDING RESPONSE MATRIX")
    print("=" * 70)

    # Organize by model
    models = sorted(all_results.keys())
    print(f"  Models: {models}")

    # Collect all unique item IDs across all models
    all_item_ids = set()
    for model_name, results in all_results.items():
        for r in results:
            item_id = f"{r['language']}_{r['index_id']}"
            all_item_ids.add(item_id)

    all_item_ids = sorted(all_item_ids)
    print(f"  Total unique items: {len(all_item_ids)}")

    # Build matrix
    matrix_data = {}
    for model_name in models:
        results = all_results[model_name]
        scores = {}
        for r in results:
            item_id = f"{r['language']}_{r['index_id']}"
            gt = r["category"]
            pred = r["predicted_category"]

            if pred is None:
                scores[item_id] = np.nan  # Could not parse prediction
            else:
                scores[item_id] = 1 if pred == gt else 0

        col = [scores.get(iid, np.nan) for iid in all_item_ids]
        matrix_data[model_name] = col

    response_matrix = pd.DataFrame(matrix_data, index=all_item_ids)
    response_matrix.index.name = "item_id"

    print(f"  Matrix shape: {response_matrix.shape}")
    print(f"  Density (non-NaN): {response_matrix.notna().mean().mean():.4f}")

    for model_name in models:
        col = response_matrix[model_name]
        valid = col.dropna()
        acc = valid.mean() if len(valid) > 0 else 0
        print(f"    {model_name}: accuracy={acc:.4f}, "
              f"valid={len(valid)}/{len(col)} "
              f"({len(valid)/len(col)*100:.1f}%)")

    return response_matrix


# ──────────────────────────────────────────────────────────────────────
# Step 4: Build task metadata
# ──────────────────────────────────────────────────────────────────────
def build_task_metadata(all_results):
    """
    Build task_metadata.csv with item-level information.

    Uses one model's results to get text and category (they should be the same
    across models since they come from the same test set).
    """
    print("\n" + "=" * 70)
    print("BUILDING TASK METADATA")
    print("=" * 70)

    # Use the first model's results for metadata
    first_model = sorted(all_results.keys())[0]
    results = all_results[first_model]

    metadata = {}
    for r in results:
        item_id = f"{r['language']}_{r['index_id']}"
        if item_id not in metadata:
            # Parse language and script from language code
            lang_parts = r["language"].split("_")
            lang_iso = lang_parts[0] if len(lang_parts) >= 1 else r["language"]
            lang_script = lang_parts[1] if len(lang_parts) >= 2 else ""

            text_truncated = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]

            metadata[item_id] = {
                "item_id": item_id,
                "language": r["language"],
                "language_iso": lang_iso,
                "language_script": lang_script,
                "category": r["category"],
                "index_id": r["index_id"],
                "text": text_truncated,
            }

    meta_df = pd.DataFrame(list(metadata.values()))
    print(f"  Total items: {len(meta_df)}")
    print(f"  Languages: {meta_df['language'].nunique()}")
    print(f"  Categories: {sorted(meta_df['category'].unique())}")
    print(f"  Items per category:")
    for cat, cnt in meta_df["category"].value_counts().items():
        print(f"    {cat}: {cnt}")

    return meta_df


# ──────────────────────────────────────────────────────────────────────
# Step 5: Build model summary
# ──────────────────────────────────────────────────────────────────────
def build_model_summary(response_matrix, meta_df):
    """Build model_summary.csv with per-model accuracy overall and per-language."""
    print("\n" + "=" * 70)
    print("BUILDING MODEL SUMMARY")
    print("=" * 70)

    # Create a lookup from item_id to language
    item_lang = dict(zip(meta_df["item_id"], meta_df["language"]))
    item_cat = dict(zip(meta_df["item_id"], meta_df["category"]))

    summaries = []
    for model_name in response_matrix.columns:
        col = response_matrix[model_name]
        valid = col.dropna()
        overall_acc = valid.mean() if len(valid) > 0 else np.nan
        n_valid = len(valid)
        n_correct = int(valid.sum()) if len(valid) > 0 else 0

        row = {
            "model": model_name,
            "overall_accuracy": overall_acc,
            "n_items_evaluated": n_valid,
            "n_correct": n_correct,
            "n_total": len(col),
            "parse_rate": n_valid / len(col) if len(col) > 0 else 0,
        }

        # Per-language accuracy
        lang_groups = defaultdict(list)
        for item_id, score in zip(col.index, col.values):
            lang = item_lang.get(item_id, "unknown")
            if not np.isnan(score):
                lang_groups[lang].append(score)

        for lang in sorted(lang_groups.keys()):
            scores = lang_groups[lang]
            row[f"acc_{lang}"] = np.mean(scores)

        # Per-category accuracy
        cat_groups = defaultdict(list)
        for item_id, score in zip(col.index, col.values):
            cat = item_cat.get(item_id, "unknown")
            if not np.isnan(score):
                cat_groups[cat].append(score)

        for cat in sorted(cat_groups.keys()):
            scores = cat_groups[cat]
            row[f"acc_cat_{cat}"] = np.mean(scores)

        summaries.append(row)

    summary_df = pd.DataFrame(summaries)

    # Print a compact overview
    for _, r in summary_df.iterrows():
        print(f"  {r['model']}: overall={r['overall_accuracy']:.4f}, "
              f"parsed={r['n_items_evaluated']}/{r['n_total']} "
              f"({r['parse_rate']*100:.1f}%)")

    return summary_df


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SIB-200 Response Matrix Builder")
    print("=" * 70)

    # Step 1: Ensure repo
    print("\n[Step 1] Checking SIB-200 repo...")
    if not ensure_repo():
        sys.exit(1)

    # Step 2: Parse LLM results
    print("\n[Step 2] Parsing LLM results...")
    model_dirs = {
        "gpt-4": LLM_RESULTS_DIR / "output_gpt4",
        "gpt-3.5": LLM_RESULTS_DIR / "output_gpt3.5",
    }

    all_results = {}
    for model_name, model_dir in model_dirs.items():
        if not model_dir.exists():
            print(f"  WARNING: {model_dir} not found, skipping {model_name}")
            continue

        print(f"\n  --- Parsing {model_name} ---")
        results, match_stats = parse_model_results(model_dir, model_name)
        all_results[model_name] = results

        total = sum(match_stats.values())
        print(f"    Total predictions: {total}")
        print(f"    Match statistics:")
        for method, count in sorted(match_stats.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"      {method:25s}: {count:6d} ({pct:.1f}%)")

        # Show some unmatched examples
        unmatched = [r for r in results if r["predicted_category"] is None]
        if unmatched:
            print(f"    Unmatched examples (first 5):")
            for u in unmatched[:5]:
                raw = u["prediction_raw"][:120]
                print(f"      [{u['language']}] GT={u['category']} | Response: {raw}")

    if not all_results:
        print("ERROR: No model results parsed!")
        sys.exit(1)

    # Step 3: Build response matrix
    response_matrix = build_response_matrix(all_results)

    # Step 4: Build task metadata
    meta_df = build_task_metadata(all_results)

    # Step 5: Build model summary
    summary_df = build_model_summary(response_matrix, meta_df)

    # ── Save outputs ──
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # response_matrix.csv
    out_path = PROCESSED_DIR / "response_matrix.csv"
    response_matrix.to_csv(out_path)
    print(f"  Saved response_matrix.csv: {out_path}")
    print(f"    Shape: {response_matrix.shape} "
          f"({response_matrix.shape[0]} items x {response_matrix.shape[1]} models)")

    # task_metadata.csv
    out_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(out_path, index=False)
    print(f"  Saved task_metadata.csv: {out_path}")
    print(f"    Shape: {meta_df.shape}")

    # model_summary.csv
    out_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  Saved model_summary.csv: {out_path}")
    print(f"    Entries: {len(summary_df)}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(f"\n  Response Matrix:")
    print(f"    Items: {response_matrix.shape[0]}")
    print(f"    Models: {response_matrix.shape[1]}")
    print(f"    Density (non-NaN): {response_matrix.notna().mean().mean():.4f}")

    for model_name in response_matrix.columns:
        col = response_matrix[model_name]
        valid = col.dropna()
        acc = valid.mean() if len(valid) > 0 else 0
        print(f"    {model_name}: accuracy={acc:.4f}, "
              f"parsed={len(valid)}/{len(col)}")

    print(f"\n  Task Metadata:")
    print(f"    Items: {len(meta_df)}")
    print(f"    Languages: {meta_df['language'].nunique()}")
    print(f"    Scripts: {sorted(meta_df['language_script'].unique())}")

    # Per-language accuracy summary for best model
    best_model = response_matrix.mean().idxmax()
    print(f"\n  Per-language accuracy for {best_model} (top/bottom 10):")
    item_lang = dict(zip(meta_df["item_id"], meta_df["language"]))
    lang_acc = {}
    col = response_matrix[best_model]
    for item_id, score in zip(col.index, col.values):
        lang = item_lang.get(item_id, "unknown")
        lang_acc.setdefault(lang, []).append(score)
    lang_acc_mean = {lang: np.nanmean(scores) for lang, scores in lang_acc.items()}
    sorted_langs = sorted(lang_acc_mean.items(), key=lambda x: x[1], reverse=True)

    print(f"    Top 10:")
    for lang, acc in sorted_langs[:10]:
        print(f"      {lang:15s}: {acc:.4f}")
    print(f"    Bottom 10:")
    for lang, acc in sorted_langs[-10:]:
        print(f"      {lang:15s}: {acc:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
