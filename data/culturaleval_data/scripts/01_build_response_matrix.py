"""
Build task metadata and item content CSVs for MENA/Arabic and cross-cultural NLP benchmarks.

Collects data from the following HuggingFace datasets:
  1. CulturalBench (kellycyy/CulturalBench) - 4,908 cultural knowledge questions, 45 regions
  2. CVQA (afaji/cvqa) - Culturally-driven multilingual VQA, 26 languages
  3. ArabicMMLU (MBZUAI/ArabicMMLU) - Arabic MMLU-style knowledge QA, 40+ subjects
  4. EXAMS (mhardalov/exams) - Multilingual exam QA, 16 languages
  5. INCLUDE (CohereLabs/include-base-44) - Regional exam knowledge, 44 languages

Outputs (in ../processed/):
  - task_metadata.csv: item_id, text, label, task_type, language, country_region,
                       source_dataset, split
  - item_content.csv:  item_id, full_content
  - summary_statistics.txt: Per-dataset counts, language/region breakdowns

Notes:
  - Datasets that fail to load are skipped gracefully with diagnostic messages.
  - CVQA: only text/metadata portion is kept (images are not downloaded).
  - ArabicMMLU uses streaming mode to avoid brotli decompression errors.
  - INCLUDE loads each of 45 language configs separately.
  - No trust_remote_code is used.
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate(text, max_len=200):
    """Return first max_len chars of text (for the 'text' column)."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def safe_str(val):
    """Convert value to string, handling None/NaN/'None'."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).strip()
    if s.lower() == "none":
        return ""
    return s


# Accumulators
all_task_metadata = []
all_item_content = []
dataset_stats = {}
skipped_datasets = []


# ============================================================================
# 1. CulturalBench  (kellycyy/CulturalBench)
#    Columns: data_idx, question_idx, prompt_question, prompt_option, answer (bool), country
#    This is CulturalBench-Hard: binary True/False per (question, option) pair.
# ============================================================================
def load_culturalbench():
    print("\n" + "=" * 70)
    print("[1/5] Loading CulturalBench (kellycyy/CulturalBench) ...")
    print("=" * 70)
    try:
        from datasets import load_dataset
        ds = load_dataset("kellycyy/CulturalBench")
        print(f"  Loaded successfully. Splits: {list(ds.keys())}")

        count = 0
        for split_name in ds:
            split = ds[split_name]
            cols = split.column_names
            print(f"  Split '{split_name}': {len(split)} rows, columns={cols}")

            for idx, row in enumerate(split):
                item_id = f"culturalbench_{split_name}_{idx}"

                question = safe_str(row.get("prompt_question", ""))
                option = safe_str(row.get("prompt_option", ""))
                answer = row.get("answer", "")  # boolean
                country = safe_str(row.get("country", ""))

                # Full content: question + option being evaluated
                full_content = f"{question} | Option: {option} | Answer: {answer}"
                label = str(answer)  # True or False

                all_task_metadata.append({
                    "item_id": item_id,
                    "text": truncate(full_content),
                    "label": label,
                    "task_type": "cultural_knowledge",
                    "language": "en",
                    "country_region": country,
                    "source_dataset": "CulturalBench",
                    "split": split_name,
                })
                all_item_content.append({
                    "item_id": item_id,
                    "full_content": full_content,
                })
                count += 1

        dataset_stats["CulturalBench"] = {
            "total_items": count,
            "splits": list(ds.keys()),
        }
        print(f"  -> Collected {count} items from CulturalBench")

    except Exception as e:
        print(f"  ERROR loading CulturalBench: {e}")
        traceback.print_exc()
        skipped_datasets.append(("CulturalBench", str(e)))


# ============================================================================
# 2. CVQA  (afaji/cvqa) -- text/metadata only
#    Columns: image, ID, Subset ("('Language', 'Country')"), Question,
#             Translated Question, Options (list), Translated Options (list),
#             Label (int index), Category, Image Type, Image Source, License
# ============================================================================
def load_cvqa():
    print("\n" + "=" * 70)
    print("[2/5] Loading CVQA (afaji/cvqa) ...")
    print("=" * 70)
    try:
        from datasets import load_dataset
        import ast

        # Use streaming to avoid downloading all images
        ds = load_dataset("afaji/cvqa", streaming=True)

        count = 0
        for split_name in ds:
            print(f"  Processing split '{split_name}' (streaming) ...")
            for idx, row in enumerate(ds[split_name]):
                item_id = f"cvqa_{split_name}_{idx}"

                question = safe_str(row.get("Question", ""))
                translated_q = safe_str(row.get("Translated Question", ""))
                options = row.get("Options", [])
                translated_options = row.get("Translated Options", [])
                label_idx = row.get("Label", "")
                category = safe_str(row.get("Category", ""))
                subset = safe_str(row.get("Subset", ""))

                # Parse language and country from Subset: "('Language', 'Country')"
                language = ""
                country = ""
                if subset:
                    try:
                        parsed = ast.literal_eval(subset)
                        if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
                            language = str(parsed[0]).strip()
                            country = str(parsed[1]).strip()
                    except (ValueError, SyntaxError):
                        pass

                # Build answer text from label index
                answer = ""
                if isinstance(label_idx, int) and isinstance(options, list) and 0 <= label_idx < len(options):
                    answer = safe_str(options[label_idx])

                # Build options string
                opt_labels = ["A", "B", "C", "D", "E"]
                options_str = ""
                if isinstance(options, list):
                    options_str = " | ".join(
                        f"{opt_labels[i]}: {safe_str(o)}"
                        for i, o in enumerate(options)
                        if i < len(opt_labels)
                    )

                full_content = question
                if options_str:
                    full_content += " | " + options_str
                if translated_q and translated_q != question:
                    full_content += f" | [EN] {translated_q}"
                if category:
                    full_content = f"[{category}] {full_content}"

                all_task_metadata.append({
                    "item_id": item_id,
                    "text": truncate(full_content),
                    "label": answer if answer else str(label_idx),
                    "task_type": "visual_qa",
                    "language": language,
                    "country_region": country,
                    "source_dataset": "CVQA",
                    "split": split_name,
                })
                all_item_content.append({
                    "item_id": item_id,
                    "full_content": full_content,
                })
                count += 1

                if (idx + 1) % 2000 == 0:
                    print(f"    ... processed {idx + 1} rows")

        dataset_stats["CVQA"] = {
            "total_items": count,
        }
        print(f"  -> Collected {count} items from CVQA")

    except Exception as e:
        print(f"  ERROR loading CVQA: {e}")
        traceback.print_exc()
        skipped_datasets.append(("CVQA", str(e)))


# ============================================================================
# 3. ArabicMMLU  (MBZUAI/ArabicMMLU, config='All')
#    Columns: ID, Source, Country, Group, Subject, Level, Question, Context,
#             Answer Key, Option 1..5, is_few_shot
#    Uses streaming to avoid brotli decompression errors.
# ============================================================================
def load_arabicmmlu():
    print("\n" + "=" * 70)
    print("[3/5] Loading ArabicMMLU (MBZUAI/ArabicMMLU, config='All') ...")
    print("=" * 70)
    try:
        from datasets import load_dataset

        ds = load_dataset("MBZUAI/ArabicMMLU", "All", streaming=True)

        count = 0
        for split_name in ds:
            print(f"  Processing split '{split_name}' (streaming) ...")
            for idx, row in enumerate(ds[split_name]):
                item_id = f"arabicmmlu_{split_name}_{idx}"

                question = safe_str(row.get("Question", ""))
                context = safe_str(row.get("Context", ""))
                subject = safe_str(row.get("Subject", ""))
                country = safe_str(row.get("Country", ""))
                level = safe_str(row.get("Level", ""))
                group = safe_str(row.get("Group", ""))
                answer_key = safe_str(row.get("Answer Key", ""))

                # Build options
                options_parts = []
                for i in range(1, 6):
                    opt_val = safe_str(row.get(f"Option {i}", ""))
                    if opt_val:
                        opt_label = chr(ord("A") + i - 1)
                        options_parts.append(f"{opt_label}: {opt_val}")
                options_str = " | ".join(options_parts)

                # Build full content
                full_content = question
                if context:
                    full_content = f"Context: {context}\n{full_content}"
                if options_str:
                    full_content += " | " + options_str

                prefix_parts = []
                if subject:
                    prefix_parts.append(subject)
                if level:
                    prefix_parts.append(level)
                if group:
                    prefix_parts.append(group)
                if prefix_parts:
                    full_content = f"[{' / '.join(prefix_parts)}] {full_content}"

                all_task_metadata.append({
                    "item_id": item_id,
                    "text": truncate(full_content),
                    "label": answer_key,
                    "task_type": "multiple_choice",
                    "language": "ar",
                    "country_region": country if country else "MENA",
                    "source_dataset": "ArabicMMLU",
                    "split": split_name,
                })
                all_item_content.append({
                    "item_id": item_id,
                    "full_content": full_content,
                })
                count += 1

                if (idx + 1) % 2000 == 0:
                    print(f"    ... processed {idx + 1} rows")

        dataset_stats["ArabicMMLU"] = {
            "total_items": count,
        }
        print(f"  -> Collected {count} items from ArabicMMLU")

    except Exception as e:
        print(f"  ERROR loading ArabicMMLU: {e}")
        traceback.print_exc()
        skipped_datasets.append(("ArabicMMLU", str(e)))


# ============================================================================
# 4. EXAMS  (mhardalov/exams, config='multilingual')
#    Columns: id, question (dict with stem, choices), answerKey, info (dict)
#    Only load the 'multilingual' config (core multilingual exam data).
# ============================================================================
def load_exams():
    print("\n" + "=" * 70)
    print("[4/5] Loading EXAMS (mhardalov/exams, config='multilingual') ...")
    print("=" * 70)
    try:
        from datasets import load_dataset

        ds = load_dataset("mhardalov/exams", "multilingual")
        print(f"  Loaded config 'multilingual'. Splits: {list(ds.keys())}")

        count = 0
        for split_name in ds:
            split = ds[split_name]
            cols = split.column_names
            print(f"  Split '{split_name}': {len(split)} rows, columns={cols}")

            for idx, row in enumerate(split):
                item_id = f"exams_{split_name}_{idx}"

                # Parse question dict
                question_obj = row.get("question", {})
                if isinstance(question_obj, dict):
                    stem = safe_str(question_obj.get("stem", ""))
                    choices = question_obj.get("choices", {})
                    if isinstance(choices, dict):
                        labels_list = choices.get("label", [])
                        texts_list = choices.get("text", [])
                        options_str = " | ".join(
                            f"{l}: {t}" for l, t in zip(labels_list, texts_list)
                        )
                    else:
                        options_str = safe_str(choices)
                else:
                    stem = safe_str(question_obj)
                    options_str = ""

                answer_key = safe_str(row.get("answerKey", ""))

                # Parse info dict for language, subject
                info = row.get("info", {})
                language = ""
                subject = ""
                if isinstance(info, dict):
                    language = safe_str(info.get("language", ""))
                    subject = safe_str(info.get("subject", ""))

                full_content = stem
                if options_str:
                    full_content += " | " + options_str
                if subject:
                    full_content = f"[{subject}] {full_content}"

                # Derive country from language where possible
                lang_to_country = {
                    "Arabic": "MENA",
                    "Bulgarian": "Bulgaria",
                    "Croatian": "Croatia",
                    "Hungarian": "Hungary",
                    "Italian": "Italy",
                    "Lithuanian": "Lithuania",
                    "Macedonian": "North Macedonia",
                    "North Macedonian": "North Macedonia",
                    "Polish": "Poland",
                    "Portuguese": "Portugal",
                    "Serbian": "Serbia",
                    "Turkish": "Turkey",
                    "Vietnamese": "Vietnam",
                    "Albanian": "Albania",
                    "French": "France",
                    "German": "Germany",
                    "Spanish": "Spain",
                }
                country = lang_to_country.get(language, "")

                all_task_metadata.append({
                    "item_id": item_id,
                    "text": truncate(full_content),
                    "label": answer_key,
                    "task_type": "exam_qa",
                    "language": language,
                    "country_region": country,
                    "source_dataset": "EXAMS",
                    "split": split_name,
                })
                all_item_content.append({
                    "item_id": item_id,
                    "full_content": full_content,
                })
                count += 1

        dataset_stats["EXAMS"] = {
            "total_items": count,
            "splits": list(ds.keys()),
        }
        print(f"  -> Collected {count} items from EXAMS")

    except Exception as e:
        print(f"  ERROR loading EXAMS: {e}")
        traceback.print_exc()
        skipped_datasets.append(("EXAMS", str(e)))


# ============================================================================
# 5. INCLUDE  (CohereLabs/include-base-44)
#    Must load each language config separately.
#    Columns: language, country, domain, subject, regional_feature, level,
#             question, option_a, option_b, option_c, option_d, answer (index str)
# ============================================================================
INCLUDE_CONFIGS = [
    "Albanian", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian",
    "Bengali", "Bulgarian", "Chinese", "Croatian", "Dutch", "Dutch - Flemish",
    "Dutch-Flemish", "Estonian", "Finnish", "French", "Georgian", "German",
    "Greek", "Hebrew", "Hindi", "Hungarian", "Indonesian", "Italian",
    "Japanese", "Kazakh", "Korean", "Lithuanian", "Malay", "Malayalam",
    "Nepali", "North Macedonian", "Persian", "Polish", "Portuguese",
    "Russian", "Serbian", "Spanish", "Tagalog", "Tamil", "Telugu",
    "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese",
]


def load_include():
    print("\n" + "=" * 70)
    print("[5/5] Loading INCLUDE (CohereLabs/include-base-44) ...")
    print("=" * 70)
    try:
        from datasets import load_dataset, get_dataset_config_names

        dataset_id = "CohereLabs/include-base-44"

        # Get actual configs from HuggingFace
        try:
            configs = get_dataset_config_names(dataset_id)
            print(f"  Available configs ({len(configs)}): {configs[:10]}...")
        except Exception:
            configs = INCLUDE_CONFIGS
            print(f"  Using hardcoded config list ({len(configs)} languages)")

        count = 0
        loaded_configs = []
        failed_configs = []

        for config in configs:
            try:
                ds = load_dataset(dataset_id, config)
                loaded_configs.append(config)

                for split_name in ds:
                    split = ds[split_name]
                    if not count:
                        print(f"  First config '{config}' split '{split_name}': "
                              f"{len(split)} rows, columns={split.column_names}")

                    for idx, row in enumerate(split):
                        item_id = f"include_{config.lower().replace(' ', '_')}_{split_name}_{idx}"

                        question = safe_str(row.get("question", ""))
                        answer = safe_str(row.get("answer", ""))
                        language = safe_str(row.get("language", config))
                        country = safe_str(row.get("country", ""))
                        subject = safe_str(row.get("subject", ""))
                        domain = safe_str(row.get("domain", ""))
                        level = safe_str(row.get("level", ""))

                        # Build options
                        opt_a = safe_str(row.get("option_a", ""))
                        opt_b = safe_str(row.get("option_b", ""))
                        opt_c = safe_str(row.get("option_c", ""))
                        opt_d = safe_str(row.get("option_d", ""))
                        options_parts = []
                        for lbl, val in [("A", opt_a), ("B", opt_b), ("C", opt_c), ("D", opt_d)]:
                            if val:
                                options_parts.append(f"{lbl}: {val}")
                        options_str = " | ".join(options_parts)

                        # Map answer index to letter
                        answer_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
                        answer_letter = answer_map.get(answer, answer)

                        full_content = question
                        if options_str:
                            full_content += " | " + options_str
                        prefix_parts = []
                        if subject:
                            prefix_parts.append(subject)
                        if domain:
                            prefix_parts.append(domain)
                        if level:
                            prefix_parts.append(level)
                        if prefix_parts:
                            full_content = f"[{' / '.join(prefix_parts)}] {full_content}"

                        all_task_metadata.append({
                            "item_id": item_id,
                            "text": truncate(full_content),
                            "label": answer_letter,
                            "task_type": "exam_qa",
                            "language": language if language else config,
                            "country_region": country,
                            "source_dataset": "INCLUDE",
                            "split": split_name,
                        })
                        all_item_content.append({
                            "item_id": item_id,
                            "full_content": full_content,
                        })
                        count += 1

                if len(loaded_configs) % 10 == 0:
                    print(f"    ... loaded {len(loaded_configs)}/{len(configs)} configs, "
                          f"{count} items so far")

            except Exception as e_cfg:
                failed_configs.append((config, str(e_cfg)[:100]))
                print(f"    WARNING: Could not load config '{config}': {str(e_cfg)[:100]}")

        dataset_stats["INCLUDE"] = {
            "total_items": count,
            "configs_loaded": len(loaded_configs),
            "configs_failed": len(failed_configs),
        }
        if failed_configs:
            print(f"  Failed configs: {[c for c, _ in failed_configs]}")
        print(f"  -> Collected {count} items from INCLUDE "
              f"({len(loaded_configs)} configs loaded, {len(failed_configs)} failed)")

    except Exception as e:
        print(f"  ERROR loading INCLUDE: {e}")
        traceback.print_exc()
        skipped_datasets.append(("INCLUDE", str(e)))


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Cultural Evaluation Data Collection")
    print("Building task_metadata.csv and item_content.csv")
    print("=" * 70)

    # Load each dataset
    load_culturalbench()
    load_cvqa()
    load_arabicmmlu()
    load_exams()
    load_include()

    # -----------------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Writing output files ...")
    print("=" * 70)

    if not all_task_metadata:
        print("  WARNING: No data collected from any dataset!")
        sys.exit(1)

    # task_metadata.csv
    df_meta = pd.DataFrame(all_task_metadata)
    meta_path = os.path.join(PROCESSED_DIR, "task_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"  task_metadata.csv: {len(df_meta)} rows -> {meta_path}")

    # item_content.csv
    df_content = pd.DataFrame(all_item_content)
    content_path = os.path.join(PROCESSED_DIR, "item_content.csv")
    df_content.to_csv(content_path, index=False)
    print(f"  item_content.csv: {len(df_content)} rows -> {content_path}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("CULTURAL EVALUATION DATA - SUMMARY STATISTICS")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # Per-dataset stats
    summary_lines.append("DATASETS LOADED:")
    for ds_name, stats in dataset_stats.items():
        summary_lines.append(f"  {ds_name}: {stats['total_items']} items")
        for k, v in stats.items():
            if k != "total_items":
                summary_lines.append(f"    {k}: {v}")
    summary_lines.append("")

    if skipped_datasets:
        summary_lines.append("DATASETS SKIPPED (could not load):")
        for ds_name, reason in skipped_datasets:
            summary_lines.append(f"  {ds_name}: {reason[:200]}")
        summary_lines.append("")

    # Overall stats
    summary_lines.append(f"TOTAL ITEMS: {len(df_meta)}")
    summary_lines.append("")

    # Per-source breakdown
    summary_lines.append("ITEMS PER SOURCE DATASET:")
    for src, cnt in df_meta["source_dataset"].value_counts().items():
        summary_lines.append(f"  {src}: {cnt:,}")
    summary_lines.append("")

    # Language breakdown
    summary_lines.append("ITEMS PER LANGUAGE (top 30):")
    lang_counts = df_meta["language"].value_counts().head(30)
    for lang, cnt in lang_counts.items():
        summary_lines.append(f"  {lang}: {cnt:,}")
    summary_lines.append(f"  ... ({df_meta['language'].nunique()} unique languages total)")
    summary_lines.append("")

    # Region breakdown
    summary_lines.append("ITEMS PER COUNTRY/REGION (top 30):")
    region_series = df_meta["country_region"].replace("", np.nan).dropna()
    region_counts = region_series.value_counts().head(30)
    for reg, cnt in region_counts.items():
        summary_lines.append(f"  {reg}: {cnt:,}")
    summary_lines.append(f"  ... ({region_series.nunique()} unique regions total)")
    summary_lines.append("")

    # Task type breakdown
    summary_lines.append("ITEMS PER TASK TYPE:")
    for tt, cnt in df_meta["task_type"].value_counts().items():
        summary_lines.append(f"  {tt}: {cnt:,}")
    summary_lines.append("")

    # Split breakdown
    summary_lines.append("ITEMS PER SPLIT:")
    for sp, cnt in df_meta["split"].value_counts().items():
        summary_lines.append(f"  {sp}: {cnt:,}")

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(PROCESSED_DIR, "summary_statistics.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  summary_statistics.txt -> {summary_path}")
    print(f"\n{summary_text}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
