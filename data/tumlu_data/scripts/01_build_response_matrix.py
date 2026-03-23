"""
Build TUMLU response matrix from Turkic Multilingual Language Understanding data.

Data source:
  - https://github.com/TURNA-NLP/TUMLU (assumed cloned to raw/)
  - 12 language variants (9 base languages, some with script variants)
  - ~14 models x 2 prompting variants (CoT / no-CoT)
  - Multiple subjects per language

Score format:
  - Binary 0/1: whether extracted answer letter matches gold answer
  - Multilingual answer extraction with Turkic language patterns

Outputs:
  - raw/TUMLU/: Cloned GitHub repo
  - processed/response_matrix.csv: Models (rows) x items (columns)
  - processed/item_content.csv: Per-item metadata (question + choices)
"""

import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

REPO_URL = "https://github.com/ceferisbarov/TUMLU.git"

# Model name normalization
MODEL_NAMES = {
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "claude-3-5-haiku-20241022": "claude-3.5-haiku",
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "deepseek-chat": "deepseek-chat",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "google/gemma-2-27b-it": "gemma-2-27b",
    "google/gemma-2-9b-it": "gemma-2-9b",
    "gpt-4o-2024-11-20": "gpt-4o",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "llama-3.1-405b",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-hyperbolic": "llama-3.1-405b-hyp",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-deepinfra": "llama-3.1-405b-deepinfra",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama-3.1-70b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b",
}


def extract_answer(output_text):
    """Extract the answer letter (A/B/C/D) from model output.

    Handles multilingual patterns from Turkic languages.
    """
    if not output_text or not isinstance(output_text, str):
        return None

    text = output_text.strip()

    # Pattern 1: Just the letter alone
    if re.match(r'^[A-Da-d][\.\)\s]*$', text):
        return text[0].upper()

    # Pattern 2: "Cavab: X", "Answer: X", "Cevap: X", etc.
    m = re.search(
        r'(?:cavab|answer|cevap|\u0436\u0430\u0443\u0430\u043f|jawap|\u062c\u0627\u06cb\u0627\u067e'
        r'|\u062c\u0648\u0627\u0628|javobi|\u04b9\u0430\u0432\u0430\u043f|\u0436\u0430\u0432\u0430\u043f'
        r'|\u0436\u0430\u0432\u0430\u0431|cawab|cevab\u0131'
        r'|d\u00fczg\u00fcn cavab)\s*[:\u0EDF]\s*\**\s*([A-Da-d])',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 2b: "X variantidir" / "X варианты" / "X varianti"
    matches_variant = re.findall(
        r'\b([A-Da-d])\s*(?:variant\u0131d\u0131r|\u0432\u0430\u0440\u0438\u0430\u043d\u0442\u044b'
        r'|varianti|variant\u0131)',
        text, re.IGNORECASE
    )
    if matches_variant:
        return matches_variant[-1].upper()

    # Pattern 2c: "cavab X variantidir" without colon
    m = re.search(
        r'(?:cavab|cevap|\u0436\u0430\u0443\u0430\u043f|jawap'
        r'|d\u00fczg\u00fcn cavab|do\u011fru cevap'
        r'|\u0442\u043e\u0493\u0440\u044b \u04b9\u0430\u0432\u0430\u043f)\s+\**([A-Da-d])\b',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 3: **X)** or **X.**
    m = re.search(r'\*\*([A-Da-d])[\)\.]', text)
    if m:
        return m.group(1).upper()

    # Pattern 3b: **X** standalone
    m = re.search(r'\*\*([A-Da-d])\*\*', text)
    if m:
        return m.group(1).upper()

    # Pattern 4: Standalone letter at start "A)" or "B."
    m = re.match(r'\s*([A-Da-d])\s*[\)\.]', text)
    if m:
        return m.group(1).upper()

    # Pattern 5: Last occurrence of a letter option reference
    matches = re.findall(r'\b([A-Da-d])\)', text)
    if matches:
        return matches[-1].upper()

    # Pattern 6: First letter if very short response
    m = re.match(r'\s*([A-Da-d])\b', text)
    if m and len(text) < 20:
        return m.group(1).upper()

    return None


def normalize_subject(filename):
    """Normalize subject filename to a clean identifier."""
    name = filename.replace('.json', '').lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def normalize_language(lang):
    """Normalize language directory name."""
    return lang.lower().replace('-', '_')


def discover_models(variant_dir):
    """Discover all model paths under a variant directory.

    Some models are flat (e.g., gpt-4o-2024-11-20/Physics.json)
    while others are nested (e.g., Qwen/Qwen2.5-72B-Instruct/Physics.json).
    """
    models = []
    if not os.path.exists(variant_dir):
        return models

    for entry in sorted(os.listdir(variant_dir)):
        entry_path = os.path.join(variant_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        contents = os.listdir(entry_path)
        has_json = any(f.endswith('.json') for f in contents)
        has_subdirs = any(os.path.isdir(os.path.join(entry_path, f)) for f in contents)

        if has_json:
            models.append((entry, entry_path))

        if has_subdirs:
            for sub in sorted(contents):
                sub_path = os.path.join(entry_path, sub)
                if os.path.isdir(sub_path):
                    sub_contents = os.listdir(sub_path)
                    if any(f.endswith('.json') for f in sub_contents):
                        model_key = f"{entry}/{sub}"
                        models.append((model_key, sub_path))

    return models


def main():
    print("TUMLU Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Clone repo
    tumlu_dir = os.path.join(RAW_DIR, "TUMLU")
    if not os.path.exists(tumlu_dir):
        print("  Cloning TUMLU...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, tumlu_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            sys.exit(1)

    data_dir = os.path.join(tumlu_dir, "data")
    if not os.path.exists(data_dir):
        print(f"  No data directory found at {data_dir}")
        sys.exit(1)

    languages = sorted(os.listdir(data_dir))
    print(f"  Languages: {languages}")

    # Step 2: Register all items
    item_registry = {}
    item_order = []

    for lang in languages:
        lang_norm = normalize_language(lang)
        ref_dir = os.path.join(data_dir, lang, "outputs", "no_cot_instruct")
        if not os.path.exists(ref_dir):
            continue

        ref_models = discover_models(ref_dir)
        if not ref_models:
            continue

        # Prefer gpt-4o as reference
        ref_key, ref_path = ref_models[0]
        for mk, mp in ref_models:
            if 'gpt-4o' in mk:
                ref_key, ref_path = mk, mp
                break

        for subj_file in sorted(os.listdir(ref_path)):
            if not subj_file.endswith('.json'):
                continue
            subj_norm = normalize_subject(subj_file)

            with open(os.path.join(ref_path, subj_file), encoding="utf-8") as f:
                items = json.load(f)
            for idx, item in enumerate(items):
                item_id = f"{lang_norm}_{subj_norm}_{idx}"
                question = item.get('question', '')
                choices = item.get('choices', [])
                choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                choices_str = " | ".join(
                    f"{choice_labels[i]}) {c}" for i, c in enumerate(choices)
                )
                content = f"{question} [{choices_str}]"
                item_registry[item_id] = {
                    'content': content,
                    'answer': item.get('answer', ''),
                    'language': lang,
                    'subject': subj_norm,
                }
                item_order.append(item_id)

    print(f"  Items registered: {len(item_order)}")

    # Fill gaps from other models
    for lang in languages:
        lang_norm = normalize_language(lang)
        for variant in ["no_cot_instruct"]:
            vdir = os.path.join(data_dir, lang, "outputs", variant)
            if not os.path.exists(vdir):
                continue
            for mk, mp in discover_models(vdir):
                for subj_file in sorted(os.listdir(mp)):
                    if not subj_file.endswith('.json'):
                        continue
                    subj_norm = normalize_subject(subj_file)
                    test_id = f"{lang_norm}_{subj_norm}_0"
                    if test_id not in item_registry:
                        with open(os.path.join(mp, subj_file), encoding="utf-8") as f:
                            items = json.load(f)
                        for idx, item in enumerate(items):
                            item_id = f"{lang_norm}_{subj_norm}_{idx}"
                            if item_id in item_registry:
                                continue
                            question = item.get('question', '')
                            choices = item.get('choices', [])
                            choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                            choices_str = " | ".join(
                                f"{choice_labels[i]}) {c}" for i, c in enumerate(choices)
                            )
                            content = f"{question} [{choices_str}]"
                            item_registry[item_id] = {
                                'content': content,
                                'answer': item.get('answer', ''),
                                'language': lang,
                                'subject': subj_norm,
                            }
                            item_order.append(item_id)

    print(f"  Items after filling gaps: {len(item_order)}")

    # Step 3: Build response matrix
    response_data = {}

    for variant in ["no_cot_instruct", "cot_instruct"]:
        variant_suffix = "" if variant == "no_cot_instruct" else "_cot"

        for lang in languages:
            lang_norm = normalize_language(lang)
            variant_dir = os.path.join(data_dir, lang, "outputs", variant)

            for model_key, model_path in discover_models(variant_dir):
                model_name = MODEL_NAMES.get(model_key, model_key) + variant_suffix

                if model_name not in response_data:
                    response_data[model_name] = {}

                for subj_file in sorted(os.listdir(model_path)):
                    if not subj_file.endswith('.json'):
                        continue
                    subj_norm = normalize_subject(subj_file)

                    with open(os.path.join(model_path, subj_file), encoding="utf-8") as f:
                        items = json.load(f)
                    for idx, item in enumerate(items):
                        item_id = f"{lang_norm}_{subj_norm}_{idx}"
                        if item_id not in item_registry:
                            continue

                        gold = item.get('answer', '').strip().upper()
                        predicted = extract_answer(item.get('output', ''))
                        correct = 1 if (predicted is not None and predicted == gold) else 0
                        response_data[model_name][item_id] = correct

    # Step 4: Save response matrix
    models = sorted(response_data.keys())
    matrix = []
    for model in models:
        row = [model]
        for item_id in item_order:
            val = response_data[model].get(item_id, '')
            row.append(val)
        matrix.append(row)

    columns = ['model'] + item_order
    df = pd.DataFrame(matrix, columns=columns)
    df.to_csv(os.path.join(PROCESSED_DIR, "response_matrix.csv"), index=False)
    print(f"\n  Response matrix saved: {df.shape[0]} models x {df.shape[1] - 1} items")

    # Step 5: Save item_content.csv
    with open(os.path.join(PROCESSED_DIR, "item_content.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'content'])
        for item_id in item_order:
            writer.writerow([item_id, item_registry[item_id]['content']])
    print(f"  Item content saved: {len(item_order)} items")

    # Step 6: Summary
    print(f"\n{'=' * 60}")
    print(f"TUMLU SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Models: {len(models)}")
    print(f"  Items: {len(item_order)}")

    lang_items = defaultdict(int)
    for item_id in item_order:
        lang = item_registry[item_id]['language']
        lang_items[lang] += 1
    print(f"\n  Items per language:")
    for lang in sorted(lang_items):
        print(f"    {lang}: {lang_items[lang]}")

    print(f"\n  Accuracy by model:")
    for model in models:
        vals = [response_data[model][iid] for iid in item_order if iid in response_data[model]]
        if vals:
            acc = sum(vals) / len(vals)
            print(f"    {model}: {acc:.3f} ({len(vals)} items)")


if __name__ == "__main__":
    main()
