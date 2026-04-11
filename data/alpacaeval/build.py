"""
Build AlpacaEval 2.0 response matrices from per-model per-item annotations.

Data source:
  - GitHub: tatsu-lab/alpaca_eval, results/ directory
  - Each model has results/<model>/weighted_alpaca_eval_gpt4_turbo/annotations.json
  - Each annotations.json contains 805 entries (one per instruction prompt)
  - Each entry has a 'preference' float in [1, 2]:
      preference close to 1 => reference model (gpt4_1106_preview) wins
      preference close to 2 => evaluated model wins
  - Binary win: preference > 1.5

AlpacaEval 2.0 overview:
  - 805 instruction-following prompts from 5 datasets
    (helpful_base, vicuna, koala, selfinstruct, oasst)
  - GPT-4 Turbo as judge (weighted annotator)
  - Reference model: gpt4_1106_preview
  - Length-controlled win rates to mitigate length bias

Outputs:
  - response_matrix.csv: Binary win/loss (models x items), 1=model wins
  - response_matrix_preference.csv: Raw preference floats (models x items)
  - item_metadata.csv: Per-item metadata (instruction, dataset, index)
  - model_summary.csv: Per-model aggregate statistics
"""

INFO = {
    'description': 'Build AlpacaEval 2.0 response matrices from per-model per-item annotations',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2404.04475',
    'data_source_url': 'https://github.com/tatsu-lab/alpaca_eval',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'Apache-2.0',
    'citation': """@misc{dubois2025lengthcontrolledalpacaevalsimpleway,
      title={Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators}, 
      author={Yann Dubois and Balázs Galambosi and Percy Liang and Tatsunori B. Hashimoto},
      year={2025},
      eprint={2404.04475},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.04475}, 
}""",
    'tags': ['reasoning'],
}


from pathlib import Path
import os
import json
import urllib.request
import urllib.error
import time
import sys

import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# GitHub raw content base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main"

# Annotator subdirectory for AlpacaEval 2.0
ANNOTATOR_DIR = "weighted_alpaca_eval_gpt4_turbo"

# All 228 model directories in tatsu-lab/alpaca_eval/results/
ALL_MODELS = [
    "Conifer-7B-DPO",
    "Contextual-KTO-Mistral-PairRM",
    "Ein-70B-v0.1",
    "FsfairX-Zephyr-Chat-v0.1",
    "FuseChat-Gemma-2-9B-Instruct",
    "FuseChat-Llama-3.1-8B-Instruct",
    "FuseChat-Llama-3.2-1B-Instruct",
    "FuseChat-Llama-3.2-3B-Instruct",
    "FuseChat-Qwen-2.5-7B-Instruct",
    "GPO-Llama-3-8B-Instruct-GPM-2B",
    "Infinity-Instruct-3M-0613-Llama3-70B",
    "Infinity-Instruct-3M-0613-Mistral-7B",
    "Infinity-Instruct-3M-0625-Llama3-70B",
    "Infinity-Instruct-3M-0625-Llama3-8B",
    "Infinity-Instruct-3M-0625-Mistral-7B",
    "Infinity-Instruct-3M-0625-Qwen2-7B",
    "Infinity-Instruct-3M-0625-Yi-1.5-9B",
    "Infinity-Instruct-7M-Gen-Llama3_1-70B",
    "Infinity-Instruct-7M-Gen-Llama3_1-8B",
    "Infinity-Instruct-7M-Gen-mistral-7B",
    "LMCocktail-10.7B-v1",
    "Llama-3-8B-Instruct-SkillMix",
    "Llama-3-Instruct-8B-RainbowPO",
    "Llama-3-Instruct-8B-SimPO",
    "Llama-3-Instruct-8B-SimPO-ExPO",
    "Llama-3-Instruct-8B-WPO-HB-v2",
    "Llama3-PBM-Nova-70B",
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3.1-405B-Instruct-Turbo",
    "Meta-Llama-3.1-70B-Instruct-Turbo",
    "Meta-Llama-3.1-8B-Instruct-Turbo",
    "Mistral-7B+RAHF-DUAL+LoRA",
    "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.3",
    "Mistral-7B-ReMax-v0.1",
    "Mixtral-8x22B-Instruct-v0.1",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x7B-Instruct-v0.1_concise",
    "Mixtral-8x7B-Instruct-v0.1_verbose",
    "Nanbeige-Plus-Chat-v0.1",
    "Nanbeige2-16B-Chat",
    "Nanbeige2-8B-Chat",
    "NullModel",
    "OpenHermes-2.5-Mistral-7B",
    "Qwen-14B-Chat",
    "Qwen1.5-1.8B-Chat",
    "Qwen1.5-110B-Chat",
    "Qwen1.5-14B-Chat",
    "Qwen1.5-72B-Chat",
    "Qwen1.5-7B-Chat",
    "Qwen2-72B-Instruct",
    "REBEL-Llama-3-8B-Instruct",
    "REBEL-Llama-3-8B-Instruct-Armo",
    "SPPO-Gemma-2-9B-It-PairRM",
    "SPPO-Llama-3-8B-Instruct-GPM-2B",
    "SPPO-Llama-3-Instruct-8B-PairRM",
    "SPPO-Mistral7B-PairRM",
    "SPPO-Mistral7B-PairRM-ExPO",
    "Samba-CoE-v0.1",
    "Samba-CoE-v0.2",
    "Samba-CoE-v0.2-best-of-16",
    "SelfMoA_gemma-2-9b-it-SimPO",
    "SelfMoA_gemma-2-9b-it-WPO-HB",
    "Shopee-SlimMoA-v1",
    "Snorkel-Mistral-PairRM-DPO",
    "Snorkel-Mistral-PairRM-DPO-best-of-16",
    "Starling-LM-7B-alpha",
    "Starling-LM-7B-alpha-ExPO",
    "Starling-LM-7B-beta-ExPO",
    "Storm-7B",
    "Storm-7B-best-of-64",
    "TOA",
    "TempNet-LLaMA2-Chat-13B-v0.1",
    "TempNet-LLaMA2-Chat-70B-v0.1",
    "TempNet-LLaMA2-Chat-7B-v0.1",
    "Together-MoA",
    "Together-MoA-Lite",
    "Yi-34B-Chat",
    "airoboros-33b",
    "airoboros-65b",
    "aligner-2b_claude-3-opus-20240229",
    "aligner-2b_gpt-4-turbo-2024-04-09",
    "aligner-2b_qwen1.5-72b-chat",
    "alpaca-7b",
    "alpaca-7b-neft",
    "alpaca-7b_concise",
    "alpaca-7b_verbose",
    "alpaca-farm-ppo-human",
    "alpaca-farm-ppo-sim-gpt4-20k",
    "baichuan-13b-chat",
    "baize-v2-13b",
    "baize-v2-7b",
    "bedrock_claude",
    "blendaxai-gm-l3-v35",
    "blendaxai-gm-l6-vo31",
    "causallm-14b",
    "chatglm2-6b",
    "claude",
    "claude-2",
    "claude-2.1",
    "claude-2.1_concise",
    "claude-2.1_verbose",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-instant-1.2",
    "claude2-alpaca-13b",
    "cohere",
    "cut-13b",
    "dbrx-instruct",
    "deepseek-llm-67b-chat",
    "deita-7b-v1.0",
    "dolphin-2.2.1-mistral-7b",
    "evo-7b",
    "evo-v2-7b",
    "falcon-40b-instruct",
    "falcon-7b-instruct",
    "gemini-pro",
    "gemma-2-9b-it-DPO",
    "gemma-2-9b-it-SimPO",
    "gemma-2-9b-it-WPO-HB",
    "gemma-2b-it",
    "gemma-7b-it",
    "ghost-7b-alpha",
    "ghost-8b-beta-disl-0x5",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-1106_concise",
    "gpt-3.5-turbo-1106_verbose",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt35_turbo_instruct",
    "gpt4",
    "gpt4_0314",
    "gpt4_0613",
    "gpt4_0613_concise",
    "gpt4_0613_verbose",
    "gpt4_1106_preview",
    "gpt4_1106_preview_concise",
    "gpt4_1106_preview_verbose",
    "gpt4_gamed",
    "guanaco-13b",
    "guanaco-33b",
    "guanaco-65b",
    "guanaco-7b",
    "higgs-llama-3-70b-v2",
    "humpback-llama-65b",
    "humpback-llama2-70b",
    "internlm2-chat-20b-ExPO",
    "internlm2-chat-20b-ppo",
    "internlm2-chat-7b-ExPO",
    "jina-chat",
    "llama-2-13b-chat-hf",
    "llama-2-70b-chat-hf",
    "llama-2-7b-chat-hf",
    "llama-2-chat-7b-evol70k-neft",
    "merlinite-7B-AOT",
    "minichat-1.5-3b",
    "minichat-3b",
    "minotaur-13b",
    "mistral-large-2402",
    "mistral-medium",
    "mistral-orpo-beta",
    "nous-hermes-13b",
    "oasst-rlhf-llama-33b",
    "oasst-sft-llama-33b",
    "oasst-sft-pythia-12b",
    "openbuddy-falcon-40b-v9",
    "openbuddy-falcon-7b-v6",
    "openbuddy-llama-30b-v7.1",
    "openbuddy-llama-65b-v8",
    "openbuddy-llama2-13b-v11.1",
    "openbuddy-llama2-70b-v10.1",
    "openchat-13b",
    "openchat-v2-13b",
    "openchat-v2-w-13b",
    "openchat-v3.1-13b",
    "openchat8192-13b",
    "opencoderplus-15b",
    "openpipe-moa-gpt-4-turbo-v1",
    "pairrm-Yi-34B-Chat",
    "pairrm-tulu-2-13b",
    "pairrm-tulu-2-70b",
    "pairrm-zephyr-7b-beta",
    "phi-2",
    "phi-2-dpo",
    "phi-2-sft",
    "platolm-7b",
    "pythia-12b-mix-sft",
    "recycled-wizardlm-7b-v1.0",
    "recycled-wizardlm-7b-v2.0",
    "text_davinci_001",
    "text_davinci_003",
    "tulu-2-dpo-13b",
    "tulu-2-dpo-13b-ExPO",
    "tulu-2-dpo-70b",
    "tulu-2-dpo-70b-ExPO",
    "tulu-2-dpo-7b",
    "tulu-2-dpo-7b-ExPO",
    "ultralm-13b",
    "ultralm-13b-best-of-16",
    "ultralm-13b-v2.0",
    "ultralm-13b-v2.0-best-of-16",
    "vicuna-13b",
    "vicuna-13b-v1.3",
    "vicuna-13b-v1.5",
    "vicuna-13b-v1.5-togetherai",
    "vicuna-33b-v1.3",
    "vicuna-7b",
    "vicuna-7b-v1.3",
    "vicuna-7b-v1.5",
    "wizardlm-13b",
    "wizardlm-13b-v1.1",
    "wizardlm-13b-v1.2",
    "wizardlm-70b",
    "xwinlm-13b-v0.1",
    "xwinlm-70b-v0.1",
    "xwinlm-70b-v0.3",
    "xwinlm-7b-v0.1",
    "yi-large-preview",
    "zephyr-7b-alpha",
    "zephyr-7b-alpha-ExPO",
    "zephyr-7b-beta",
    "zephyr-7b-beta-ExPO",
]


def download_file(url, dest_path, retries=3, delay=1.0):
    """Download a file from URL with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return False
    return False


def download_annotations():
    """Download weighted_alpaca_eval_gpt4_turbo annotations for all models."""
    print("\nSTEP 1: Downloading AlpacaEval 2.0 annotations from GitHub")
    print("-" * 60)
    print(f"  Source: {GITHUB_RAW_BASE}/results/<model>/{ANNOTATOR_DIR}/annotations.json")
    print(f"  Models to download: {len(ALL_MODELS)}")

    downloaded = 0
    skipped = 0
    failed = 0
    failed_models = []

    for i, model in enumerate(ALL_MODELS):
        dest = os.path.join(RAW_DIR, f"{model}__annotations.json")

        # Skip if already downloaded and non-empty
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            skipped += 1
            continue

        url = (
            f"{GITHUB_RAW_BASE}/results/{model}/"
            f"{ANNOTATOR_DIR}/annotations.json"
        )

        success = download_file(url, dest)
        if success:
            downloaded += 1
        else:
            failed += 1
            failed_models.append(model)

        # Progress update every 20 models
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(ALL_MODELS)} "
                  f"(downloaded={downloaded}, cached={skipped}, failed={failed})")
            time.sleep(0.3)

    print(f"\n  Downloaded: {downloaded}")
    print(f"  Cached (skipped): {skipped}")
    print(f"  Failed: {failed}")
    if failed_models:
        print(f"  Failed models: {failed_models[:10]}"
              + (" ..." if len(failed_models) > 10 else ""))

    return failed_models


def download_leaderboard():
    """Download the official AlpacaEval 2.0 leaderboard CSV."""
    print("\nSTEP 2: Downloading leaderboard CSV")
    print("-" * 60)

    url = (
        f"{GITHUB_RAW_BASE}/src/alpaca_eval/leaderboards/"
        "data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv"
    )
    dest = os.path.join(RAW_DIR, "leaderboard.csv")

    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        print(f"  Cached: {dest}")
    else:
        success = download_file(url, dest)
        if success:
            print(f"  Saved: {dest}")
        else:
            print(f"  FAILED to download leaderboard")


def parse_annotations(json_path):
    """Parse a model's annotations JSON into per-item scores.

    Returns:
        dict mapping instruction text -> {
            'preference': float (1-2),
            'dataset': str,
            'generator_2': str (model name)
        }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    items = {}
    for entry in data:
        instruction = entry["instruction"]
        preference = entry.get("preference", None)
        if preference is not None:
            try:
                preference = float(preference)
                # Treat -1 as missing/error (invalid annotation)
                if preference < 0:
                    preference = None
            except (ValueError, TypeError):
                preference = None

        items[instruction] = {
            "preference": preference,
            "dataset": entry.get("dataset", ""),
            "generator_2": entry.get("generator_2", ""),
        }
    return items


def build_response_matrices():
    """Build response matrices from downloaded annotation files."""
    print("\nSTEP 3: Building response matrices")
    print("-" * 60)

    # Parse all model annotations
    all_model_data = {}
    all_instructions = set()
    successful_models = []

    for model in ALL_MODELS:
        json_path = os.path.join(RAW_DIR, f"{model}__annotations.json")
        if not os.path.exists(json_path) or os.path.getsize(json_path) < 1000:
            continue

        try:
            items = parse_annotations(json_path)
            all_model_data[model] = items
            all_instructions.update(items.keys())
            successful_models.append(model)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Failed to parse {model}: {e}")

    print(f"  Successfully parsed: {len(successful_models)} models")
    print(f"  Total unique instructions: {len(all_instructions)}")

    # Sort instructions for consistent ordering
    # Use a stable ordering: sort by the instruction text
    instructions = sorted(all_instructions)

    # Create instruction index mapping (0-based integer IDs)
    instr_to_idx = {instr: i for i, instr in enumerate(instructions)}

    # Build preference matrix (models x items)
    pref_data = {}
    for model in successful_models:
        model_items = all_model_data[model]
        pref_data[model] = [
            model_items.get(instr, {}).get("preference", None)
            for instr in instructions
        ]

    pref_df = pd.DataFrame(pref_data, index=range(len(instructions)))
    pref_df.index.name = "item_idx"

    # Build binary win matrix: 1 if model wins (preference > 1.5), 0 otherwise
    win_df = pref_df.map(
        lambda x: 1 if x is not None and not pd.isna(x) and x > 1.5 else (
            0 if x is not None and not pd.isna(x) else None
        )
    )

    # ---- Print comprehensive statistics ----

    n_models = len(successful_models)
    n_items = len(instructions)
    total_cells = n_models * n_items

    print(f"\n{'='*60}")
    print(f"  RESPONSE MATRIX STATISTICS")
    print(f"{'='*60}")
    print(f"  Models:          {n_models}")
    print(f"  Items:           {n_items}")
    print(f"  Matrix dims:     {n_models} x {n_items}")
    print(f"  Total cells:     {total_cells:,}")

    # Fill rate
    n_valid = pref_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    fill_rate = n_valid / total_cells if total_cells > 0 else 0
    print(f"  Valid cells:     {n_valid:,} ({n_valid/total_cells*100:.1f}%)")
    print(f"  Missing cells:   {n_missing:,} ({n_missing/total_cells*100:.1f}%)")
    print(f"  Fill rate:       {fill_rate*100:.1f}%")

    # Binary win statistics
    n_wins = int(win_df.sum().sum())
    n_losses = int((win_df == 0).sum().sum())
    print(f"\n  Binary wins:     {n_wins:,} ({n_wins/n_valid*100:.1f}% of valid)")
    print(f"  Binary losses:   {n_losses:,} ({n_losses/n_valid*100:.1f}% of valid)")

    # Preference distribution
    all_prefs = pref_df.values.flatten()
    valid_prefs = all_prefs[~pd.isna(all_prefs)].astype(float)
    print(f"\n  Preference distribution (raw float, 1=ref wins, 2=model wins):")
    print(f"    Mean:   {np.mean(valid_prefs):.4f}")
    print(f"    Median: {np.median(valid_prefs):.4f}")
    print(f"    Std:    {np.std(valid_prefs):.4f}")
    print(f"    Min:    {np.min(valid_prefs):.4f}")
    print(f"    Max:    {np.max(valid_prefs):.4f}")

    # Preference histogram (binned)
    print(f"\n  Preference histogram (binned):")
    bins = [(1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, 1.5),
            (1.5, 1.6), (1.6, 1.7), (1.7, 1.8), (1.8, 1.9), (1.9, 2.01)]
    for lo, hi in bins:
        count = np.sum((valid_prefs >= lo) & (valid_prefs < hi))
        pct = count / len(valid_prefs) * 100
        bar = "#" * int(pct)
        label = f"[{lo:.1f}, {hi:.1f})"
        print(f"    {label:12s}: {count:8,} ({pct:5.1f}%) {bar}")

    # Per-model statistics
    per_model_winrate = win_df.mean(axis=0)
    print(f"\n  Per-model win rate (binary):")
    best_model = per_model_winrate.idxmax()
    worst_model = per_model_winrate.idxmin()
    print(f"    Best:   {per_model_winrate.max()*100:.1f}% ({best_model})")
    print(f"    Worst:  {per_model_winrate.min()*100:.1f}% ({worst_model})")
    print(f"    Median: {per_model_winrate.median()*100:.1f}%")
    print(f"    Mean:   {per_model_winrate.mean()*100:.1f}%")
    print(f"    Std:    {per_model_winrate.std()*100:.1f}%")

    # Per-item statistics
    per_item_winrate = win_df.mean(axis=1)
    print(f"\n  Per-item win rate (across models):")
    print(f"    Min:    {per_item_winrate.min()*100:.1f}%")
    print(f"    Max:    {per_item_winrate.max()*100:.1f}%")
    print(f"    Median: {per_item_winrate.median()*100:.1f}%")
    print(f"    Std:    {per_item_winrate.std()*100:.1f}%")

    # Item difficulty distribution
    unsolved = (per_item_winrate == 0).sum()
    easy = (per_item_winrate > 0.9).sum()
    hard = (per_item_winrate < 0.1).sum()
    print(f"\n  Item difficulty distribution:")
    print(f"    No model wins (0%):   {unsolved}")
    print(f"    Hard (<10% win):      {hard}")
    print(f"    Easy (>90% win):      {easy}")

    # Top 15 models by win rate
    top_models = per_model_winrate.sort_values(ascending=False)
    print(f"\n  Top 15 models by binary win rate:")
    for i, (model, wr) in enumerate(top_models.head(15).items()):
        n_items_scored = win_df[model].notna().sum()
        print(f"    {i+1:3d}. {model:50s}  {wr*100:5.1f}%  ({n_items_scored} items)")

    print(f"\n  Bottom 10 models by binary win rate:")
    for i, (model, wr) in enumerate(top_models.tail(10).items()):
        n_items_scored = win_df[model].notna().sum()
        print(f"    {n_models-9+i:3d}. {model:50s}  {wr*100:5.1f}%  ({n_items_scored} items)")

    # Dataset breakdown
    # Collect dataset info from first available model
    first_model = successful_models[0]
    first_model_data = all_model_data[first_model]
    instr_datasets = {}
    for instr in instructions:
        if instr in first_model_data:
            instr_datasets[instr] = first_model_data[instr].get("dataset", "unknown")

    dataset_counts = {}
    for instr in instructions:
        ds = instr_datasets.get(instr, "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

    print(f"\n  Dataset breakdown (source of instructions):")
    for ds, count in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        # Compute mean win rate for items in this dataset
        ds_items = [i for i, instr in enumerate(instructions)
                    if instr_datasets.get(instr, "") == ds]
        ds_winrate = win_df.iloc[ds_items].mean().mean() * 100
        print(f"    {ds:20s}  n={count:4d}  mean_win_rate={ds_winrate:.1f}%")

    # ---- Save outputs ----

    # 1. Binary response matrix (transposed: rows=models, columns=items)
    win_df_t = win_df.T
    win_df_t.index.name = "Model"
    win_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    win_df_t.to_csv(win_path)
    print(f"\n  Saved binary win matrix: {win_path}")

    # 2. Raw preference matrix (transposed: rows=models, columns=items)
    pref_df_t = pref_df.T
    pref_df_t.index.name = "Model"
    pref_path = os.path.join(PROCESSED_DIR, "response_matrix_preference.csv")
    pref_df_t.to_csv(pref_path)
    print(f"  Saved preference matrix: {pref_path}")

    # 3. Item metadata
    item_rows = []
    for i, instr in enumerate(instructions):
        item_rows.append({
            "item_idx": i,
            "instruction": instr,
            "dataset": instr_datasets.get(instr, "unknown"),
            "n_models_scored": int(pref_df.iloc[i].notna().sum()),
            "mean_preference": float(pref_df.iloc[i].mean()),
            "mean_win_rate": float(win_df.iloc[i].mean()),
        })
    item_meta_df = pd.DataFrame(item_rows)
    item_meta_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    item_meta_df.to_csv(item_meta_path, index=False)
    print(f"  Saved item metadata: {item_meta_path}")

    return successful_models, per_model_winrate, pref_df, win_df


def build_model_summary(successful_models, per_model_winrate, pref_df):
    """Build model summary CSV with leaderboard data merged in."""
    print(f"\nSTEP 4: Building model summary")
    print("-" * 60)

    # Load leaderboard if available
    leaderboard_path = os.path.join(RAW_DIR, "leaderboard.csv")
    lb_data = {}
    if os.path.exists(leaderboard_path):
        lb_df = pd.read_csv(leaderboard_path, index_col=0)
        for model_name in lb_df.index:
            lb_data[model_name] = {
                "official_win_rate": lb_df.loc[model_name].get("win_rate", None),
                "lc_win_rate": lb_df.loc[model_name].get(
                    "length_controlled_winrate", None
                ),
                "avg_length": lb_df.loc[model_name].get("avg_length", None),
                "mode": lb_df.loc[model_name].get("mode", ""),
                "n_total": lb_df.loc[model_name].get("n_total", None),
            }

    rows = []
    for model in sorted(successful_models):
        row = {"model": model}

        # Our computed stats
        row["binary_win_rate"] = round(float(per_model_winrate[model]), 4)
        row["mean_preference"] = round(float(pref_df[model].mean()), 4)
        row["n_items_scored"] = int(pref_df[model].notna().sum())

        # Official leaderboard stats
        lb = lb_data.get(model, {})
        row["official_win_rate"] = lb.get("official_win_rate", None)
        row["lc_win_rate"] = lb.get("lc_win_rate", None)
        row["avg_length"] = lb.get("avg_length", None)
        row["mode"] = lb.get("mode", "")

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        "binary_win_rate", ascending=False, na_position="last"
    )

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"  Total models: {len(summary_df)}")
    n_with_lb = summary_df["official_win_rate"].notna().sum()
    print(f"  Models with official leaderboard data: {n_with_lb}")

    # Correlation between our binary win rate and official win rate
    if n_with_lb > 5:
        valid = summary_df.dropna(subset=["official_win_rate"])
        corr = valid["binary_win_rate"].corr(valid["official_win_rate"] / 100)
        print(f"  Correlation (our binary WR vs official WR): {corr:.4f}")

    # Mode distribution
    if "mode" in summary_df.columns:
        mode_counts = summary_df["mode"].value_counts()
        print(f"\n  Mode distribution:")
        for mode, count in mode_counts.items():
            if mode:
                print(f"    {mode:20s}  n={count}")

    print(f"\n  Saved: {output_path}")
    return summary_df


def _extract_item_content():
    """Extract item_content.csv: instruction text from item_metadata.csv."""
    meta_path = os.path.join(PROCESSED_DIR, "item_metadata.csv")
    if not os.path.exists(meta_path):
        print("  No item_metadata.csv found; skipping item_content extraction")
        return
    meta = pd.read_csv(meta_path)
    items = [
        {"item_id": str(row.get("item_idx", i)), "content": str(row["instruction"])[:2000]}
        for i, (_, row) in enumerate(meta.iterrows())
        if pd.notna(row.get("instruction"))
    ]
    out_path = os.path.join(PROCESSED_DIR, "item_content.csv")
    pd.DataFrame(items).to_csv(out_path, index=False)
    print(f"  Extracted {len(items)} items to {out_path}")


def main():
    print("AlpacaEval 2.0 Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print(f"  Annotator:          {ANNOTATOR_DIR}")
    print(f"  Reference model:    gpt4_1106_preview")
    print(f"  Judge:              GPT-4 Turbo (weighted)")
    print(f"  Total models:       {len(ALL_MODELS)}")
    print()

    # Step 1: Download annotations
    failed_models = download_annotations()

    # Step 2: Download leaderboard
    download_leaderboard()

    # Step 3: Build response matrices
    successful_models, per_model_winrate, pref_df, win_df = build_response_matrices()

    # Step 4: Build model summary
    build_model_summary(successful_models, per_model_winrate, pref_df)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  PRIMARY response matrix (binary win/loss):")
    print(f"    Dimensions: {len(successful_models)} models x {len(pref_df)} items")
    n_valid = win_df.notna().sum().sum()
    total = len(successful_models) * len(pref_df)
    print(f"    Fill rate:  {n_valid/total*100:.1f}%")
    print(f"    Score type: Binary (1=model wins vs gpt4_1106_preview, 0=loss)")
    print(f"    Threshold:  preference > 1.5 => win")
    print(f"    Evaluator:  GPT-4 Turbo (weighted AlpacaEval 2.0)")

    print(f"\n  All output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")

    print(f"\n  Score interpretation:")
    print(f"    Binary matrix: 1 = model output preferred over reference")
    print(f"                   0 = reference output preferred")
    print(f"    Preference matrix: float in [1, 2]")
    print(f"      1.0 = strong reference preference")
    print(f"      1.5 = tie")
    print(f"      2.0 = strong model preference")

    # Step 5: Extract item content
    print("\nSTEP 5: Extracting item content")
    print("-" * 60)
    _extract_item_content()


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
