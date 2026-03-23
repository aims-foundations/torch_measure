"""
Build MME response matrix from VLMEval/OpenVLMRecords.

Data source:
  - HuggingFace dataset: VLMEval/OpenVLMRecords
  - Per-model xlsx files at mmeval/{model}/{model}_MME.xlsx
  - Binary yes/no vision perception + cognition benchmark

Score format:
  - Binary 0/1: whether extracted yes/no matches ground truth
  - NaN if answer cannot be parsed from model prediction

Outputs:
  - raw/{model}_MME.xlsx: Downloaded xlsx files
  - processed/response_matrix.csv: Models (rows) x items (columns)
  - processed/item_content.csv: Per-item metadata
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

REPO_ID = "VLMEval/OpenVLMRecords"
BENCHMARK_SUFFIX = "MME"


def list_all_models():
    """List all model directories in the repo."""
    items = list(list_repo_tree(REPO_ID, path_in_repo="mmeval", repo_type="dataset"))
    models = []
    for item in items:
        name = item.path.replace("mmeval/", "")
        if "/" not in name:
            models.append(name)
    return sorted(models)


def find_models_for_benchmark(all_models):
    """Find which models have results for this benchmark."""
    models_with_bench = []
    for model in all_models:
        try:
            items = list(list_repo_tree(REPO_ID, path_in_repo=f"mmeval/{model}", repo_type="dataset"))
            filenames = [item.path.split("/")[-1] for item in items]
            target = f"{model}_{BENCHMARK_SUFFIX}.xlsx"
            if target in filenames:
                models_with_bench.append(model)
        except Exception:
            pass
    return models_with_bench


def extract_yesno(prediction_text):
    """Extract yes/no from a model's prediction."""
    if pd.isna(prediction_text):
        return None
    pred = str(prediction_text).strip().lower()

    if pred in ("yes", "no"):
        return pred.capitalize()

    if pred.startswith("yes"):
        return "Yes"
    if pred.startswith("no"):
        return "No"

    if re.search(r"\byes\b", pred, re.IGNORECASE):
        if not re.search(r"\bno\b", pred, re.IGNORECASE):
            return "Yes"
        yes_pos = re.search(r"\byes\b", pred, re.IGNORECASE).start()
        no_pos = re.search(r"\bno\b", pred, re.IGNORECASE).start()
        return "No" if no_pos > yes_pos else "Yes"
    if re.search(r"\bno\b", pred, re.IGNORECASE):
        return "No"

    return None


def score_yesno(row):
    """Score a yes/no question."""
    pred = extract_yesno(str(row["prediction"]))
    gold = str(row["answer"]).strip().capitalize()
    if pred is None:
        return np.nan
    return 1 if pred == gold else 0


def main():
    print(f"MME Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    print("Listing all models in VLMEval/OpenVLMRecords...")
    all_models = list_all_models()
    print(f"Found {len(all_models)} models total")

    print(f"Scanning for {BENCHMARK_SUFFIX}...")
    models_with_bench = find_models_for_benchmark(all_models)
    print(f"Found {len(models_with_bench)} models with {BENCHMARK_SUFFIX}")

    if not models_with_bench:
        print("No models found, exiting.")
        sys.exit(1)

    all_scores = {}
    item_info = None
    failed_models = []

    for i, model in enumerate(models_with_bench):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Downloading {i + 1}/{len(models_with_bench)}: {model}")

        try:
            fpath = f"mmeval/{model}/{model}_{BENCHMARK_SUFFIX}.xlsx"
            local_path = hf_hub_download(REPO_ID, fpath, repo_type="dataset")

            raw_dest = os.path.join(RAW_DIR, f"{model}_{BENCHMARK_SUFFIX}.xlsx")
            if not os.path.exists(raw_dest):
                import shutil
                shutil.copy2(local_path, raw_dest)

            df = pd.read_excel(local_path)

            if len(df) == 0:
                failed_models.append((model, "empty dataframe"))
                continue

            if "index" not in df.columns:
                df["index"] = range(len(df))

            df["score"] = df.apply(score_yesno, axis=1)
            scores = df.set_index("index")["score"]
            all_scores[model] = scores

            if item_info is None:
                item_cols = ["index", "question"]
                for c in ["category", "answer"]:
                    if c in df.columns:
                        item_cols.append(c)
                item_info = df[item_cols].copy()

        except Exception as e:
            failed_models.append((model, str(e)[:100]))

    if not all_scores:
        print("No successful downloads")
        sys.exit(1)

    response_matrix = pd.DataFrame(all_scores).sort_index().T
    response_matrix.index.name = "model"

    n_models = len(response_matrix)
    n_items = len(response_matrix.columns)
    model_acc = response_matrix.mean(axis=1)
    item_acc = response_matrix.mean(axis=0)
    missing_rate = response_matrix.isna().mean().mean()

    print(f"\n{'=' * 60}")
    print(f"  Models: {n_models}")
    print(f"  Items: {n_items}")
    print(f"  Missing rate: {missing_rate:.3f}")
    print(f"  Model accuracy range: {model_acc.min():.3f} - {model_acc.max():.3f}")
    print(f"  Item accuracy range: {item_acc.min():.3f} - {item_acc.max():.3f}")

    response_matrix.to_csv(os.path.join(PROCESSED_DIR, "response_matrix.csv"))
    print(f"  Saved response_matrix.csv")

    if item_info is not None:
        item_info = item_info.drop_duplicates(subset=["index"]).sort_values("index")
        item_info.to_csv(os.path.join(PROCESSED_DIR, "item_content.csv"), index=False)
        print(f"  Saved item_content.csv")

    if failed_models:
        print(f"\n  Failed models ({len(failed_models)}):")
        for m, e in failed_models[:10]:
            print(f"    {m}: {e}")


if __name__ == "__main__":
    main()
