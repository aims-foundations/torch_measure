"""
Build response matrix from JailbreakBench artifacts.

Source: https://github.com/JailbreakBench/artifacts
Paper: Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs", 2024

Structure: models x behaviors -> binary jailbroken (per attack method)
Models: vicuna-13b-v1.5, llama-2-7b-chat-hf, gpt-3.5-turbo-1106, gpt-4-0125-preview
Behaviors: 100 harmful behaviors across 10 categories
Attack methods: PAIR, GCG, DSN, JBC, prompt_with_random_search

Output:
  - response_matrix_{method}.csv: model x behavior -> binary jailbroken (1=unsafe, 0=safe)
  - response_matrix_all.csv: (model_method) x behavior -> binary jailbroken (all methods combined)
  - item_metadata.csv: per-behavior goal, category
"""

import json
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw" / "artifacts" / "attack-artifacts"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_attack_results(method_dir: Path) -> list[dict]:
    """Load all model results for one attack method."""
    results = []
    # Check all subdirectories (black_box, white_box, transfer, manual)
    for subdir in method_dir.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        if subdir.name in ("attack-info.json",):
            continue
        for json_file in sorted(subdir.glob("*.json")):
            if json_file.name == "attack-info.json":
                continue
            model_name = json_file.stem
            with open(json_file) as f:
                data = json.load(f)
            jailbreaks = data.get("jailbreaks", [])
            for j in jailbreaks:
                results.append({
                    "model": model_name,
                    "behavior_index": j["index"],
                    "goal": j["goal"],
                    "behavior": j.get("behavior", ""),
                    "category": j.get("category", ""),
                    "jailbroken": int(j.get("jailbroken", False)),
                })
    return results


def main():
    print("Loading JailbreakBench artifacts...")

    methods = [d.name for d in sorted(RAW_DIR.iterdir())
               if d.is_dir() and d.name != "test-artifact"]

    all_method_matrices = {}
    all_rows = []

    for method in methods:
        method_dir = RAW_DIR / method
        results = load_attack_results(method_dir)
        if not results:
            print(f"  {method}: no results found")
            continue

        df = pd.DataFrame(results)
        matrix = df.pivot_table(
            index="model", columns="behavior_index", values="jailbroken", aggfunc="first"
        )
        matrix.to_csv(OUTPUT_DIR / f"response_matrix_{method.lower()}.csv")
        all_method_matrices[method] = matrix

        asr = matrix.mean(axis=1)
        print(f"\n  {method}: {matrix.shape[0]} models x {matrix.shape[1]} behaviors")
        for model in matrix.index:
            print(f"    {model}: ASR={matrix.loc[model].mean():.3f}")

        # Add to combined
        for _, row in df.iterrows():
            row_dict = dict(row)
            row_dict["model"] = f"{row['model']}_{method}"
            all_rows.append(row_dict)

    # Combined matrix: (model_method) x behavior
    if all_rows:
        combined = pd.DataFrame(all_rows)
        combined_matrix = combined.pivot_table(
            index="model", columns="behavior_index", values="jailbroken", aggfunc="first"
        )
        combined_matrix.to_csv(OUTPUT_DIR / "response_matrix_all.csv")
        print(f"\nCombined: {combined_matrix.shape[0]} (model x method) x {combined_matrix.shape[1]} behaviors")

    # Item metadata from first available method
    if all_rows:
        items = pd.DataFrame(all_rows).drop_duplicates(subset=["behavior_index"])[
            ["behavior_index", "goal", "behavior", "category"]
        ].sort_values("behavior_index")
        items.to_csv(OUTPUT_DIR / "item_metadata.csv", index=False)
        print(f"\nItem metadata: {len(items)} behaviors across {items['category'].nunique()} categories")
        print(f"Category distribution:\n{items['category'].value_counts()}")


if __name__ == "__main__":
    main()
