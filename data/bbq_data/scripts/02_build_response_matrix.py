"""
Build response matrix from BBQ (Bias Benchmark for QA).

Source: https://github.com/nyu-mll/BBQ
Paper: Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question Answering", ACL 2022

Structure: models x items -> binary correct
Models available:
  - UnifiedQA-T5-11B (3 prompting variants: race, qonly, arc)
  - RoBERTa-Base, RoBERTa-Large (race variant)
  - DeBERTaV3-Base, DeBERTaV3-Large (race variant)
Items: 58,492 questions across 11 bias categories

Output:
  - response_matrix.csv: model x item -> binary correct
  - item_metadata.csv: per-item category, polarity, context condition
"""

import json
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw" / "repo"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_gender",
    "Race_x_SES", "Religion", "SES", "Sexual_orientation",
]


def load_unifiedqa_results() -> dict[str, pd.Series]:
    """Load UnifiedQA predictions and compute binary correctness."""
    models = {}
    pred_cols = [
        ("unifiedqa-t5-11b_pred_race", "UnifiedQA-T5-11B-race"),
        ("unifiedqa-t5-11b_pred_qonly", "UnifiedQA-T5-11B-qonly"),
        ("unifiedqa-t5-11b_pred_arc", "UnifiedQA-T5-11B-arc"),
    ]

    all_rows = []
    for cat in CATEGORIES:
        path = RAW_DIR / "results" / "UnifiedQA" / f"preds_{cat}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                all_rows.append(row)

    if not all_rows:
        return {}

    df = pd.DataFrame(all_rows)
    # Create unique IDs using category + example_id
    unique_ids = [f"{cat}_{eid}" for cat, eid in zip(df["category"].values, df["example_id"].values)]

    for pred_col, model_name in pred_cols:
        if pred_col not in df.columns:
            continue
        # Compare prediction text to the correct answer text
        correct_answer_text = df.apply(lambda r: r[f"ans{r['label']}"].lower().strip(), axis=1)
        pred_text = df[pred_col].str.lower().str.strip()
        correct = (pred_text == correct_answer_text).astype(int).values
        models[model_name] = pd.Series(correct, index=unique_ids)

    return models


def load_roberta_results() -> dict[str, pd.Series]:
    """Load RoBERTa/DeBERTaV3 logits and compute binary correctness."""
    csv_path = RAW_DIR / "results" / "RoBERTa_and_DeBERTaV3" / "df_bbq.csv"
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)

    # Load ground truth labels from data files
    gt = {}
    for cat in CATEGORIES:
        data_path = RAW_DIR / "data" / f"{cat}.jsonl"
        if not data_path.exists():
            continue
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                gt[row["example_id"]] = row["label"]

    models = {}
    model_name_map = {
        "roberta-base-race": "RoBERTa-Base",
        "roberta-large-race": "RoBERTa-Large",
        "deberta-v3-base-race": "DeBERTaV3-Base",
        "deberta-v3-large-race": "DeBERTaV3-Large",
    }

    for model_id, model_name in model_name_map.items():
        mdf = df[df["model"] == model_id].copy()
        if len(mdf) == 0:
            continue
        # Predicted answer = argmax of logits
        logits = mdf[["ans0", "ans1", "ans2"]].values
        predicted = logits.argmax(axis=1)
        # Compare to ground truth — use (cat, index) as unique ID
        labels = mdf["index"].map(gt).values
        correct = (predicted == labels).astype(int)
        unique_ids = [f"{cat}_{idx}" for cat, idx in zip(mdf["cat"].values, mdf["index"].values)]
        models[model_name] = pd.Series(correct, index=unique_ids)

    return models


def main():
    print("Loading BBQ results...")

    uqa_models = load_unifiedqa_results()
    print(f"UnifiedQA models: {list(uqa_models.keys())}")

    roberta_models = load_roberta_results()
    print(f"RoBERTa/DeBERTa models: {list(roberta_models.keys())}")

    all_models = {**uqa_models, **roberta_models}
    if not all_models:
        print("No model results found!")
        return

    # Build response matrix
    matrix = pd.DataFrame(all_models).T
    matrix.index.name = "model"
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix.to_csv(OUTPUT_DIR / "response_matrix.csv")
    print(f"\nResponse matrix: {matrix.shape[0]} models x {matrix.shape[1]} items")
    print(f"Per-model accuracy:")
    for model in matrix.index:
        acc = matrix.loc[model].mean()
        n_items = matrix.loc[model].notna().sum()
        print(f"  {model}: {acc:.3f} ({n_items} items)")

    # Build item metadata
    all_items = []
    for cat in CATEGORIES:
        data_path = RAW_DIR / "data" / f"{cat}.jsonl"
        if not data_path.exists():
            continue
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                all_items.append({
                    "example_id": row["example_id"],
                    "category": row["category"],
                    "question_polarity": row["question_polarity"],
                    "context_condition": row["context_condition"],
                    "label": row["label"],
                    "question": row["question"],
                    "context": row["context"],
                })

    if all_items:
        item_df = pd.DataFrame(all_items)
        item_df.to_csv(OUTPUT_DIR / "item_metadata.csv", index=False)
        print(f"\nItem metadata: {len(item_df)} items across {item_df['category'].nunique()} categories")
        print(f"Category distribution:\n{item_df['category'].value_counts()}")


if __name__ == "__main__":
    main()
