"""
Build HLE (Humanity's Last Exam) response matrix from publicly available per-item data.

Data sources:
  1. supaihq/hle (GitHub) — judged_hle_pro.json: Per-question judged results for up to 19 models
     on 1,369 questions from HLE. Models include Sup AI (ensemble), GPT-5 Pro, GPT-5.1,
     Claude Opus 4.5, Claude Sonnet 4.5, Gemini 3 Pro Preview, Gemini 2.5 Pro,
     Grok-4, DeepSeek v3.2, GLM-4.6, Kimi K2, Qwen3, Mistral, MiniMax, etc.
     Source: https://github.com/supaihq/hle

  2. deepwriter-ai/hle-gemini-3-0 (GitHub) — questions_and_answer_hle_gem3pro.csv:
     Per-question results for Google Gemini 3 Pro on 878 text-only HLE questions.
     Source: https://github.com/deepwriter-ai/hle-gemini-3-0

  3. Scale AI SEAL Leaderboard — Aggregate scores only (no per-item data available).
     Used for reference/validation. 44 models with aggregate accuracy + CI.
     Source: https://scale.com/leaderboard/humanitys_last_exam

Notes:
  - The HLE benchmark has 2,500 total questions. Our per-item data covers a subset:
    1,369 questions from supaihq + 423 additional from deepwriter = ~1,792 unique questions.
  - Not all models are evaluated on all questions (sparse matrix).
  - We filter to models with at least 50 evaluated items for meaningful coverage.
  - Values: 1 = correct, 0 = incorrect, NaN = not evaluated.
  - The "Sup AI" model in supaihq data is an ensemble/agentic system, not a single LLM.

Outputs:
  - response_matrix.csv: Binary (models x items) matrix with NaN for unevaluated
  - response_matrix_dense.csv: Same but filtered to models/items with good coverage
  - model_summary.csv: Per-model aggregate statistics
"""

import os
import json
import csv
import numpy as np
import pandas as pd

# Paths
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_supaihq_data(json_path):
    """Load per-question judged results from supaihq/hle repository.

    Returns dict: {question_id: {model_name: 0_or_1}}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    for qid, qdata in data.items():
        judge_response = qdata.get("judge_response", {})
        results[qid] = {}
        for model_name, judgment in judge_response.items():
            if isinstance(judgment, dict) and "correct" in judgment:
                correct = 1 if judgment["correct"].lower() == "yes" else 0
                results[qid][model_name] = correct
    return results


def load_deepwriter_data(csv_path):
    """Load per-question results for Gemini 3 Pro from deepwriter-ai repo.

    Returns dict: {question_id: 0_or_1}
    """
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"].strip()
            if qid == "Totals:" or not qid:
                continue
            try:
                score = int(float(row["score"]))
                results[qid] = score
            except (ValueError, KeyError):
                continue
    return results


def build_response_matrix():
    """Build the full response matrix from all data sources."""
    print("=" * 70)
    print("  HLE Response Matrix Builder")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load supaihq judged data
    # -------------------------------------------------------------------------
    supaihq_path = os.path.join(RAW_DIR, "judged_hle_pro.json")
    print(f"\n[1] Loading supaihq/hle judged data from: {supaihq_path}")
    supaihq_data = load_supaihq_data(supaihq_path)
    print(f"    Questions: {len(supaihq_data)}")

    # Collect all models and their counts
    model_question_counts = {}
    for qid, model_results in supaihq_data.items():
        for model_name in model_results:
            model_question_counts[model_name] = model_question_counts.get(model_name, 0) + 1

    print(f"    Models found: {len(model_question_counts)}")
    for model, count in sorted(model_question_counts.items(), key=lambda x: -x[1]):
        print(f"      {model:50s} {count:5d} questions")

    # -------------------------------------------------------------------------
    # 2. Load deepwriter Gemini 3 Pro data
    # -------------------------------------------------------------------------
    deepwriter_path = os.path.join(RAW_DIR, "questions_and_answer_hle_gem3pro.csv")
    print(f"\n[2] Loading deepwriter-ai/hle-gemini-3-0 data from: {deepwriter_path}")
    deepwriter_data = load_deepwriter_data(deepwriter_path)
    print(f"    Questions: {len(deepwriter_data)}")

    # Check overlap with supaihq
    supaihq_qids = set(supaihq_data.keys())
    deepwriter_qids = set(deepwriter_data.keys())
    overlap = supaihq_qids & deepwriter_qids
    deepwriter_only = deepwriter_qids - supaihq_qids
    print(f"    Overlap with supaihq: {len(overlap)} questions")
    print(f"    Deepwriter-only questions: {len(deepwriter_only)}")

    # -------------------------------------------------------------------------
    # 3. Merge data into a unified structure
    # -------------------------------------------------------------------------
    print(f"\n[3] Merging data sources...")

    # Rename models for clarity
    MODEL_RENAMES = {
        "main": "Sup-AI-Ensemble",
        "alibaba/qwen3-max": "Qwen3-Max",
        "alibaba/qwen3-next-80b-a3b-thinking": "Qwen3-Next-80B-A3B-Thinking",
        "alibaba/qwen3-vl-thinking": "Qwen3-VL-Thinking",
        "anthropic/claude-opus-4.5": "Claude-Opus-4.5",
        "anthropic/claude-sonnet-4.5": "Claude-Sonnet-4.5",
        "deepseek/deepseek-v3.2-exp-thinking": "DeepSeek-V3.2-Exp-Thinking",
        "deepseek/deepseek-v3.2-thinking": "DeepSeek-V3.2-Thinking",
        "google/gemini-2.5-flash": "Gemini-2.5-Flash",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro",
        "google/gemini-3-pro-preview": "Gemini-3-Pro-Preview",
        "minimax/minimax-m2": "MiniMax-M2",
        "mistral/magistral-medium": "Mistral-Magistral-Medium",
        "mistral/mistral-large": "Mistral-Large",
        "moonshotai/kimi-k2-thinking-turbo": "Kimi-K2-Thinking-Turbo",
        "openai/gpt-5-pro": "GPT-5-Pro",
        "openai/gpt-5.1": "GPT-5.1",
        "xai/grok-4": "Grok-4",
        "zai/glm-4.6": "GLM-4.6",
    }

    # Build unified dict: {qid: {renamed_model: 0/1}}
    unified = {}

    # Add supaihq data
    for qid, model_results in supaihq_data.items():
        if qid not in unified:
            unified[qid] = {}
        for model_name, score in model_results.items():
            renamed = MODEL_RENAMES.get(model_name, model_name)
            unified[qid][renamed] = score

    # Add deepwriter data (Gemini 3 Pro) for questions not in supaihq
    # or merge if the model is already present
    deepwriter_model = "Gemini-3-Pro-Preview"
    added_from_deepwriter = 0
    for qid, score in deepwriter_data.items():
        if qid not in unified:
            unified[qid] = {}
        if deepwriter_model not in unified[qid]:
            unified[qid][deepwriter_model] = score
            added_from_deepwriter += 1

    print(f"    Additional Gemini-3-Pro-Preview items from deepwriter: {added_from_deepwriter}")

    # Collect all unique question IDs and model names
    all_qids = sorted(unified.keys())
    all_models = set()
    for qid_results in unified.values():
        all_models.update(qid_results.keys())
    all_models = sorted(all_models)

    print(f"    Total unique questions: {len(all_qids)}")
    print(f"    Total unique models: {len(all_models)}")

    # -------------------------------------------------------------------------
    # 4. Build the full response matrix (models x items)
    # -------------------------------------------------------------------------
    print(f"\n[4] Building response matrix...")

    # Create DataFrame with models as rows and question IDs as columns
    matrix_data = {}
    for model in all_models:
        row = {}
        for qid in all_qids:
            val = unified.get(qid, {}).get(model, np.nan)
            row[qid] = val
        matrix_data[model] = row

    df_full = pd.DataFrame(matrix_data).T
    df_full.index.name = "Model"

    # Print full matrix stats
    total_cells = df_full.shape[0] * df_full.shape[1]
    filled_cells = df_full.notna().sum().sum()
    fill_rate = filled_cells / total_cells
    correct_cells = (df_full == 1).sum().sum()
    incorrect_cells = (df_full == 0).sum().sum()

    print(f"    Full matrix dimensions: {df_full.shape[0]} models x {df_full.shape[1]} items")
    print(f"    Total cells: {total_cells:,}")
    print(f"    Filled cells: {int(filled_cells):,} ({fill_rate*100:.1f}%)")
    print(f"    Correct (1): {int(correct_cells):,} ({correct_cells/filled_cells*100:.1f}% of filled)")
    print(f"    Incorrect (0): {int(incorrect_cells):,} ({incorrect_cells/filled_cells*100:.1f}% of filled)")

    # Per-model stats
    print(f"\n    Per-model statistics:")
    for model in all_models:
        row = df_full.loc[model]
        n_eval = row.notna().sum()
        n_correct = (row == 1).sum()
        accuracy = n_correct / n_eval * 100 if n_eval > 0 else 0
        print(f"      {model:45s}  evaluated={int(n_eval):5d}  correct={int(n_correct):4d}  accuracy={accuracy:.1f}%")

    # Save full matrix
    full_output = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    df_full.to_csv(full_output)
    print(f"\n    Saved full matrix: {full_output}")

    # -------------------------------------------------------------------------
    # 5. Build dense matrix (filter to well-covered models and items)
    # -------------------------------------------------------------------------
    print(f"\n[5] Building dense response matrix (models with >= 50 items)...")

    # Filter models with at least 50 evaluated items
    MIN_ITEMS = 50
    model_eval_counts = df_full.notna().sum(axis=1)
    dense_models = model_eval_counts[model_eval_counts >= MIN_ITEMS].index.tolist()

    df_dense = df_full.loc[dense_models]

    # Also filter items: keep items that have at least 2 model evaluations
    item_eval_counts = df_dense.notna().sum(axis=0)
    dense_items = item_eval_counts[item_eval_counts >= 2].index.tolist()
    df_dense = df_dense[dense_items]

    dense_total = df_dense.shape[0] * df_dense.shape[1]
    dense_filled = df_dense.notna().sum().sum()
    dense_fill_rate = dense_filled / dense_total if dense_total > 0 else 0
    dense_correct = (df_dense == 1).sum().sum()

    print(f"    Dense matrix dimensions: {df_dense.shape[0]} models x {df_dense.shape[1]} items")
    print(f"    Total cells: {dense_total:,}")
    print(f"    Filled cells: {int(dense_filled):,} ({dense_fill_rate*100:.1f}%)")
    print(f"    Correct (1): {int(dense_correct):,} ({dense_correct/dense_filled*100:.1f}% of filled)")

    dense_output = os.path.join(PROCESSED_DIR, "response_matrix_dense.csv")
    df_dense.to_csv(dense_output)
    print(f"    Saved dense matrix: {dense_output}")

    # -------------------------------------------------------------------------
    # 6. Build model summary
    # -------------------------------------------------------------------------
    print(f"\n[6] Building model summary...")

    # Load SEAL leaderboard for reference
    seal_path = os.path.join(RAW_DIR, "seal_leaderboard_scores.csv")
    seal_df = pd.read_csv(seal_path)

    summary_rows = []
    for model in all_models:
        row_data = df_full.loc[model]
        n_eval = int(row_data.notna().sum())
        n_correct = int((row_data == 1).sum())
        n_incorrect = int((row_data == 0).sum())
        accuracy = n_correct / n_eval * 100 if n_eval > 0 else 0.0

        summary_rows.append({
            "model": model,
            "n_items_evaluated": n_eval,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy_pct": round(accuracy, 2),
            "source": "supaihq+deepwriter",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("accuracy_pct", ascending=False)

    summary_output = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(summary_output, index=False)
    print(f"    Saved model summary: {summary_output}")

    # -------------------------------------------------------------------------
    # 7. Final summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Data sources:")
    print(f"    1. supaihq/hle (GitHub): 19 models, 1,369 questions (per-item)")
    print(f"    2. deepwriter-ai/hle-gemini-3-0 (GitHub): 1 model, 878 questions (per-item)")
    print(f"    3. Scale AI SEAL leaderboard: 44 models, aggregate scores only")
    print(f"\n  Response matrix (full):")
    print(f"    Dimensions: {df_full.shape[0]} models x {df_full.shape[1]} items")
    print(f"    Fill rate: {fill_rate*100:.1f}%")
    print(f"    Value distribution:")
    print(f"      1 (correct):     {int(correct_cells):,} ({correct_cells/filled_cells*100:.1f}%)")
    print(f"      0 (incorrect):   {int(incorrect_cells):,} ({incorrect_cells/filled_cells*100:.1f}%)")
    print(f"      NaN (missing):   {int(total_cells - filled_cells):,} ({(1-fill_rate)*100:.1f}%)")
    print(f"\n  Response matrix (dense, models with >= {MIN_ITEMS} items):")
    print(f"    Dimensions: {df_dense.shape[0]} models x {df_dense.shape[1]} items")
    print(f"    Fill rate: {dense_fill_rate*100:.1f}%")
    print(f"\n  Note: HLE has 2,500 total questions. Per-item data covers")
    print(f"  {len(all_qids)} unique questions ({len(all_qids)/2500*100:.1f}% of full benchmark).")
    print(f"  The remaining questions have only aggregate scores available")
    print(f"  from the Scale AI SEAL leaderboard (saved in raw/ for reference).")

    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    build_response_matrix()
