#!/usr/bin/env python3
"""
Build task metadata, item content, and response matrix for FINANCE benchmarks.

This script consolidates data from multiple financial NLP/LLM benchmarks,
focusing on non-English sources. It builds:
  - task_metadata.csv  : item_id, question (truncated), answer_key,
                         subject/config, split, source_dataset, language, task_type
  - item_content.csv   : item_id, full question + options text
  - model_summary.csv  : placeholder (no per-item model predictions found publicly)
  - response_matrix.csv: gold answers only (item_id, gold_answer)

─────────────────────────────────────────────────────────────────────────
DATA SOURCES (per-item benchmark data available)
─────────────────────────────────────────────────────────────────────────

1. FinEval (Chinese) — SUFE-AIFLM-Lab/FinEval
   URL: https://github.com/SUFE-AIFLM-Lab/FinEval
   HuggingFace: https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval
   Items: ~4,661 questions across 34 financial subjects (val set has answers)
   Format: CSV with id, question, A, B, C, D, answer
   Language: Chinese
   Status: Dataset publicly available; NO published per-item model predictions.
     - Test set (3,374 items): No answers — requires server submission
     - Val set (1,179 items): Has answers — evaluable locally
     - Dev set (347 items): Has answers — few-shot examples
   Searched for predictions:
     - GitHub repo: No results/ or predictions/ directories, no branches/tags/releases
     - OpenCompass compass_academic_predictions: Gated; does NOT include FinEval
     - Open LLM Leaderboard: Does not include FinEval
     - Open FinLLM Leaderboard: Aggregate scores only, no per-item
     - FinEval leaderboard: Aggregate per-subject accuracy only

2. CFLUE (Chinese) — aliyun/cflue
   URL: https://github.com/aliyun/cflue
   Items: 3,864 knowledge MC + 125 application NLP tasks = 3,989 total
   Format: JSON with question, choices, answer, analysis
   Language: Chinese
   Status: Dataset publicly available; NO published per-item model predictions.
     - Knowledge: Multiple-choice from 15 Chinese financial qualification exams
     - Application: Text classification, translation, RE, RC, text generation
   Searched for predictions:
     - GitHub repo: Only dataset + eval code, no results dir
     - submission_example.json: Shows per-item format (question + model_response)
       but only one illustrative example, not actual eval results
     - No leaderboard with downloadable per-item data

─────────────────────────────────────────────────────────────────────────
DATA SOURCES (aggregate scores only — no per-item predictions)
─────────────────────────────────────────────────────────────────────────

3. FinBen — The-FinAI (NeurIPS 2024)
   URL: https://github.com/The-FinAI/FinBen
   Leaderboard: https://huggingface.co/datasets/TheFinAI/results
   Items: 42 datasets, 24 tasks, English/Chinese/Spanish
   Status: AGGREGATE per-task scores only (F1, Acc, RMSE per task).
     - Old-FinBen-Leaderboard CSV files: model × task metric (not per-item)
     - TheFinAI/results on HuggingFace: JSON files with per-task aggregate metrics
     - 36 model result files covering ChatGPT, GPT-4, Gemini, Llama, Qwen, etc.
     - Raw data: finben_english.csv (14 models × 52 tasks), finben_chinese.csv
       (8 models × 48 tasks), finben_spanish.csv (10 models × 13 tasks)

4. PIXIU / FLARE — chancefocus/PIXIU, The-FinAI/PIXIU
   URL: https://github.com/chancefocus/PIXIU
   Status: AGGREGATE only. Evaluation framework + datasets, but no published
   per-item prediction files. FLARE covers 6 NLP tasks + 2 prediction tasks.

5. ICE-PIXIU / ICE-FLARE — YY0649/ICE-PIXIU
   URL: https://github.com/YY0649/ICE-PIXIU
   Status: Cross-lingual (Chinese-English) benchmark with 40 datasets.
   No per-item model prediction data published.

6. FLARE-ES / "Toison de Oro" — Spanish financial NLP
   Referenced in: arxiv.org/abs/2402.07405
   Status: 21 datasets for Spanish financial NLP. The FLARE-ES benchmark
   evaluates bilingual performance but no standalone GitHub repo with
   per-item results was found. Part of FinBen spanish results.

7. BBT-Fin / BBT-CFLEB — ssymmetry/BBT-FinCUGE-Applications
   URL: https://github.com/ssymmetry/BBT-FinCUGE-Applications
   Status: Chinese financial pre-training + 6-task benchmark.
   Results hosted on external website (bbt.ssymmetry.com). No per-item
   prediction data in the GitHub repository.

8. CFBenchmark — TongjiFinLab/CFBenchmark
   URL: https://github.com/TongjiFinLab/CFBenchmark
   Status: 3,917 items, Chinese financial assistant evaluation.
   Framework stores model responses + scores locally when run, but
   no pre-computed per-item results are published.

9. SuperCLUE-Fin — CLUEbenchmark/SuperCLUE-Fin
   URL: https://github.com/CLUEbenchmark/SuperCLUE-Fin
   Status: Repository contains only a README. No data, no results.
   25 tasks, 6 categories, grade-based evaluation. Results mentioned
   in paper but not downloadable.

10. AraFinNLP / ArBanking77 — SinaLab/ArBanking77
    URL: https://github.com/SinaLab/ArBanking77
    Items: 31,038 Arabic banking intent queries (MSA + 4 dialects)
    Status: DATASET available (train/val/test CSV), but NO per-item model
    predictions published. Shared task had 11 participating teams but
    their per-item submissions are not public.

11. IndicFinNLP — Hindi/Bengali/Telugu financial NLP
    Referenced in: LREC-COLING 2024 (Ghosh et al.)
    Status: No public GitHub repository found. 9 datasets, 3 tasks,
    3 Indian languages. The datasets are reportedly CC BY-NC-SA 4.0
    but no direct download link or per-item results found.

12. FinanceBench — financebench/results on HuggingFace
    URL: https://huggingface.co/datasets/financebench/results
    Status: Only 8 rows of AGGREGATE model scores (4 models × 2 splits).
    The PatronusAI/financebench dataset has 150 annotated QA pairs but
    no per-item model prediction data.

13. Open FinLLM Leaderboard — finosfoundation
    URL: https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard
    Status: Interactive leaderboard with per-task aggregate scores.
    No per-item prediction data downloadable.

14. OpenCompass compass_academic_predictions
    URL: https://huggingface.co/datasets/opencompass/compass_academic_predictions
    Status: GATED dataset requiring access request. Contains results_stations/
    directory with model predictions. May include FinEval but cannot be confirmed
    without access. 1-10M records, extract_predictions.py available.

15. FinEval-KR (Korean extension)
    URL: https://github.com/SUFE-AIFLM-Lab/FinEval-KR
    Status: 625 sample items (from 9,782 total) in JSONL format.
    Dataset only — no per-item model evaluation results.

─────────────────────────────────────────────────────────────────────────
SUMMARY OF PER-ITEM DATA AVAILABILITY
─────────────────────────────────────────────────────────────────────────

*** NO public per-item model × item response matrices were found for
    ANY financial benchmark as of March 2026. ***

All benchmarks publish only aggregate scores (per-task or per-subject).
The closest sources to per-item data are:

  a) OpenCompass compass_academic_predictions (gated, may have FinEval)
  b) FinEval's eval framework saves per-item CSV when do_save_csv=True,
     but no one has published these files
  c) CFLUE's submission_example.json shows per-item format but is not
     a real evaluation result
  d) FinBen's Old-FinBen-Leaderboard has per-task metrics for ~36 models
     across English/Chinese/Spanish financial tasks

This script builds task metadata and gold answers from the two benchmark
datasets that have downloadable per-item test data: FinEval and CFLUE.

All paths use Path(__file__).resolve().parent.parent
"""

import json
import sys
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1. FinEval (Chinese financial domain, 34 subjects)
# ──────────────────────────────────────────────────────────────────────
def load_fineval():
    """Load FinEval from raw zip file."""
    zip_path = RAW_DIR / "fineval.zip"
    if not zip_path.exists():
        print(f"  WARNING: {zip_path} not found. Skipping FinEval.")
        return []

    all_rows = []
    item_id = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]

        for csv_name in sorted(csv_files):
            parts = csv_name.split("/")
            if len(parts) < 2:
                continue
            split_name = parts[0]  # dev, val, test
            subject = parts[1].replace(f"_{split_name}.csv", "")

            with zf.open(csv_name) as f:
                try:
                    df = pd.read_csv(f, encoding="utf-8-sig")
                except Exception as e:
                    print(f"  WARNING: Could not read {csv_name}: {e}")
                    continue

            has_answer = "answer" in df.columns

            for _, row in df.iterrows():
                q = str(row.get("question", ""))
                options = []
                for opt in ["A", "B", "C", "D"]:
                    if opt in df.columns:
                        options.append(f"{opt}: {row[opt]}")
                options_text = "\n".join(options)
                full_content = f"{q}\n\n{options_text}"

                answer_key = str(row["answer"]).strip() if has_answer else ""

                all_rows.append({
                    "item_id": f"fineval_{item_id:06d}",
                    "question_short": q[:200],
                    "answer_key": answer_key,
                    "config": subject,
                    "split": split_name,
                    "source_dataset": "SUFE-AIFLM-Lab/FinEval",
                    "language": "chinese",
                    "task_type": "multiple_choice",
                    "full_content": full_content,
                })
                item_id += 1

    return all_rows


# ──────────────────────────────────────────────────────────────────────
# 2. CFLUE Knowledge (Chinese financial qualification exams)
# ──────────────────────────────────────────────────────────────────────
def load_cflue_knowledge():
    """Load CFLUE knowledge assessment from JSON."""
    json_path = RAW_DIR / "cflue_knowledge.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found. Skipping CFLUE knowledge.")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = []
    for i, item in enumerate(data):
        q = str(item.get("question", ""))
        choices_raw = item.get("choices", "{}")
        answer_key = str(item.get("answer", "")).strip()
        analysis = str(item.get("analysis", "") or "")
        exam_name = str(item.get("名称", "unknown"))
        task_type_cn = str(item.get("task", ""))

        # Parse choices
        try:
            if isinstance(choices_raw, str):
                choices_dict = eval(choices_raw)  # Safe: controlled data
            else:
                choices_dict = choices_raw
            options_text = "\n".join(
                f"{k}: {v}" for k, v in sorted(choices_dict.items())
            )
        except Exception:
            options_text = str(choices_raw)

        full_content = f"{q}\n\n{options_text}"
        if analysis:
            full_content += f"\n\nAnalysis: {analysis}"

        # Map Chinese task type
        task_map = {
            "单项选择题": "single_choice",
            "多项选择题": "multiple_choice",
            "判断题": "true_false",
        }
        task_type = task_map.get(task_type_cn, "other")

        all_rows.append({
            "item_id": f"cflue_k_{i:06d}",
            "question_short": q[:200],
            "answer_key": answer_key,
            "config": exam_name,
            "split": "test",  # CFLUE knowledge is evaluation data
            "source_dataset": "aliyun/cflue",
            "language": "chinese",
            "task_type": task_type,
            "full_content": full_content,
        })

    return all_rows


# ──────────────────────────────────────────────────────────────────────
# 3. CFLUE Application (Chinese financial NLP tasks)
# ──────────────────────────────────────────────────────────────────────
def load_cflue_application():
    """Load CFLUE application assessment from JSON."""
    json_path = RAW_DIR / "cflue_application.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found. Skipping CFLUE application.")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = []
    for i, item in enumerate(data):
        instruction = str(item.get("instruction", ""))
        inp = str(item.get("input", ""))
        output = str(item.get("output", ""))
        task = str(item.get("task", "unknown"))
        sub_task = str(item.get("sub_task", ""))

        full_content = instruction
        if inp:
            full_content += f"\n\nInput: {inp}"

        config = f"{task}/{sub_task}" if sub_task else task

        all_rows.append({
            "item_id": f"cflue_a_{i:06d}",
            "question_short": instruction[:200],
            "answer_key": output[:200],
            "config": config,
            "split": "test",
            "source_dataset": "aliyun/cflue",
            "language": "chinese",
            "task_type": "generation",
            "full_content": full_content,
        })

    return all_rows


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Financial Benchmarks: Task Metadata & Response Matrix Builder")
    print("=" * 70)

    # Load all sources
    all_rows = []

    print("\n[1/3] Loading FinEval (Chinese)...")
    fineval_rows = load_fineval()
    all_rows.extend(fineval_rows)
    print(f"  Loaded {len(fineval_rows)} items from FinEval")

    print("\n[2/3] Loading CFLUE Knowledge (Chinese)...")
    cflue_k_rows = load_cflue_knowledge()
    all_rows.extend(cflue_k_rows)
    print(f"  Loaded {len(cflue_k_rows)} items from CFLUE Knowledge")

    print("\n[3/3] Loading CFLUE Application (Chinese)...")
    cflue_a_rows = load_cflue_application()
    all_rows.extend(cflue_a_rows)
    print(f"  Loaded {len(cflue_a_rows)} items from CFLUE Application")

    if not all_rows:
        print("\nERROR: No data loaded from any source.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal items loaded: {len(df)}")
    print(f"By source: {df['source_dataset'].value_counts().to_dict()}")
    print(f"By split: {df['split'].value_counts().to_dict()}")
    print(f"By language: {df['language'].value_counts().to_dict()}")
    print(f"By task_type: {df['task_type'].value_counts().to_dict()}")

    # ── task_metadata.csv ──
    meta_cols = [
        "item_id", "question_short", "answer_key", "config",
        "split", "source_dataset", "language", "task_type",
    ]
    meta_df = df[meta_cols].copy()
    meta_path = PROCESSED_DIR / "task_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved task_metadata.csv: {meta_path}")
    print(f"  Shape: {meta_df.shape}")

    # ── item_content.csv ──
    content_df = df[["item_id", "full_content"]].copy()
    content_df.columns = ["item_id", "content"]
    content_path = PROCESSED_DIR / "item_content.csv"
    content_df.to_csv(content_path, index=False)
    print(f"Saved item_content.csv: {content_path}")
    print(f"  Shape: {content_df.shape}")

    # ── response_matrix.csv (gold answers only, no model responses) ──
    # Include items that have answer keys
    has_answer = df[df["answer_key"].str.len() > 0].copy()
    rm = has_answer[["item_id", "answer_key"]].set_index("item_id")
    rm.columns = ["gold_answer"]
    rm_path = PROCESSED_DIR / "response_matrix.csv"
    rm.to_csv(rm_path)
    print(f"Saved response_matrix.csv (gold answers): {rm_path}")
    print(f"  Shape: {rm.shape}")

    # ── model_summary.csv (placeholder) ──
    summary_data = []
    # Add FinBen aggregate data as reference
    finben_note = (
        "FinBen leaderboard: ~36 models evaluated on 42+ financial datasets. "
        "Per-task aggregate scores available in raw/finben_*.csv files. "
        "No per-item predictions published."
    )
    summary_data.append({
        "model": "FinBen_aggregate",
        "source": "TheFinAI/Old-FinBen-Leaderboard",
        "overall_accuracy": None,
        "n_items": None,
        "notes": finben_note,
    })
    summary_df = pd.DataFrame(summary_data)
    summary_path = PROCESSED_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model_summary.csv: {summary_path}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Total items: {len(df)}")
    print(f"  Items with gold answers: {len(has_answer)}")
    print(f"  Sources: {df['source_dataset'].nunique()}")
    print(f"  Languages: {df['language'].unique().tolist()}")
    print(f"  Has model predictions: No")
    print(f"  Has gold answers: Yes (FinEval val/dev + CFLUE)")

    # Per-source breakdown
    for src in df["source_dataset"].unique():
        src_df = df[df["source_dataset"] == src]
        n_with_answer = (src_df["answer_key"].str.len() > 0).sum()
        print(f"\n  {src}:")
        print(f"    Total items: {len(src_df)}")
        print(f"    Items with answers: {n_with_answer}")
        print(f"    Configs: {src_df['config'].nunique()}")
        print(f"    Splits: {src_df['split'].value_counts().to_dict()}")

    # Per-item prediction status
    print("\n" + "=" * 70)
    print("PER-ITEM MODEL PREDICTION DATA STATUS")
    print("=" * 70)
    print("""
  NONE of the following financial benchmarks publish per-item
  model x item response matrices (binary correct/incorrect):

  Source                    Language(s)     Status
  ────────────────────────  ──────────────  ─────────────────────────────
  FinEval                   Chinese         Dataset only, no predictions
  CFLUE                     Chinese         Dataset only, no predictions
  FinBen/FLARE              EN/ZH/ES        Aggregate per-task scores only
  PIXIU/FLARE               English         Aggregate only
  ICE-PIXIU/ICE-FLARE       EN/ZH           Aggregate only
  BBT-Fin/CFLEB             Chinese         External leaderboard only
  CFBenchmark               Chinese         Framework only, no published results
  SuperCLUE-Fin             Chinese         Paper results only
  ArBanking77/AraFinNLP     Arabic          Dataset only, shared task closed
  IndicFinNLP               HI/BN/TE        No public repo found
  FLARE-ES                  Spanish         Part of FinBen spanish
  FinanceBench              English         8-row aggregate summary only
  FinEval-KR                Chinese         Dataset only (625 sample items)
  Open FinLLM Leaderboard   Multi           Aggregate task scores only
  OpenCompass predictions    Multi           Gated; may have FinEval (unconfirmed)

  RECOMMENDATION: To obtain per-item response matrices, you must:
    1. Run evaluations yourself using lm-evaluation-harness or OpenCompass
       on the FinEval val set (1,179 items with ground truth)
    2. Request access to opencompass/compass_academic_predictions on HuggingFace
    3. Run CFLUE evaluation code against target models
    4. Contact FinBen/PIXIU maintainers to request raw per-item data
""")

    print("\nDone!")


if __name__ == "__main__":
    main()
