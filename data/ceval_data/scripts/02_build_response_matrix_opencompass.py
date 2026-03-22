#!/usr/bin/env python3
"""
Build response matrix for C-Eval from OpenCompass per-item evaluation results.

Data source status:
  The OpenCompass compass_academic_predictions dataset on HuggingFace does NOT
  currently include C-Eval subjects. As of March 2026, the dataset contains
  CMMLU subjects, MMLU, BBH, and other benchmarks, but NOT C-Eval.

  Other searched sources with NO public per-item C-Eval results:
  - github.com/hkust-nlp/ceval: Code + data only, no model predictions
  - cevalbenchmark.com leaderboard: Only aggregate per-subject scores
  - OpenCompass: Has C-Eval in their eval configs but per-item predictions
    are not in the public compass_academic_predictions dataset
  - Qwen/ChatGLM/Baichuan repos: Only aggregate scores, no per-item data
  - FlagEval: No per-item results publicly available
  - C-Eval test set requires submission to cevalbenchmark.com for scoring

Potential future sources:
  1. OpenCompass compass_academic_predictions may add C-Eval subjects
     (currently gated at huggingface.co/datasets/opencompass/compass_academic_predictions)
  2. Run lm-evaluation-harness locally on C-Eval val set
     (val set has ground truth answers; test set requires server submission)

Current state:
  - task_metadata.csv: 15,631 items (52 subjects, test/val/dev splits)
  - response_matrix.csv: Gold answers only (item_id, gold_answer)
  - model_summary.csv: Empty placeholder

All paths use Path(__file__).resolve().parent.parent
"""

import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _BENCHMARK_DIR / "processed"


def main():
    print("=" * 70)
    print("C-Eval Response Matrix Builder")
    print("=" * 70)

    print()
    print("STATUS: No public per-item model prediction data found for C-Eval.")
    print()
    print("Searched sources (all negative):")
    print("  1. github.com/hkust-nlp/ceval -- Code only, no model predictions")
    print("  2. cevalbenchmark.com -- Aggregate leaderboard scores only")
    print("  3. OpenCompass compass_academic_predictions -- CMMLU yes, C-Eval no")
    print("  4. Qwen/ChatGLM/Baichuan repos -- Aggregate per-subject only")
    print("  5. FlagEval -- No per-item results publicly available")
    print("  6. HuggingFace Open LLM Leaderboard -- English benchmarks only")
    print()
    print("C-Eval test set is unique in that it requires server-side submission")
    print("to cevalbenchmark.com for scoring. Neither the questions' answers nor")
    print("the models' predictions are published per-item.")
    print()
    print("C-Eval val set (1,346 items across 52 subjects) has ground truth")
    print("answers and can be evaluated locally with lm-evaluation-harness.")
    print()
    print("Recommendation:")
    print("  1. Request access to opencompass/compass_academic_predictions")
    print("     (may add C-Eval in future)")
    print("  2. Run lm-evaluation-harness on C-Eval val set for target models")
    print("  3. Check cevalbenchmark.com periodically for data releases")
    print()
    print("Current files:")
    for f in sorted(PROCESSED_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
