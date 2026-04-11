"""
build.py — Build AgentHarm response matrix.

Source: https://huggingface.co/datasets/ai-safety-institute/AgentHarm
Paper: https://arxiv.org/abs/2410.09024 (Andriushchenko et al., UK AISI; ICLR 2025)

The HF dataset releases harmful agent tasks (prompts + grading functions), but per-
model per-item outputs are not public. The paper (Table 9) reports per-model Harm
Score, Refusal Rate, and Non-refusal Harm Score across three attack conditions
(None, Forced tool call, Template) for 15 LLMs on the public test set.

We transcribe Table 9 into a response matrix with:
  - rows = 15 models (subjects)
  - columns = 9 items: {Harm, Refusal, NonRefusalHarm} x {None, Forced, Template}
  - values = metric in [0, 1] (values that are missing in the source, like
    Forced-tool-call rows for Gemini/Llama, are NaN)

We also download the HF dataset for reference metadata.
"""

INFO = {
    'description': 'Build AgentHarm response matrix',
    'testing_condition': """Cells are aggregate refusal/harm rates from paper Table 9 (15 models × 9 conditions). Use for model-level comparison only.""",
    'paper_url': 'https://arxiv.org/abs/2410.09024',
    'data_source_url': 'https://huggingface.co/datasets/ai-safety-institute/AgentHarm',
    'subject_type': 'model',
    'item_type': 'condition',
    'license': 'MIT',
    'citation': """@misc{andriushchenko2025agentharmbenchmarkmeasuringharmfulness,
      title={AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents}, 
      author={Maksym Andriushchenko and Alexandra Souly and Mateusz Dziemian and Derek Duenas and Maxwell Lin and Justin Wang and Dan Hendrycks and Andy Zou and Zico Kolter and Matt Fredrikson and Eric Winsor and Jerome Wynne and Yarin Gal and Xander Davies},
      year={2025},
      eprint={2410.09024},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.09024}, 
}""",
    'tags': ['aggregate-only'],
}


import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Table 9 from arXiv:2410.09024 — public test set results.
# Columns per attack: Harm Score, Refusals, Non-refusal Harm Score.
# Values are in percent; we convert to [0, 1]. NaN means not reported.
NAN = float("nan")

TABLE_9 = [
    # (model, harm_none, refuse_none, nrharm_none,
    #         harm_forced, refuse_forced, nrharm_forced,
    #         harm_template, refuse_template, nrharm_template)
    ("GPT-3.5-Turbo",      62.2,  0.8, 62.7,  63.2,  1.1, 63.9,  62.0,  0.4, 61.9),
    ("GPT-4o-mini",        62.5, 22.0, 77.5,  68.4, 15.5, 77.1,  68.8,  2.3, 69.8),
    ("GPT-4o",             48.4, 48.9, 90.1,  57.7, 40.5, 85.4,  72.7, 13.6, 84.2),
    ("Claude-3-Haiku",     11.1, 83.7, 61.7,  33.9, 63.3, 72.3,   6.6, 90.5, 68.1),
    ("Claude-3-Sonnet",    20.7, 76.1, 79.7,  42.8, 63.1, 80.0,  52.8, 25.7, 70.4),
    ("Claude-3-Opus",      14.4, 85.2, 72.2,  29.5, 84.5, 84.0,  45.7, 41.7, 77.1),
    ("Claude-3.5-Sonnet",  13.5, 85.2, 91.0,  26.9, 80.3, 87.5,  68.7, 16.7, 81.9),
    ("Mistral-Small-2",    72.0,  7.2, 75.8,  73.7,  6.0, 76.1,  72.7,  0.8, 73.0),
    ("Mistral-Large-2",    82.2,  1.1, 83.0,  80.9,  3.4, 83.0,  80.5,  0.0, 80.5),
    ("Gemini-1.0-Pro",      7.4, 75.5, 29.3,   NAN,  NAN,  NAN,  23.3, 42.3, 39.0),
    ("Gemini-1.5-Flash",   20.7, 69.2, 65.2,   NAN,  NAN,  NAN,  56.6,  1.8, 57.5),
    ("Gemini-1.5-Pro",     15.7, 78.4, 69.5,   NAN,  NAN,  NAN,  56.1,  3.5, 58.1),
    ("Llama-3.1-8B",        3.1, 93.9, 40.6,   NAN,  NAN,  NAN,  27.5, 33.3, 39.1),
    ("Llama-3.1-70B",      14.0, 70.45, 45.7,  NAN,  NAN,  NAN,  15.0, 63.2, 40.3),
    ("Llama-3.1-405B",      4.3, 91.2, 33.2,   NAN,  NAN,  NAN,   4.3, 92.0, 41.5),
]

# Column schema for the response matrix.
# Semantics: `refuse_*` is "safe = refuse" -> higher is safer.
# `harm_*` is "harm score" -> lower is safer.
# We keep values as-reported (so rows represent raw scores; users can negate
# or reinterpret as needed via the column name).
ITEM_COLUMNS = [
    "harm_none", "refuse_none", "nrharm_none",
    "harm_forced", "refuse_forced", "nrharm_forced",
    "harm_template", "refuse_template", "nrharm_template",
]


def build_response_matrix() -> pd.DataFrame:
    """Build AgentHarm response matrix from Table 9."""
    models = [r[0] for r in TABLE_9]
    data = np.array([r[1:] for r in TABLE_9], dtype=float) / 100.0
    df = pd.DataFrame(data, index=models, columns=ITEM_COLUMNS)
    df.index.name = "model"
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally pull the HF dataset for metadata (prompts/tasks only).
    dataset_dir = RAW_DIR / "agentharm"
    if not (dataset_dir.exists() and any(dataset_dir.iterdir())):
        try:
            from datasets import load_dataset
            print("Downloading ai-safety-institute/AgentHarm (prompts only)...")
            ds = load_dataset("ai-safety-institute/AgentHarm", trust_remote_code=True)
            ds.save_to_disk(str(dataset_dir))
            print(f"Saved HF snapshot: {ds}")
        except Exception as e:
            print(f"[WARN] Could not download HF dataset: {e}")

    print("Building AgentHarm response matrix from paper Table 9...")
    rm = build_response_matrix()
    print(f"Shape: {rm.shape}  (models x items)")
    print(rm.to_string())

    out_path = PROCESSED_DIR / "response_matrix.csv"
    rm.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    summary = pd.DataFrame({
        "metric": ["n_models", "n_items", "n_nonnull",
                   "mean_harm_none", "mean_refuse_none", "mean_refuse_template"],
        "value": [
            len(rm), rm.shape[1], int(rm.notna().values.sum()),
            rm["harm_none"].mean(), rm["refuse_none"].mean(), rm["refuse_template"].mean(),
        ],
    })
    summary.to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)


if __name__ == "__main__":
    main()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import os
    import subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
