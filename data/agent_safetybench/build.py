"""
build.py — Build Agent-SafetyBench response matrix.

Benchmark for evaluating safety of LLM agents in interactive environments.
Source: https://huggingface.co/datasets/thu-coai/Agent-SafetyBench
Paper: https://arxiv.org/abs/2412.14470

The repository/dataset releases prompts + environments, but per-model, per-item
responses are not publicly available. The paper (Tables 5 and 6) does report
per-model safety scores for 16 LLM agents across 10 risk categories / failure
modes. We transcribe those tables to produce a response matrix with:
  - rows = 16 models (subjects)
  - columns = 18 items: 8 risk categories + 10 failure modes
  - values = safety score in [0, 1] (higher = safer)
"""

INFO = {
    'description': 'Build Agent-SafetyBench response matrix',
    'testing_condition': """Cells are aggregate pass rates extracted from paper Tables 5+6, not per-item responses. Rows are agents, columns are 18 risk categories. Use for model-level comparison only, not IRT.""",
    'paper_url': 'https://arxiv.org/abs/2412.14470',
    'data_source_url': 'https://huggingface.co/datasets/thu-coai/Agent-SafetyBench',
    'subject_type': 'agent',
    'item_type': 'risk_category',
    'license': 'MIT',
    'citation': """@misc{zhang2025agentsafetybenchevaluatingsafetyllm,
      title={Agent-SafetyBench: Evaluating the Safety of LLM Agents}, 
      author={Zhexin Zhang and Shiyao Cui and Yida Lu and Jingzhuo Zhou and Junxiao Yang and Hongning Wang and Minlie Huang},
      year={2025},
      eprint={2412.14470},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14470}, 
}""",
    'tags': ['aggregate-only'],
}


import sys
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Table 5 from arXiv:2412.14470 — Safety scores by risk category (%, higher = safer).
# Columns: Total, Behavior, Content, + 8 risk categories (Leak, Property, Spread,
# Physical, Law, Availability, Code, Produce).
# Model name, [Total, Behavior, Content, Leak, Property, Spread, Physical, Law, Availability, Code, Produce]
TABLE_5_RISK_CATEGORIES = [
    "Leak", "Property", "Spread", "Physical", "Law", "Availability", "Code", "Produce",
]
TABLE_5_ROWS = [
    ("Claude-3-Opus",           [59.8, 53.2, 84.9, 60.4, 60.4, 35.6, 61.6, 56.8, 43.2, 60.0, 100.0]),
    ("Claude-3.5-Sonnet",       [59.4, 51.9, 88.6, 57.6, 58.4, 32.4, 69.6, 52.0, 40.4, 64.8, 100.0]),
    ("Claude-3.5-Haiku",        [55.1, 40.7, 86.4, 47.2, 46.0, 33.6, 45.6, 41.2, 26.4, 60.8, 100.0]),
    ("GPT-4o",                  [44.2, 36.9, 72.5, 44.4, 48.4, 12.4, 53.2, 28.8, 35.2, 35.6, 95.6]),
    ("GPT-4-Turbo",             [41.9, 33.9, 72.7, 36.8, 43.2, 12.4, 38.8, 33.2, 37.6, 38.4, 94.4]),
    ("Gemini-1.5-Flash",        [41.6, 34.6, 69.1, 39.2, 41.6, 20.8, 38.8, 32.0, 30.0, 48.4, 82.4]),
    ("Gemini-1.5-Pro",          [37.5, 29.2, 69.3, 30.0, 37.6, 18.8, 28.8, 26.8, 30.8, 42.0, 84.8]),
    ("Qwen2.5-72B-Instruct",    [37.3, 28.6, 71.0, 32.8, 38.0, 12.0, 29.6, 24.0, 35.2, 29.6, 97.2]),
    ("GLM4-9B-Chat",            [36.5, 34.6, 44.3, 38.4, 48.0,  6.0, 41.6, 27.2, 50.8, 23.2, 57.2]),
    ("Llama3.1-405B-Instruct",  [35.4, 24.0, 79.6, 25.2, 27.6, 14.4, 24.4, 32.8, 19.6, 40.4, 98.8]),
    ("DeepSeek-V2.5",           [34.2, 28.6, 55.7, 31.2, 36.8,  8.8, 34.4, 22.0, 33.2, 30.4, 76.8]),
    ("Qwen2.5-14B-Instruct",    [31.9, 24.4, 60.6, 24.4, 31.2, 11.2, 28.0, 20.4, 29.2, 29.2, 81.2]),
    ("GPT-4o-mini",             [31.2, 20.5, 72.5, 28.0, 30.0,  6.8, 24.4, 13.2, 23.6, 25.2, 98.4]),
    ("Llama3.1-70B-Instruct",   [31.2, 21.2, 69.8, 20.0, 28.4, 10.8, 23.2, 20.4, 24.0, 29.6, 93.2]),
    ("Llama3.1-8B-Instruct",    [19.9,  9.9, 58.6, 10.0, 12.4,  6.4, 11.2,  6.8, 12.8, 24.8, 74.8]),
    ("Qwen2.5-7B-Instruct",     [18.8, 13.5, 38.9, 13.2, 15.6,  7.6, 17.6, 10.4, 17.2, 10.8, 57.6]),
]

# Table 6 from arXiv:2412.14470 — Safety scores on 10 failure modes M1..M10 (%, higher = safer).
# [Total, M1..M10]
TABLE_6_ROWS = [
    ("Claude-3-Opus",           [59.8, 86.2, 36.6, 63.6, 59.0, 48.0, 81.1, 35.1, 72.2, 59.5, 81.5]),
    ("Claude-3.5-Sonnet",       [59.4, 89.8, 27.6, 55.8, 58.3, 48.5, 79.5, 18.3, 63.3, 63.4, 81.5]),
    ("Claude-3.5-Haiku",        [55.1, 87.5, 15.2, 35.1, 31.9, 39.9, 68.0,  9.9, 49.4, 64.8, 71.8]),
    ("GPT-4o",                  [44.2, 74.5, 26.1, 37.7, 45.5, 23.5, 74.6,  9.9, 49.4, 42.2, 67.7]),
    ("GPT-4-Turbo",             [41.9, 73.5, 20.6, 42.9, 35.2, 24.2, 72.1, 19.8, 50.6, 36.9, 69.4]),
    ("Gemini-1.5-Flash",        [41.6, 71.4, 19.5, 19.5, 28.6, 27.6, 64.8, 20.6, 34.2, 49.7, 63.7]),
    ("Gemini-1.5-Pro",          [37.5, 70.7, 18.7, 27.3, 23.8, 22.1, 70.5, 35.9, 36.7, 28.8, 65.3]),
    ("Qwen2.5-72B-Instruct",    [37.3, 73.5, 19.1, 19.5, 24.4, 17.6, 65.6,  6.9, 35.4, 38.8, 65.3]),
    ("GLM4-9B-Chat",            [36.5, 45.4, 45.5, 27.3, 34.9, 19.2, 60.7,  9.9, 45.6, 36.9, 58.1]),
    ("Llama3.1-405B-Instruct",  [35.4, 81.4,  6.6, 16.9, 21.4, 30.2, 51.6, 11.5, 29.1, 21.2, 60.5]),
    ("DeepSeek-V2.5",           [34.2, 57.9, 15.6, 29.9, 23.5, 16.2, 70.5,  8.4, 44.3, 38.3, 68.5]),
    ("Qwen2.5-14B-Instruct",    [31.9, 62.2, 14.8, 16.9, 21.7, 16.6, 62.3,  5.3, 34.2, 27.7, 62.1]),
    ("GPT-4o-mini",             [31.2, 74.7,  6.2, 11.7, 13.8,  8.1, 68.0,  2.3, 24.1, 31.3, 61.3]),
    ("Llama3.1-70B-Instruct",   [31.2, 71.9,  8.6, 11.7, 16.0, 16.9, 49.2,  3.8, 25.3, 28.2, 57.3]),
    ("Llama3.1-8B-Instruct",    [19.9, 58.9,  3.1,  9.1,  5.4,  7.4, 32.0,  0.8, 17.7, 15.6, 33.1]),
    ("Qwen2.5-7B-Instruct",     [18.8, 41.6,  6.6,  7.8,  8.7,  5.7, 42.6,  1.5, 19.0, 16.5, 42.7]),
]

FAILURE_MODES = [f"M{i}" for i in range(1, 11)]


def build_response_matrix() -> pd.DataFrame:
    """Build Agent-SafetyBench response matrix from paper tables.

    Rows: models, Columns: 8 risk categories + 10 failure modes (18 items total).
    Values: safety score in [0, 1] (higher = safer).
    """
    rows = {}
    for model, vals in TABLE_5_ROWS:
        # Drop Total/Behavior/Content (first 3 cols), keep 8 risk categories
        risk_vals = vals[3:]
        rows[model] = dict(zip(TABLE_5_RISK_CATEGORIES, risk_vals, strict=True))

    for model, vals in TABLE_6_ROWS:
        # Drop Total (first col), keep 10 failure modes
        fm_vals = vals[1:]
        rows[model].update(dict(zip(FAILURE_MODES, fm_vals, strict=True)))

    df = pd.DataFrame(rows).T  # rows=models, cols=items
    df.index.name = "model"
    # Convert from percentages (0-100) to [0, 1]
    df = df / 100.0
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally download the dataset (prompts only) for reference / metadata.
    dataset_dir = RAW_DIR / "agent_safetybench"
    if not (dataset_dir.exists() and any(dataset_dir.iterdir())):
        try:
            from datasets import load_dataset
            print("Downloading thu-coai/Agent-SafetyBench (prompts only)...")
            ds = load_dataset("thu-coai/Agent-SafetyBench", trust_remote_code=True)
            ds.save_to_disk(str(dataset_dir))
            print(f"Saved HF snapshot: {ds}")
        except Exception as e:
            print(f"[WARN] Could not download HF dataset: {e}")

    print("Building Agent-SafetyBench response matrix from paper tables...")
    rm = build_response_matrix()
    print(f"Shape: {rm.shape}  (models x items)")
    print(rm.to_string())

    out_path = PROCESSED_DIR / "response_matrix.csv"
    rm.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    # Also save a summary
    summary = pd.DataFrame({
        "metric": ["n_models", "n_items", "mean_score", "min_score", "max_score"],
        "value": [len(rm), rm.shape[1], rm.values.mean(), rm.values.min(), rm.values.max()],
    })
    summary.to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"Saved summary to {PROCESSED_DIR / 'summary_statistics.csv'}")


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
