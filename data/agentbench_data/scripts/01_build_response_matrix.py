"""
Build AgentBench response matrices from published evaluation data.

Data source:
  - AgentBench: Evaluating LLMs as Agents (ICLR 2024)
    Paper: https://arxiv.org/abs/2308.03688
    GitHub: https://github.com/THUDM/AgentBench
  - Table 3 from the paper: Per-model test-set scores across 8 environments
  - 29 models evaluated across 8 environments:
      OS  = Operating System (Success Rate %)
      DB  = Database (Success Rate %)
      KG  = Knowledge Graph (F1 %)
      DCG = Digital Card Game (Reward %)
      LTP = Lateral Thinking Puzzles (Game Progress %)
      HH  = House Holding (Success Rate %)
      WS  = Web Shopping (Reward %)
      WB  = Web Browsing (Step SR %)

Note on data granularity:
  AgentBench does NOT release per-task/per-episode scores publicly.
  The finest granularity available is per-model per-environment aggregate scores
  (Table 3 in the paper). The test set sizes per environment are:
      OS=144, DB=300, KG=150, DCG=20, LTP=50, HH=50, WS=200, WB=100
  Individual episode logs and per-item results are not distributed in the
  GitHub repository or any supplementary materials.

  The response matrix here is therefore (29 models x 8 environments) with
  continuous scores (0-100 scale, representing percentages).

Outputs:
  - response_matrix.csv: Continuous scores (models x environments), 29 x 8
  - response_matrix_with_overall.csv: Same + Overall AgentBench score column
  - model_metadata.csv: Model metadata (type, size, category)
  - environment_metadata.csv: Environment metadata (metric, test size, weight)
  - data_availability_notes.txt: Documentation of data sourcing

"""

import os
import json
import csv

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Raw data from AgentBench paper Table 3 (test set results)
# Each entry: (model_name, model_type, model_size, OA, OS, DB, KG, DCG, LTP, HH, WS, WB)
#   OA  = Overall AgentBench score (weighted)
#   Scores are percentages (0-100 scale)
# ---------------------------------------------------------------------------
PAPER_TABLE3 = [
    # API-based commercial models
    ("gpt-4",             "API",   None,  4.01, 42.4, 32.0, 58.8, 74.5, 16.6, 78.0, 61.1, 29.0),
    ("claude-3",          "API",   None,  3.11, 22.9, 51.7, 34.6, 44.5, 14.3, 70.0, 27.9, 26.0),
    ("glm-4",             "API",   None,  2.89, 29.2, 42.3, 46.3, 34.1, 14.2, 34.0, 61.6, 27.0),
    ("claude-2",          "API",   None,  2.49, 18.1, 27.3, 41.3, 55.5,  8.4, 54.0, 61.4,  0.0),
    ("claude",            "API",   None,  2.44,  9.7, 22.0, 38.9, 40.9,  8.2, 58.0, 55.7, 25.0),
    ("gpt-3.5-turbo",    "API",   None,  2.32, 32.6, 36.7, 25.9, 33.7, 10.5, 16.0, 64.1, 20.0),
    ("text-davinci-003",  "API",   None,  1.71, 20.1, 16.3, 34.9,  3.0,  7.1, 20.0, 61.7, 26.0),
    ("claude-instant",    "API",   None,  1.60, 16.7, 18.0, 20.8,  5.9, 12.6, 30.0, 49.7,  4.0),
    ("chat-bison-001",    "API",   None,  1.39,  9.7, 19.7, 23.0, 16.6,  4.4, 18.0, 60.5, 12.0),
    ("text-davinci-002",  "API",   None,  1.25,  8.3, 16.7, 41.5, 11.8,  0.5, 16.0, 56.3,  9.0),
    # Open-source large (>=65B)
    ("llama-2-70b",       "OSS",   70,   0.78,  9.7, 13.0,  8.0, 21.3,  0.0,  2.0,  5.6, 19.0),
    ("guanaco-65b",       "OSS",   65,   0.54,  8.3, 14.7,  1.9,  0.1,  1.5, 12.0,  0.9, 10.0),
    # Open-source medium (30-34B)
    ("codellama-34b",     "OSS",   34,   0.96,  2.8, 14.0, 23.5,  8.4,  0.7,  4.0, 52.1, 20.0),
    ("vicuna-33b",        "OSS",   33,   0.73, 15.3, 11.0,  1.2, 16.3,  1.0,  6.0, 23.9,  7.0),
    ("wizardlm-30b",      "OSS",   30,   0.46, 13.9, 12.7,  2.9,  0.3,  1.8,  6.0,  4.4,  1.0),
    ("guanaco-33b",       "OSS",   33,   0.39, 11.1,  9.3,  3.2,  0.3,  0.0,  6.0,  6.2,  5.0),
    # Open-source small (13B)
    ("vicuna-13b",        "OSS",   13,   0.93, 10.4,  6.7,  9.4,  0.1,  8.0,  8.0, 41.7, 12.0),
    ("llama-2-13b",       "OSS",   13,   0.77,  4.2, 11.7,  3.6, 26.4,  0.0,  6.0, 25.3, 13.0),
    ("openchat-13b",      "OSS",   13,   0.70, 15.3, 12.3,  5.5,  0.1,  0.0,  0.0, 46.9, 15.0),
    ("wizardlm-13b",      "OSS",   13,   0.66,  9.0, 12.7,  1.7,  1.9,  0.0, 10.0, 43.7, 12.0),
    ("codellama-13b",     "OSS",   13,   0.56,  3.5,  9.7, 10.4,  0.0,  0.0,  0.0, 43.8, 14.0),
    ("koala-13b",         "OSS",   13,   0.34,  3.5,  5.0,  0.4,  0.1,  4.4,  0.0,  3.9,  7.0),
    # Open-source very small (6-12B)
    ("vicuna-7b",         "OSS",    7,   0.56,  9.7,  8.7,  2.5,  0.3,  6.4,  0.0,  2.2,  9.0),
    ("codellama-7b",      "OSS",    7,   0.50,  4.9, 12.7,  8.2,  0.0,  0.0,  2.0, 25.2, 12.0),
    ("llama-2-7b",        "OSS",    7,   0.34,  4.2,  8.0,  2.1,  6.9,  0.0,  0.0, 11.6,  7.0),
    ("codegeex2-6b",      "OSS",    6,   0.27,  1.4,  0.0,  4.8,  0.3,  0.0,  0.0, 20.9, 11.0),
    ("dolly-12b",         "OSS",   12,   0.14,  0.0,  0.0,  0.0,  0.1,  1.2,  0.0,  0.4,  9.0),
    ("chatglm-6b",        "OSS",    6,   0.11,  4.9,  0.3,  0.0,  0.0,  0.0,  0.0,  0.5,  4.9),
    ("oasst-12b",         "OSS",   12,   0.03,  1.4,  0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  1.0),
]

ENVIRONMENTS = ["OS", "DB", "KG", "DCG", "LTP", "HH", "WS", "WB"]

ENVIRONMENT_FULL_NAMES = {
    "OS":  "Operating System",
    "DB":  "Database",
    "KG":  "Knowledge Graph",
    "DCG": "Digital Card Game",
    "LTP": "Lateral Thinking Puzzles",
    "HH":  "House Holding",
    "WS":  "Web Shopping",
    "WB":  "Web Browsing",
}

ENVIRONMENT_METRICS = {
    "OS":  "Success Rate",
    "DB":  "Success Rate",
    "KG":  "F1",
    "DCG": "Reward",
    "LTP": "Game Progress",
    "HH":  "Success Rate",
    "WS":  "Reward",
    "WB":  "Step SR",
}

ENVIRONMENT_TEST_SIZES = {
    "OS": 144, "DB": 300, "KG": 150, "DCG": 20,
    "LTP": 50, "HH": 50, "WS": 200, "WB": 100,
}

ENVIRONMENT_WEIGHTS = {
    "OS": 10.8, "DB": 13.0, "KG": 13.9, "DCG": 12.0,
    "LTP": 3.5, "HH": 13.0, "WS": 30.7, "WB": 11.6,
}


def save_raw_data():
    """Save raw paper data as JSON for reproducibility."""
    records = []
    for row in PAPER_TABLE3:
        model, mtype, msize, oa, os_, db, kg, dcg, ltp, hh, ws, wb = row
        records.append({
            "model": model,
            "type": mtype,
            "size_B": msize,
            "overall": oa,
            "OS": os_, "DB": db, "KG": kg, "DCG": dcg,
            "LTP": ltp, "HH": hh, "WS": ws, "WB": wb,
        })

    raw_path = os.path.join(RAW_DIR, "agentbench_table3_results.json")
    with open(raw_path, "w") as f:
        json.dump({
            "source": "AgentBench: Evaluating LLMs as Agents (ICLR 2024)",
            "paper": "https://arxiv.org/abs/2308.03688",
            "github": "https://github.com/THUDM/AgentBench",
            "table": "Table 3 - Test set results",
            "score_scale": "Percentages (0-100) for environments; weighted score for overall",
            "environments": ENVIRONMENT_FULL_NAMES,
            "metrics": ENVIRONMENT_METRICS,
            "test_sizes": ENVIRONMENT_TEST_SIZES,
            "weights": ENVIRONMENT_WEIGHTS,
            "results": records,
        }, f, indent=2)
    print(f"  Saved raw data: {raw_path}")
    return records


def build_response_matrix(records):
    """Build and save the response matrix CSV (models x environments)."""
    models = [r["model"] for r in records]
    matrix_data = {env: [r[env] for r in records] for env in ENVIRONMENTS}
    matrix_df = pd.DataFrame(matrix_data, index=models)
    matrix_df.index.name = "Model"

    # Save environment-only matrix (primary output)
    out_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    matrix_df.to_csv(out_path)
    print(f"  Saved: {out_path}")

    # Save with overall score included
    overall_scores = [r["overall"] for r in records]
    matrix_with_oa = matrix_df.copy()
    matrix_with_oa.insert(0, "Overall", overall_scores)
    out_path2 = os.path.join(PROCESSED_DIR, "response_matrix_with_overall.csv")
    matrix_with_oa.to_csv(out_path2)
    print(f"  Saved: {out_path2}")

    return matrix_df, overall_scores


def build_model_metadata(records):
    """Save model metadata."""
    rows = []
    for r in records:
        # Categorize model size
        if r["type"] == "API":
            size_cat = "API"
        elif r["size_B"] is not None and r["size_B"] >= 65:
            size_cat = "Large (>=65B)"
        elif r["size_B"] is not None and r["size_B"] >= 30:
            size_cat = "Medium (30-34B)"
        elif r["size_B"] is not None and r["size_B"] >= 13:
            size_cat = "Small (13B)"
        else:
            size_cat = "Very Small (6-12B)"

        rows.append({
            "model": r["model"],
            "type": r["type"],
            "size_B": r["size_B"],
            "size_category": size_cat,
            "overall_score": r["overall"],
        })

    meta_df = pd.DataFrame(rows)
    out_path = os.path.join(PROCESSED_DIR, "model_metadata.csv")
    meta_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return meta_df


def build_environment_metadata():
    """Save environment metadata."""
    rows = []
    for env in ENVIRONMENTS:
        rows.append({
            "environment": env,
            "full_name": ENVIRONMENT_FULL_NAMES[env],
            "metric": ENVIRONMENT_METRICS[env],
            "test_size": ENVIRONMENT_TEST_SIZES[env],
            "weight": ENVIRONMENT_WEIGHTS[env],
        })
    env_df = pd.DataFrame(rows)
    out_path = os.path.join(PROCESSED_DIR, "environment_metadata.csv")
    env_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return env_df


def save_data_availability_notes():
    """Document what data is and isn't publicly available."""
    notes = """AgentBench Data Availability Notes
====================================
Date compiled: 2026-03-05

Source: AgentBench: Evaluating LLMs as Agents (ICLR 2024)
Paper: https://arxiv.org/abs/2308.03688
GitHub: https://github.com/THUDM/AgentBench

What IS available publicly:
---------------------------
1. Per-model per-environment AGGREGATE scores (Table 3 in the paper)
   - 29 models x 8 environments
   - Scores are percentages (success rate, F1, reward, etc.)
   - Test set results only (dev set results not published)

2. Environment metadata (Table 2)
   - Metric type per environment
   - Dev/test split sizes
   - Weights for overall score calculation

3. Task definitions and evaluation code (GitHub repo)
   - Task data files in data/ directory
   - Docker environments for running evaluations
   - Server-client evaluation framework

What is NOT available publicly:
-------------------------------
1. Per-task/per-episode individual scores
   - The paper only reports aggregate scores per environment
   - No per-item result logs are released
   - The GitHub repo contains evaluation infrastructure but not result logs

2. Dev set model performance
   - Only test set results are reported in the paper

3. Individual interaction trajectories/logs
   - The paper includes case studies but not full trajectory data

4. Google Sheets leaderboard
   - Referenced in README but contains the same aggregate data
   - May include additional models submitted after publication

5. HuggingFace dataset (iFurySt/AgentBench)
   - Contains TASK DEFINITIONS only (144 OS tasks)
   - Does NOT contain evaluation results or model scores

Consequence for response matrix:
---------------------------------
The response matrix is (29 models x 8 environments) with CONTINUOUS scores,
not a traditional binary (pass/fail) per-item matrix. This is the finest
publicly available granularity. To obtain per-item scores, one would need
to re-run the full evaluation using the AgentBench framework.
"""
    out_path = os.path.join(PROCESSED_DIR, "data_availability_notes.txt")
    with open(out_path, "w") as f:
        f.write(notes)
    print(f"  Saved: {out_path}")


def print_summary_statistics(matrix_df, overall_scores, records):
    """Print comprehensive summary statistics."""
    matrix = matrix_df.values
    models = matrix_df.index.tolist()
    n_models = len(models)
    n_envs = len(ENVIRONMENTS)

    print(f"\n{'='*70}")
    print(f"  AGENTBENCH RESPONSE MATRIX SUMMARY")
    print(f"{'='*70}")
    print(f"  Source:          Table 3, AgentBench (ICLR 2024)")
    print(f"  Models:          {n_models}")
    print(f"  Environments:    {n_envs}")
    print(f"  Matrix dims:     {n_models} x {n_envs}")
    print(f"  Total cells:     {n_models * n_envs}")
    print(f"  Score type:      Continuous (0-100, percentages)")
    print(f"  Fill rate:       100.0%")

    # Overall score statistics
    oa = np.array(overall_scores)
    print(f"\n  Overall AgentBench Score (weighted):")
    print(f"    Mean:   {oa.mean():.2f}")
    print(f"    Median: {np.median(oa):.2f}")
    print(f"    Min:    {oa.min():.2f} ({models[oa.argmin()]})")
    print(f"    Max:    {oa.max():.2f} ({models[oa.argmax()]})")
    print(f"    Std:    {oa.std():.2f}")

    # Per-model mean across environments
    per_model_mean = matrix.mean(axis=1)
    print(f"\n  Per-model mean score (across 8 environments):")
    print(f"    Mean:   {per_model_mean.mean():.1f}%")
    print(f"    Median: {np.median(per_model_mean):.1f}%")
    print(f"    Min:    {per_model_mean.min():.1f}% ({models[per_model_mean.argmin()]})")
    print(f"    Max:    {per_model_mean.max():.1f}% ({models[per_model_mean.argmax()]})")
    print(f"    Std:    {per_model_mean.std():.1f}%")

    # Per-environment statistics
    print(f"\n  Per-environment statistics:")
    print(f"    {'Env':<6s} {'Full Name':<26s} {'Metric':<14s} "
          f"{'Mean':>6s} {'Median':>7s} {'Min':>6s} {'Max':>6s} "
          f"{'Std':>6s} {'#Test':>6s} {'Weight':>7s}")
    print(f"    {'-'*6} {'-'*26} {'-'*14} "
          f"{'-'*6} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*6} {'-'*7}")
    for i, env in enumerate(ENVIRONMENTS):
        col = matrix[:, i]
        print(f"    {env:<6s} {ENVIRONMENT_FULL_NAMES[env]:<26s} "
              f"{ENVIRONMENT_METRICS[env]:<14s} "
              f"{col.mean():5.1f}% {np.median(col):6.1f}% "
              f"{col.min():5.1f}% {col.max():5.1f}% "
              f"{col.std():5.1f}% "
              f"{ENVIRONMENT_TEST_SIZES[env]:5d} "
              f"{ENVIRONMENT_WEIGHTS[env]:6.1f}")

    # API vs OSS comparison
    api_models = [r for r in records if r["type"] == "API"]
    oss_models = [r for r in records if r["type"] == "OSS"]
    print(f"\n  API vs Open-Source comparison:")
    print(f"    API models:  {len(api_models)}")
    print(f"    OSS models:  {len(oss_models)}")

    api_oa = np.array([r["overall"] for r in api_models])
    oss_oa = np.array([r["overall"] for r in oss_models])
    print(f"    API mean overall:  {api_oa.mean():.2f}")
    print(f"    OSS mean overall:  {oss_oa.mean():.2f}")
    print(f"    Gap:               {api_oa.mean() - oss_oa.mean():.2f}")

    for env in ENVIRONMENTS:
        api_env = np.array([r[env] for r in api_models])
        oss_env = np.array([r[env] for r in oss_models])
        print(f"    {env:>4s}:  API={api_env.mean():5.1f}%  "
              f"OSS={oss_env.mean():5.1f}%  "
              f"gap={api_env.mean()-oss_env.mean():+6.1f}%")

    # Zero-score analysis
    zero_cells = (matrix == 0.0).sum()
    total_cells = n_models * n_envs
    print(f"\n  Zero-score analysis:")
    print(f"    Cells with 0.0%:  {zero_cells} / {total_cells} "
          f"({zero_cells/total_cells*100:.1f}%)")
    for i, env in enumerate(ENVIRONMENTS):
        n_zero = (matrix[:, i] == 0.0).sum()
        if n_zero > 0:
            zero_models = [models[j] for j in range(n_models)
                           if matrix[j, i] == 0.0]
            print(f"    {env:>4s}: {n_zero} models at 0% "
                  f"({', '.join(zero_models[:5])}"
                  f"{'...' if len(zero_models) > 5 else ''})")

    # Top-5 models per environment
    print(f"\n  Top-5 models per environment:")
    for i, env in enumerate(ENVIRONMENTS):
        col = matrix[:, i]
        top_idx = col.argsort()[::-1][:5]
        top_strs = [f"{models[j]} ({col[j]:.1f}%)" for j in top_idx]
        print(f"    {env:>4s}: {', '.join(top_strs)}")

    # Correlation between environments
    print(f"\n  Pairwise Pearson correlations between environments:")
    corr_matrix = np.corrcoef(matrix.T)
    header = "         " + "  ".join(f"{e:>5s}" for e in ENVIRONMENTS)
    print(f"    {header}")
    for i, env_i in enumerate(ENVIRONMENTS):
        row_str = "  ".join(f"{corr_matrix[i,j]:5.2f}" for j in range(n_envs))
        print(f"    {env_i:>5s}   {row_str}")

    # Top-10 overall ranking
    print(f"\n  Overall ranking (all 29 models):")
    sorted_idx = np.argsort(oa)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        r = records[idx]
        size_str = f"{r['size_B']}B" if r['size_B'] else "API"
        print(f"    {rank:2d}. {r['model']:22s} ({r['type']:3s}, {size_str:>5s}) "
              f"  OA={r['overall']:.2f}")


def main():
    print("AgentBench Response Matrix Builder")
    print("=" * 70)

    # Step 1: Save raw data
    print("\n[1/5] Saving raw data...")
    records = save_raw_data()

    # Step 2: Build response matrices
    print("\n[2/5] Building response matrices...")
    matrix_df, overall_scores = build_response_matrix(records)

    # Step 3: Build model metadata
    print("\n[3/5] Building model metadata...")
    model_meta = build_model_metadata(records)

    # Step 4: Build environment metadata
    print("\n[4/5] Building environment metadata...")
    env_meta = build_environment_metadata()

    # Step 5: Save availability notes
    print("\n[5/5] Saving data availability notes...")
    save_data_availability_notes()

    # Print comprehensive statistics
    print_summary_statistics(matrix_df, overall_scores, records)

    # Final file listing
    print(f"\n{'='*70}")
    print(f"  OUTPUT FILES")
    print(f"{'='*70}")
    for subdir in ["raw", "processed"]:
        dirpath = os.path.join(BASE_DIR, subdir)
        if os.path.exists(dirpath):
            for f in sorted(os.listdir(dirpath)):
                fpath = os.path.join(dirpath, f)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {subdir}/{f:45s}  {size_kb:.1f} KB")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
