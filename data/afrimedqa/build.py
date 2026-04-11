"""
Build AfriMed-QA response matrix from per-model per-item evaluation results.

Data source:
  - GitHub: intron-innovation/AfriMed-QA, results/ directory
  - Each model has a subdirectory with CSV files for different datasets/settings
  - MCQ CSV files contain: sample_id, question, answer, preds, correct (binary 0/1)
  - We focus on MCQ tasks with base-prompt, 0-shot evaluation

AfriMed-QA overview:
  - Medical QA benchmark for African healthcare contexts
  - Multiple dataset versions: v1 (3000 MCQs), v2 (3910 MCQs), v2.5 (289 MCQs)
  - Also includes MedQA-USMLE (1273 MCQs) for comparison
  - Questions span 20+ medical specialties
  - Contributors from 16 African countries

Strategy:
  - Primary matrix: afrimed-qa-v2 (base-prompt, 0-shot) — largest item set (3910)
    with 19 models
  - We also incorporate additional models that were only evaluated on other
    dataset versions (v1, v2.5, medqa) by mapping via sample_id overlap
  - For models with multiple dataset versions, we prefer v2 > v1 > v2.5

Outputs:
  - response_matrix.csv: Binary correct/incorrect (rows=items, cols=models)
  - task_metadata.csv: Per-item metadata (question, answer, specialty, country)
  - model_summary.csv: Per-model accuracy and coverage statistics
"""

INFO = {
    'description': 'Build AfriMed-QA response matrix from per-model per-item evaluation results',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2411.15640',
    'data_source_url': 'https://github.com/intron-innovation/AfriMed-QA',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-NC-SA-4.0',
    'citation': """@misc{olatunji2025afrimedqapanafricanmultispecialtymedical,
      title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset}, 
      author={Tobi Olatunji and Charles Nimo and Abraham Owodunni and Tassallah Abdullahi and Emmanuel Ayodele and Mardhiyah Sanni and Chinemelu Aka and Folafunmi Omofoye and Foutse Yuehgoh and Timothy Faniran and Bonaventure F. P. Dossou and Moshood Yekini and Jonas Kemp and Katherine Heller and Jude Chidubem Omeke and Chidi Asuzu MD and Naome A. Etori and Aimérou Ndiaye and Ifeoma Okoh and Evans Doe Ocansey and Wendy Kinara and Michael Best and Irfan Essa and Stephen Edward Moore and Chris Fourie and Mercy Nyamewaa Asiedu},
      year={2025},
      eprint={2411.15640},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      doi={https://doi.org/10.18653/v1/2025.acl-long.96},
      url={https://arxiv.org/abs/2411.15640}, 
}""",
    'tags': ['multilingual'],
}


from pathlib import Path
import os
import sys
import subprocess
import re
import csv

import pandas as pd
import numpy as np

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

REPO_URL = "https://github.com/intron-innovation/AfriMed-QA.git"
REPO_DIR = os.path.join(RAW_DIR, "AfriMed-QA")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
DATA_DIR = os.path.join(REPO_DIR, "data")


def clone_repo():
    """Clone the AfriMed-QA repo into raw/ if not already present."""
    print("STEP 1: Cloning AfriMed-QA repository")
    print("-" * 60)

    if os.path.isdir(REPO_DIR) and os.path.isdir(RESULTS_DIR):
        print(f"  Already cloned: {REPO_DIR}")
        return

    print(f"  Cloning {REPO_URL} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  ERROR: git clone failed:\n{result.stderr}")
        sys.exit(1)

    print(f"  Cloned to: {REPO_DIR}")


def classify_csv(model_dir_name, filename):
    """Classify a CSV file by dataset, prompt type, and shot setting.

    Returns:
        dict with keys: dataset, prompt, shots, or None if not an MCQ file
    """
    fname = filename.lower()
    if not fname.endswith(".csv"):
        return None
    if "mcq" not in fname:
        return None

    # Determine dataset version
    if "afrimed-qa-v2.5" in fname or "afrimed-qa-v2-5" in fname:
        dataset = "afrimedqa-v2.5"
    elif "afrimed-qa-v2" in fname:
        dataset = "afrimedqa-v2"
    elif "afrimed-qa-v1" in fname or "afrimed-qa_" in fname:
        dataset = "afrimedqa-v1"
    elif "medqa" in fname:
        dataset = "medqa"
    else:
        dataset = "unknown"

    # Determine prompt type
    if "instruct-prompt" in fname or "instruct_prompt" in fname or "instruct_0shot" in fname:
        prompt = "instruct"
    else:
        prompt = "base"

    # Determine shot count
    shot_match = re.search(r"(\d+)[_-]?shot", fname)
    shots = int(shot_match.group(1)) if shot_match else 0

    return {"dataset": dataset, "prompt": prompt, "shots": shots}


def read_mcq_csv(filepath):
    """Read an MCQ CSV and extract sample_id + correct column.

    Returns:
        DataFrame with columns [sample_id, correct], or None on failure.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"    WARNING: Could not read {filepath}: {e}")
        return None

    # Normalize column names (some files have unnamed index column)
    if "" in df.columns or "Unnamed: 0" in df.columns:
        idx_col = "" if "" in df.columns else "Unnamed: 0"
        df = df.drop(columns=[idx_col], errors="ignore")

    if "sample_id" not in df.columns:
        return None
    if "correct" not in df.columns:
        return None

    # Extract just what we need
    result = df[["sample_id"]].copy()
    result["correct"] = pd.to_numeric(df["correct"], errors="coerce")

    # Drop rows with no sample_id
    result = result.dropna(subset=["sample_id"])

    return result


def read_mcq_csv_full(filepath):
    """Read an MCQ CSV and extract full metadata.

    Returns:
        DataFrame with all available columns, or None on failure.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception:
        return None

    if "" in df.columns or "Unnamed: 0" in df.columns:
        idx_col = "" if "" in df.columns else "Unnamed: 0"
        df = df.drop(columns=[idx_col], errors="ignore")

    if "sample_id" not in df.columns:
        return None

    return df


def discover_evaluations():
    """Walk the results/ directory and discover all MCQ evaluation files.

    Returns:
        list of dicts: {model, dataset, prompt, shots, filepath, n_items}
    """
    print("\nSTEP 2: Discovering evaluation files")
    print("-" * 60)

    evaluations = []
    model_dirs = sorted(os.listdir(RESULTS_DIR))

    for model_dir_name in model_dirs:
        model_path = os.path.join(RESULTS_DIR, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        csv_files = sorted(os.listdir(model_path))
        for csv_file in csv_files:
            info = classify_csv(model_dir_name, csv_file)
            if info is None:
                continue

            filepath = os.path.join(model_path, csv_file)

            # Check if this CSV has a 'correct' column
            try:
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    if "correct" not in header or "sample_id" not in header:
                        continue
            except (StopIteration, Exception):
                continue

            # Count rows (use csv reader to handle multi-line fields)
            try:
                with open(filepath, "r") as fcount:
                    count_reader = csv.reader(fcount)
                    next(count_reader)  # skip header
                    n_items = sum(1 for _ in count_reader)
            except Exception:
                n_items = 0

            evaluations.append({
                "model_dir": model_dir_name,
                "dataset": info["dataset"],
                "prompt": info["prompt"],
                "shots": info["shots"],
                "filepath": filepath,
                "filename": csv_file,
                "n_items": n_items,
            })

    print(f"  Found {len(evaluations)} MCQ evaluation files across "
          f"{len(model_dirs)} model directories")

    # Summarize by dataset
    from collections import Counter
    ds_counts = Counter(e["dataset"] for e in evaluations)
    for ds, count in sorted(ds_counts.items()):
        print(f"    {ds:20s}: {count} files")

    return evaluations


def select_primary_evaluations(evaluations):
    """Select the best evaluation file for each model.

    Priority: base-prompt > instruct-prompt, 0-shot > few-shot
    For dataset: afrimedqa-v2 > afrimedqa-v1 > afrimedqa-v2.5 > unknown
    We exclude medqa (different benchmark) from the primary matrix.

    Returns:
        list of selected evaluation dicts
    """
    print("\nSTEP 3: Selecting primary evaluation per model")
    print("-" * 60)

    # Filter to base-prompt, 0-shot, AfriMed-QA datasets only
    candidates = [
        e for e in evaluations
        if e["prompt"] == "base"
        and e["shots"] == 0
        and e["dataset"].startswith("afrimedqa")
    ]

    # Group by model directory
    model_candidates = {}
    for e in candidates:
        model = e["model_dir"]
        if model not in model_candidates:
            model_candidates[model] = []
        model_candidates[model].append(e)

    # For each model, pick the best dataset version
    dataset_priority = {"afrimedqa-v2": 0, "afrimedqa-v1": 1, "afrimedqa-v2.5": 2}
    selected = []

    for model, cands in sorted(model_candidates.items()):
        # Sort by priority (lower = better), then by n_items (more = better)
        cands.sort(key=lambda e: (
            dataset_priority.get(e["dataset"], 99),
            -e["n_items"],
        ))
        best = cands[0]
        selected.append(best)
        if len(cands) > 1:
            alt_datasets = [c["dataset"] for c in cands[1:]]
            print(f"  {model:40s} -> {best['dataset']} ({best['n_items']} items)"
                  f"  [also available: {', '.join(alt_datasets)}]")
        else:
            print(f"  {model:40s} -> {best['dataset']} ({best['n_items']} items)")

    # Also add models that only have instruct-prompt or non-zero-shot but
    # are not yet covered. Check for gemini_pro, gemini_ultra, medlm, medpalm2.
    covered_models = {e["model_dir"] for e in selected}
    instruct_candidates = [
        e for e in evaluations
        if e["model_dir"] not in covered_models
        and e["dataset"].startswith("afrimedqa")
        and e["shots"] == 0
    ]
    # Group by model
    instruct_by_model = {}
    for e in instruct_candidates:
        model = e["model_dir"]
        if model not in instruct_by_model:
            instruct_by_model[model] = []
        instruct_by_model[model].append(e)

    for model, cands in sorted(instruct_by_model.items()):
        cands.sort(key=lambda e: (
            dataset_priority.get(e["dataset"], 99),
            -e["n_items"],
        ))
        best = cands[0]
        selected.append(best)
        print(f"  {model:40s} -> {best['dataset']} ({best['n_items']} items)"
              f"  [instruct-prompt]")

    # Check for models with unknown dataset that might be afrimedqa
    unknown_models = set()
    for e in evaluations:
        if (e["model_dir"] not in covered_models
                and e["model_dir"] not in instruct_by_model
                and e["dataset"] == "unknown"
                and e["prompt"] == "base"
                and e["shots"] == 0):
            unknown_models.add(e["model_dir"])

    # For unknown-dataset models, include them if they have 3000+ items
    # (likely afrimedqa-v1 without the version in the filename)
    for model in sorted(unknown_models):
        cands = [
            e for e in evaluations
            if e["model_dir"] == model
            and e["dataset"] == "unknown"
            and e["prompt"] == "base"
            and e["shots"] == 0
        ]
        # Pick the one closest to 3000 or 3910 items
        cands.sort(key=lambda e: -e["n_items"])
        for c in cands:
            # Include files with roughly 3000 or 3910 items (afrimedqa-v1 or v2)
            if c["n_items"] >= 2800:
                selected.append(c)
                print(f"  {model:40s} -> unknown ({c['n_items']} items)"
                      f"  [likely afrimedqa-v1]")
                break

    print(f"\n  Selected {len(selected)} model evaluations total")
    return selected


def build_response_matrix(selected_evals):
    """Build the items x models response matrix.

    Returns:
        response_df: DataFrame (items x models), values are 0/1/NaN
        metadata_df: DataFrame with per-item metadata
    """
    print("\nSTEP 4: Building response matrix")
    print("-" * 60)

    # Collect all sample_ids and their correctness per model
    model_data = {}
    all_sample_ids = set()
    metadata_source = None  # We'll pick the richest metadata file

    for ev in selected_evals:
        model_name = clean_model_name(ev["model_dir"])
        filepath = ev["filepath"]

        df = read_mcq_csv(filepath)
        if df is None:
            print(f"  WARNING: Could not read {filepath}")
            continue

        # Build sample_id -> correct mapping
        correctness = dict(zip(df["sample_id"], df["correct"]))
        model_data[model_name] = correctness
        all_sample_ids.update(correctness.keys())

        # Try to get metadata from the richest file
        if metadata_source is None or ev["n_items"] > 3800:
            full_df = read_mcq_csv_full(filepath)
            if full_df is not None and "specialty" in full_df.columns:
                metadata_source = full_df
            elif full_df is not None and metadata_source is None:
                metadata_source = full_df

    # Sort sample_ids for stable ordering
    sample_ids = sorted(all_sample_ids)
    model_names = sorted(model_data.keys())

    print(f"  Total unique items (sample_ids): {len(sample_ids)}")
    print(f"  Total models: {len(model_names)}")

    # Build the response matrix: rows = items, columns = models
    matrix = {}
    for model_name in model_names:
        correctness = model_data[model_name]
        matrix[model_name] = [
            correctness.get(sid, np.nan) for sid in sample_ids
        ]

    response_df = pd.DataFrame(matrix, index=sample_ids)
    response_df.index.name = "sample_id"

    # Ensure values are numeric 0/1 (not boolean True/False)
    for col in response_df.columns:
        response_df[col] = pd.to_numeric(response_df[col], errors="coerce")
    # Coerce to integer where not NaN
    response_df = response_df.astype("Int64")

    # Build metadata
    metadata_rows = []
    # Load the main dataset file for additional metadata
    main_data_path = os.path.join(DATA_DIR, "afri_med_qa_15k_v2.5_phase_2_15275.csv")
    main_meta = None
    if os.path.exists(main_data_path):
        main_meta = pd.read_csv(main_data_path, low_memory=False)
        main_meta = main_meta.set_index("sample_id")

    # Also build metadata from the richest results CSV
    results_meta = {}
    if metadata_source is not None:
        for _, row in metadata_source.iterrows():
            sid = row.get("sample_id")
            if sid is not None:
                results_meta[sid] = row

    for sid in sample_ids:
        row_data = {"item_id": sid}

        # Try main dataset first
        if main_meta is not None and sid in main_meta.index:
            mrow = main_meta.loc[sid]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            question = str(mrow.get("question", mrow.get("question_clean", "")))
            row_data["question"] = question[:200] if question else ""
            row_data["answer"] = str(mrow.get("correct_answer", ""))
            row_data["specialty"] = str(mrow.get("specialty", ""))
            row_data["country"] = str(mrow.get("country", ""))
            row_data["region_specific"] = str(mrow.get("region_specific", ""))
            row_data["question_type"] = str(mrow.get("question_type", ""))
        elif sid in results_meta:
            rrow = results_meta[sid]
            # Use question from results CSV
            question = str(rrow.get("question", rrow.get("question_y", rrow.get("question_x", ""))))
            row_data["question"] = question[:200] if question else ""
            row_data["answer"] = str(rrow.get("answer", rrow.get("correct_answer", "")))
            row_data["specialty"] = str(rrow.get("specialty", ""))
            row_data["country"] = str(rrow.get("country", ""))
            row_data["region_specific"] = ""
            row_data["question_type"] = "mcq"
        else:
            row_data["question"] = ""
            row_data["answer"] = ""
            row_data["specialty"] = ""
            row_data["country"] = ""
            row_data["region_specific"] = ""
            row_data["question_type"] = "mcq"

        metadata_rows.append(row_data)

    metadata_df = pd.DataFrame(metadata_rows)

    return response_df, metadata_df


def clean_model_name(model_dir_name):
    """Clean/standardize model directory name to a readable model name."""
    name = model_dir_name

    # Specific mappings for known directories
    name_map = {
        "jsl-med-llama-8b": "JSL-MedLlama-3-8B-v2.0",
        "mistral-7b": "Mistral-7B-Instruct-v0.2",
        "phi3-mini-4k": "Phi-3-mini-4k-instruct",
        "Mistral-7B-Instruct-v02": "Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v03": "Mistral-7B-Instruct-v0.3",
        "Meditron-7B-FT": "Meditron-7B",
        "PMC-LLAMA-7B-FT": "PMC-LLaMA-7B",
    }
    if name in name_map:
        return name_map[name]

    return name


def print_matrix_statistics(response_df, metadata_df):
    """Print comprehensive statistics about the response matrix."""
    n_items, n_models = response_df.shape
    total_cells = n_items * n_models

    print(f"\n{'=' * 60}")
    print("  RESPONSE MATRIX STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Items:           {n_items}")
    print(f"  Models:          {n_models}")
    print(f"  Matrix dims:     {n_items} items x {n_models} models")
    print(f"  Total cells:     {total_cells:,}")

    # Fill rate
    n_valid = response_df.notna().sum().sum()
    n_missing = total_cells - n_valid
    print(f"  Valid cells:     {n_valid:,} ({n_valid / total_cells * 100:.1f}%)")
    print(f"  Missing cells:   {n_missing:,} ({n_missing / total_cells * 100:.1f}%)")

    # Correctness distribution
    n_correct = int((response_df == 1).sum().sum())
    n_incorrect = int((response_df == 0).sum().sum())
    print(f"\n  Correct (1):     {n_correct:,} ({n_correct / n_valid * 100:.1f}% of valid)")
    print(f"  Incorrect (0):   {n_incorrect:,} ({n_incorrect / n_valid * 100:.1f}% of valid)")

    # Per-model statistics
    per_model_acc = response_df.mean(axis=0)
    per_model_coverage = response_df.notna().sum(axis=0)
    print(f"\n  Per-model accuracy:")
    print(f"    Best:   {per_model_acc.max() * 100:.1f}% ({per_model_acc.idxmax()})")
    print(f"    Worst:  {per_model_acc.min() * 100:.1f}% ({per_model_acc.idxmin()})")
    print(f"    Median: {per_model_acc.median() * 100:.1f}%")
    print(f"    Mean:   {per_model_acc.mean() * 100:.1f}%")
    print(f"    Std:    {per_model_acc.std() * 100:.1f}%")

    # Per-item statistics
    per_item_acc = response_df.mean(axis=1)
    print(f"\n  Per-item accuracy (across models):")
    print(f"    Min:    {per_item_acc.min() * 100:.1f}%")
    print(f"    Max:    {per_item_acc.max() * 100:.1f}%")
    print(f"    Median: {per_item_acc.median() * 100:.1f}%")
    print(f"    Std:    {per_item_acc.std() * 100:.1f}%")

    # Items solved by no model / all models
    unsolved = (per_item_acc == 0).sum()
    all_solved = (per_item_acc == 1).sum()
    hard = (per_item_acc < 0.1).sum()
    easy = (per_item_acc > 0.9).sum()
    print(f"\n  Item difficulty distribution:")
    print(f"    No model correct (0%):     {unsolved}")
    print(f"    Hard (<10% correct):       {hard}")
    print(f"    Easy (>90% correct):       {easy}")
    print(f"    All models correct (100%): {all_solved}")

    # Specialty breakdown
    if "specialty" in metadata_df.columns:
        spec = metadata_df["specialty"]
        valid_spec = spec[spec.notna() & (spec != "") & (spec != "nan")]
        if len(valid_spec) > 0:
            print(f"\n  Specialty breakdown:")
            spec_counts = valid_spec.value_counts()
            for s, count in spec_counts.head(15).items():
                # Get accuracy for items in this specialty
                mask = metadata_df["specialty"] == s
                spec_items = response_df.loc[metadata_df.loc[mask, "item_id"].values]
                spec_acc = spec_items.mean().mean() * 100
                print(f"    {s:35s}  n={count:4d}  mean_acc={spec_acc:.1f}%")

    # Country breakdown
    if "country" in metadata_df.columns:
        country = metadata_df["country"]
        valid_country = country[country.notna() & (country != "") & (country != "nan")]
        if len(valid_country) > 0:
            print(f"\n  Country breakdown:")
            country_counts = valid_country.value_counts()
            for c, count in country_counts.head(10).items():
                print(f"    {c:10s}  n={count}")

    # Top/bottom models
    ranked = per_model_acc.sort_values(ascending=False)
    print(f"\n  All models ranked by accuracy:")
    for i, (model, acc) in enumerate(ranked.items()):
        cov = int(per_model_coverage[model])
        print(f"    {i + 1:3d}. {model:45s}  {acc * 100:5.1f}%  ({cov} items)")


def build_model_summary(response_df, selected_evals):
    """Build model_summary.csv with per-model statistics."""
    print(f"\nSTEP 5: Building model summary")
    print("-" * 60)

    rows = []
    for model_name in sorted(response_df.columns):
        col = response_df[model_name]
        n_items = int(col.notna().sum())
        n_correct = int((col == 1).sum())
        n_incorrect = int((col == 0).sum())
        accuracy = float(col.mean()) if n_items > 0 else np.nan

        # Find the evaluation info
        ev_info = None
        for ev in selected_evals:
            if clean_model_name(ev["model_dir"]) == model_name:
                ev_info = ev
                break

        dataset = ev_info["dataset"] if ev_info else ""
        prompt_type = ev_info["prompt"] if ev_info else ""

        rows.append({
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "n_items_evaluated": n_items,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "coverage": round(n_items / len(response_df), 4),
            "source_dataset": dataset,
            "prompt_type": prompt_type,
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("accuracy", ascending=False)
    return summary_df


def _extract_item_content():
    """Extract item_content.csv: question + answer options from raw phase_2 CSV."""
    csv_path = os.path.join(
        RAW_DIR, "AfriMed-QA", "data", "afri_med_qa_15k_v2.5_phase_2_15275.csv"
    )
    if not os.path.exists(csv_path):
        print("  No phase_2 raw CSV found; skipping item_content extraction")
        return
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row.get("question_clean")):
            parts.append(str(row["question_clean"]))
        elif pd.notna(row.get("question")):
            parts.append(str(row["question"]))
        if pd.notna(row.get("answer_options")):
            parts.append(str(row["answer_options"])[:500])
        if parts:
            items.append({
                "item_id": str(row.get("sample_id", "")),
                "content": "\n".join(parts)[:2000],
            })
    out_path = os.path.join(PROCESSED_DIR, "item_content.csv")
    pd.DataFrame(items).to_csv(out_path, index=False)
    print(f"  Extracted {len(items)} items to {out_path}")


def main():
    print("AfriMed-QA Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print(f"  Task type:          MCQ (multiple choice questions)")
    print(f"  Focus:              AfriMed-QA benchmark (African medical QA)")
    print()

    # Step 1: Clone repo
    clone_repo()

    # Step 2: Discover evaluation files
    evaluations = discover_evaluations()

    # Step 3: Select primary evaluation per model
    selected_evals = select_primary_evaluations(evaluations)

    # Step 4: Build response matrix
    response_df, metadata_df = build_response_matrix(selected_evals)

    # Print statistics
    print_matrix_statistics(response_df, metadata_df)

    # Step 5: Build model summary
    summary_df = build_model_summary(response_df, selected_evals)

    # ---- Save all outputs ----
    print(f"\nSTEP 6: Saving outputs")
    print("-" * 60)

    # 1. Response matrix (rows=items, columns=models)
    response_path = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    response_df.to_csv(response_path)
    print(f"  Saved: {response_path}")
    print(f"    Shape: {response_df.shape[0]} items x {response_df.shape[1]} models")

    # 2. Task metadata
    meta_path = os.path.join(PROCESSED_DIR, "task_metadata.csv")
    metadata_df.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path}")

    # 3. Model summary
    summary_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Response matrix: {response_df.shape[0]} items x {response_df.shape[1]} models")
    n_valid = response_df.notna().sum().sum()
    total = response_df.shape[0] * response_df.shape[1]
    print(f"  Fill rate:       {n_valid / total * 100:.1f}%")
    print(f"  Score type:      Binary (1=correct, 0=incorrect)")
    print(f"  Task type:       MCQ (multiple choice, base-prompt, 0-shot)")
    print(f"  Benchmark:       AfriMed-QA (African medical QA)")

    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:40s}  {size_kb:.1f} KB")

    # Step 7: Extract item content
    print(f"\nSTEP 7: Extracting item content")
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
