"""
01_build_response_matrix.py — Download and build intervention matrices from Collab-CXR data.

Source: https://osf.io/z7apq/
Paper: Yu et al., Scientific Data 2025; Nature Medicine 2024

Downloads 227 radiologists x 324 chest X-ray cases x 4 conditions -> probabilistic diagnosis.

Structure: 2.7M rows = radiologists x cases x pathologies x conditions
Key columns:
  - uid_clean: radiologist ID
  - patient_id: case/image ID
  - pathology: one of ~104 pathology labels
  - with_ai: boolean — whether AI predictions were shown
  - with_ch: boolean — whether clinical history was shown
  - probability: radiologist's assessed probability [0, 1]
  - gt_binary_simple_all: binary ground truth

4 conditions (with_ai x with_ch):
  - (0, 0): image only
  - (0, 1): image + clinical history
  - (1, 0): image + AI predictions
  - (1, 1): image + AI predictions + clinical history

Output:
  - intervention_table.csv: full long-format (WARNING: 2.7M rows, ~500MB)
  - response_matrix_{condition}.csv: radiologist x (patient_pathology) -> probability
    One matrix per condition, aggregated across pathologies per case.
"""

import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_NAMES = {
    (0, 0): "image_only",
    (0, 1): "image_history",
    (1, 0): "image_ai",
    (1, 1): "image_ai_history",
}

FILES = {
    "data_public.txt.gz": "https://osf.io/download/myedf/",
    "clickstream_public.txt.gz": "https://osf.io/download/yf5h4/",
}


def download():
    """Download Collab-CXR dataset from OSF."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"{filename} already exists, skipping")
            continue
        # Remove partial downloads
        dest.unlink(missing_ok=True)
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Downloaded {filename} ({size_mb:.1f} MB)")

    print(f"Done. Raw files in {RAW_DIR}")


def main():
    download()

    raw_path = RAW_DIR / "data_public.txt.gz"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path}")

    print("Loading Collab-CXR data...")
    with gzip.open(raw_path, "rt") as f:
        df = pd.read_csv(f, sep="\t")

    print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Radiologists: {df['uid_clean'].nunique()}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Pathologies: {df['pathology'].nunique()}")
    print(f"with_ai values: {df['with_ai'].unique()}")
    print(f"with_ch values: {df['with_ch'].unique()}")

    # Skip saving the full 2.7M row intervention table — too large
    # Instead save a summary and the per-condition matrices

    # Create condition label
    df["condition"] = df.apply(
        lambda r: CONDITION_NAMES.get((int(r["with_ai"]), int(r["with_ch"])), "unknown"), axis=1
    )
    print(f"\nCondition distribution:\n{df['condition'].value_counts()}")

    # For each condition, build a radiologist x case matrix
    # Aggregate probability across pathologies per case (mean probability = overall diagnostic accuracy proxy)
    for (ai, ch), cond_name in CONDITION_NAMES.items():
        cond_df = df[(df["with_ai"] == ai) & (df["with_ch"] == ch)]
        if len(cond_df) == 0:
            print(f"No data for condition {cond_name}, skipping")
            continue

        # Mean probability per radiologist x patient (averaged across pathologies)
        matrix = cond_df.groupby(["uid_clean", "patient_id"])["probability"].mean().unstack(fill_value=float("nan"))
        matrix.to_csv(OUTPUT_DIR / f"response_matrix_{cond_name}.csv")
        print(f"\n{cond_name}: {matrix.shape[0]} radiologists x {matrix.shape[1]} cases")
        print(f"  Mean probability: {matrix.mean().mean():.3f}")
        print(f"  Density: {matrix.notna().mean().mean():.1%}")

    # Build item_content.csv: map each patient_id to its ground-truth pathologies
    gt_col_ic = "gt_binary_simple_all"
    if gt_col_ic in df.columns:
        positive_pathologies = (
            df[df[gt_col_ic] == 1]
            .groupby("patient_id")["pathology"]
            .apply(lambda x: "; ".join(sorted(x.unique())))
        )
        all_patients = sorted(df["patient_id"].unique())
        item_content_rows = []
        for pid in all_patients:
            if pid in positive_pathologies.index:
                content = f"Patient {pid}: positive for {positive_pathologies[pid]}"
            else:
                content = f"Patient {pid}: no positive findings"
            item_content_rows.append({"item_id": pid, "content": content})
        item_content_df = pd.DataFrame(item_content_rows)
        item_content_df.to_csv(OUTPUT_DIR / "item_content.csv", index=False)
        print(f"\nSaved item_content.csv ({len(item_content_df)} items)")

    # Also build binary accuracy matrices using ground truth
    gt_col = "gt_binary_simple_all"
    if gt_col in df.columns:
        print(f"\n--- Binary accuracy matrices (vs {gt_col}) ---")
        # Binary correct: probability > 0.5 matches ground truth, or probability <= 0.5 matches no-finding
        df["binary_pred"] = (df["probability"] > 0.5).astype(int)
        df["correct"] = (df["binary_pred"] == df[gt_col]).astype(int)

        for (ai, ch), cond_name in CONDITION_NAMES.items():
            cond_df = df[(df["with_ai"] == ai) & (df["with_ch"] == ch)]
            if len(cond_df) == 0:
                continue
            # Accuracy per radiologist x patient (fraction of pathologies correctly classified)
            acc_matrix = cond_df.groupby(["uid_clean", "patient_id"])["correct"].mean().unstack(fill_value=float("nan"))
            acc_matrix.to_csv(OUTPUT_DIR / f"accuracy_matrix_{cond_name}.csv")
            print(f"{cond_name}: accuracy={acc_matrix.mean().mean():.3f}")

    # Summary
    summary = {
        "n_radiologists": int(df["uid_clean"].nunique()),
        "n_patients": int(df["patient_id"].nunique()),
        "n_pathologies": int(df["pathology"].nunique()),
        "n_rows": int(len(df)),
        "conditions": list(CONDITION_NAMES.values()),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary.json")


if __name__ == "__main__":
    main()
