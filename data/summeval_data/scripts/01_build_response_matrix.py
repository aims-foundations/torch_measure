"""
Build SummEval response matrices from per-annotator evaluation data.

Data source:
  - Yale-LILY/SummEval: model_annotations.aligned.jsonl
    (via Google Cloud Storage)
  - 16 summarization models x 100 CNN/DailyMail source documents
  - 3 expert annotators and 5 crowd workers per summary
  - 4 quality dimensions: coherence, consistency, fluency, relevance (1-5)

Processing:
  1. Download original per-annotator annotations (JSONL)
  2. Build per-dimension response matrices (models x documents) for expert
     and crowd annotations, averaging across annotators
  3. Build overall response matrices (mean of 4 dimensions)
  4. Compute inter-annotator agreement statistics

Outputs:
  - processed/expert_<dim>.csv: Expert ratings per dimension (models x docs)
  - processed/crowd_<dim>.csv: Crowd ratings per dimension (models x docs)
  - processed/expert_overall.csv: Expert mean-of-4-dimensions (models x docs)
  - processed/crowd_overall.csv: Crowd mean-of-4-dimensions (models x docs)
  - processed/annotator_agreement.csv: Inter-annotator agreement statistics
  - processed/model_summary.csv: Per-model summary statistics
"""

import json
import os
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/sfr-summarization-repo-research/"
    "model_annotations.aligned.jsonl"
)

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def download_annotations():
    """Download the original SummEval per-annotator annotations."""
    cache_path = os.path.join(RAW_DIR, "model_annotations.aligned.jsonl")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached data: {cache_path}")
        with open(cache_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        return records

    print(f"  Downloading from {ANNOTATIONS_URL} ...")
    response = urlopen(ANNOTATIONS_URL)
    data = response.read().decode("utf-8")

    # Cache locally
    with open(cache_path, "w") as f:
        f.write(data)
    print(f"  Cached: {cache_path}")

    records = [json.loads(line) for line in data.strip().split("\n") if line.strip()]
    print(f"  Downloaded {len(records)} annotation records")
    return records


def build_response_matrices(records):
    """Build response matrices for expert and crowd annotations."""
    # Collect unique model IDs and document IDs
    model_ids = sorted(set(r["model_id"] for r in records))
    doc_ids = sorted(set(r["id"] for r in records))

    n_models = len(model_ids)
    n_docs = len(doc_ids)

    print(f"\n{'='*60}")
    print(f"  Building response matrices: {n_models} models x {n_docs} documents")
    print(f"{'='*60}")

    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    doc_to_idx = {d: i for i, d in enumerate(doc_ids)}

    # Initialize matrices for each dimension
    expert_matrices = {
        dim: np.full((n_models, n_docs), np.nan) for dim in DIMENSIONS
    }
    crowd_matrices = {
        dim: np.full((n_models, n_docs), np.nan) for dim in DIMENSIONS
    }

    # Also collect per-expert individual scores for agreement analysis
    n_experts = 3
    n_turkers = 5
    expert_individual = {
        dim: np.full((n_models, n_docs, n_experts), np.nan)
        for dim in DIMENSIONS
    }
    crowd_individual = {
        dim: np.full((n_models, n_docs, n_turkers), np.nan)
        for dim in DIMENSIONS
    }

    # Fill matrices
    for rec in records:
        m_idx = model_to_idx[rec["model_id"]]
        d_idx = doc_to_idx[rec["id"]]

        # Expert annotations
        expert_anns = rec.get("expert_annotations", [])
        for dim in DIMENSIONS:
            scores = [a[dim] for a in expert_anns if dim in a]
            if scores:
                expert_matrices[dim][m_idx, d_idx] = np.mean(scores)
            for e_idx, ann in enumerate(expert_anns):
                if e_idx < n_experts and dim in ann:
                    expert_individual[dim][m_idx, d_idx, e_idx] = ann[dim]

        # Crowd annotations
        turker_anns = rec.get("turker_annotations", [])
        for dim in DIMENSIONS:
            scores = [a[dim] for a in turker_anns if dim in a]
            if scores:
                crowd_matrices[dim][m_idx, d_idx] = np.mean(scores)
            for t_idx, ann in enumerate(turker_anns):
                if t_idx < n_turkers and dim in ann:
                    crowd_individual[dim][m_idx, d_idx, t_idx] = ann[dim]

    # Save per-dimension matrices
    all_dfs = {}

    for source, matrices in [("expert", expert_matrices), ("crowd", crowd_matrices)]:
        for dim in DIMENSIONS:
            df = pd.DataFrame(
                matrices[dim], index=model_ids, columns=doc_ids
            )
            df.index.name = "model"
            name = f"{source}_{dim}"
            output_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
            df.to_csv(output_path)
            all_dfs[name] = df

            n_valid = np.sum(~np.isnan(matrices[dim]))
            total = n_models * n_docs
            mean_val = np.nanmean(matrices[dim])
            print(f"  {name}: {n_models} x {n_docs}, "
                  f"valid={n_valid}/{total}, mean={mean_val:.3f}")

    # Build overall matrices (mean of 4 dimensions)
    for source, matrices in [("expert", expert_matrices), ("crowd", crowd_matrices)]:
        overall = np.nanmean(
            np.stack([matrices[dim] for dim in DIMENSIONS], axis=-1), axis=-1
        )
        df = pd.DataFrame(overall, index=model_ids, columns=doc_ids)
        df.index.name = "model"
        name = f"{source}_overall"
        output_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        df.to_csv(output_path)
        all_dfs[name] = df

        mean_val = np.nanmean(overall)
        print(f"  {name}: {n_models} x {n_docs}, mean={mean_val:.3f}")

    return all_dfs, model_ids, doc_ids, expert_individual, crowd_individual


def build_annotator_agreement(expert_individual, crowd_individual, model_ids, doc_ids):
    """Compute inter-annotator agreement statistics."""
    print(f"\n{'='*60}")
    print("  Inter-annotator agreement")
    print(f"{'='*60}")

    rows = []

    for source, individual in [("expert", expert_individual), ("crowd", crowd_individual)]:
        n_annotators = individual[DIMENSIONS[0]].shape[2]

        for dim in DIMENSIONS:
            data = individual[dim]  # (models, docs, annotators)

            # Flatten to (models*docs, annotators) for correlation
            flat = data.reshape(-1, n_annotators)

            # Pairwise correlations
            corrs = []
            for i in range(n_annotators):
                for j in range(i + 1, n_annotators):
                    mask = ~np.isnan(flat[:, i]) & ~np.isnan(flat[:, j])
                    if mask.sum() > 2:
                        corr = np.corrcoef(flat[mask, i], flat[mask, j])[0, 1]
                        corrs.append(corr)

            mean_corr = np.mean(corrs) if corrs else np.nan

            # Mean absolute difference
            diffs = []
            for i in range(n_annotators):
                for j in range(i + 1, n_annotators):
                    mask = ~np.isnan(flat[:, i]) & ~np.isnan(flat[:, j])
                    if mask.sum() > 0:
                        mad = np.mean(np.abs(flat[mask, i] - flat[mask, j]))
                        diffs.append(mad)

            mean_mad = np.mean(diffs) if diffs else np.nan

            rows.append({
                "source": source,
                "dimension": dim,
                "n_annotators": n_annotators,
                "mean_pairwise_corr": mean_corr,
                "mean_abs_diff": mean_mad,
            })

            print(f"  {source}/{dim}: "
                  f"mean_corr={mean_corr:.3f}, mean_abs_diff={mean_mad:.3f}")

    agreement_df = pd.DataFrame(rows)
    output_path = os.path.join(PROCESSED_DIR, "annotator_agreement.csv")
    agreement_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    return agreement_df


def build_model_summary(all_dfs, model_ids):
    """Build per-model summary statistics."""
    print(f"\n{'='*60}")
    print("  Per-model summary")
    print(f"{'='*60}")

    rows = []
    for model in model_ids:
        row = {"model": model}
        for source in ["expert", "crowd"]:
            for dim in DIMENSIONS + ["overall"]:
                name = f"{source}_{dim}"
                if name in all_dfs:
                    row[name] = all_dfs[name].loc[model].mean()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("expert_overall", ascending=False)

    output_path = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    print(f"\n  Top 5 models (by expert overall):")
    for _, r in summary_df.head(5).iterrows():
        print(f"    {r['model']:30s}  expert_overall={r['expert_overall']:.3f}")

    print(f"\n  Bottom 5 models:")
    for _, r in summary_df.tail(5).iterrows():
        print(f"    {r['model']:30s}  expert_overall={r['expert_overall']:.3f}")

    return summary_df


def print_statistics(all_dfs, model_ids, doc_ids):
    """Print detailed statistics."""
    print(f"\n{'='*60}")
    print("  SUMMEVAL STATISTICS")
    print(f"{'='*60}")

    n_models = len(model_ids)
    n_docs = len(doc_ids)

    print(f"\n  Dataset dimensions:")
    print(f"    Models:     {n_models}")
    print(f"    Documents:  {n_docs}")
    print(f"    Dimensions: {DIMENSIONS}")

    for source in ["expert", "crowd"]:
        print(f"\n  {source.upper()} score distributions (1-5 scale):")
        for dim in DIMENSIONS + ["overall"]:
            name = f"{source}_{dim}"
            if name in all_dfs:
                vals = all_dfs[name].values.flatten()
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    print(f"    {dim:15s}  "
                          f"mean={np.mean(valid):.3f}  "
                          f"std={np.std(valid):.3f}  "
                          f"min={np.min(valid):.3f}  "
                          f"max={np.max(valid):.3f}")

    # Output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


def main():
    print("SummEval Response Matrix Builder")
    print("=" * 60)
    print(f"  Raw data dir:       {RAW_DIR}")
    print(f"  Processed data dir: {PROCESSED_DIR}")
    print()

    # Step 1: Download annotations
    print("STEP 1: Downloading SummEval annotations")
    print("-" * 60)
    records = download_annotations()

    # Step 2: Build response matrices
    print("\nSTEP 2: Building response matrices")
    print("-" * 60)
    all_dfs, model_ids, doc_ids, expert_ind, crowd_ind = build_response_matrices(
        records
    )

    # Step 3: Annotator agreement
    print("\nSTEP 3: Inter-annotator agreement")
    print("-" * 60)
    build_annotator_agreement(expert_ind, crowd_ind, model_ids, doc_ids)

    # Step 4: Model summary
    print("\nSTEP 4: Per-model summary")
    print("-" * 60)
    build_model_summary(all_dfs, model_ids)

    # Step 5: Statistics
    print("\nSTEP 5: Detailed statistics")
    print("-" * 60)
    print_statistics(all_dfs, model_ids, doc_ids)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Records:    {len(records)}")
    print(f"  Models:     {len(model_ids)}")
    print(f"  Documents:  {len(doc_ids)}")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Output dir: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
