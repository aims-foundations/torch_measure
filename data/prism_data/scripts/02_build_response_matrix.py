#!/usr/bin/env python3
"""02_build_response_matrix.py -- Process PRISM (Pluralistic Alignment) dataset.

Loads data from raw/dataset/ (HuggingFace format) or raw/ JSONL files.
PRISM contains multi-turn conversations where diverse participants rate LLM responses.
Builds:
  1. participant x model rating matrix (average rating per participant-model pair)
  2. Summary statistics on participants, models, and rating distributions

Saves outputs to processed/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def try_load_hf_dataset(path: Path) -> pd.DataFrame | None:
    """Try loading a HuggingFace datasets-format directory."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        # Could be a DatasetDict or Dataset
        if hasattr(ds, "keys"):
            # DatasetDict -- combine all splits
            frames = []
            for split_name in ds:
                df = ds[split_name].to_pandas()
                df["_split"] = split_name
                frames.append(df)
            return pd.concat(frames, ignore_index=True)
        else:
            return ds.to_pandas()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PRISM (Pluralistic Alignment) Dataset Processing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Discover and load data
    # ------------------------------------------------------------------
    print(f"\nRaw directory: {RAW_DIR}")
    print(f"Contents:")
    if RAW_DIR.exists():
        for item in sorted(RAW_DIR.iterdir()):
            if item.is_dir():
                sub_items = list(item.iterdir())
                print(f"  [DIR] {item.name}/ ({len(sub_items)} items)")
            else:
                print(f"  {item.name} ({item.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [WARN] Raw directory does not exist: {RAW_DIR}")

    df = None

    # Strategy 1: Try HuggingFace datasets format
    dataset_path = RAW_DIR / "dataset"
    if dataset_path.exists():
        print(f"\nTrying HuggingFace dataset format at {dataset_path}...")
        df = try_load_hf_dataset(dataset_path)
        if df is not None:
            print(f"  Loaded HF dataset: {df.shape}")

    # Strategy 2: Try JSONL files (conversations.jsonl, utterances.jsonl)
    if df is None:
        jsonl_files = sorted(RAW_DIR.glob("*.jsonl"))
        if jsonl_files:
            print(f"\nFound JSONL files: {[f.name for f in jsonl_files]}")
            frames = {}
            for f in jsonl_files:
                try:
                    frames[f.stem] = load_jsonl(f)
                    print(f"  Loaded {f.name}: {frames[f.stem].shape}")
                except Exception as e:
                    print(f"  [WARN] Failed to load {f.name}: {e}")

            # PRISM typically has conversations + utterances
            if "conversations" in frames:
                df = frames["conversations"]
            elif "utterances" in frames:
                df = frames["utterances"]
            elif frames:
                # Just use the first available
                df = list(frames.values())[0]

    # Strategy 3: Try CSV files
    if df is None:
        csv_files = sorted(RAW_DIR.glob("*.csv"))
        if csv_files:
            print(f"\nFound CSV files: {[f.name for f in csv_files]}")
            df = pd.read_csv(csv_files[0])
            print(f"  Loaded {csv_files[0].name}: {df.shape}")

    # Strategy 4: Search recursively
    if df is None:
        print("\nSearching recursively for data files...")
        for ext in ["*.jsonl", "*.json", "*.csv", "*.parquet"]:
            found = sorted(RAW_DIR.rglob(ext))
            if found:
                print(f"  Found {len(found)} {ext} files: {[f.name for f in found[:5]]}")

        # Try parquet
        parquet_files = sorted(RAW_DIR.rglob("*.parquet"))
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            print(f"  Loaded {parquet_files[0].name}: {df.shape}")

    if df is None or df.empty:
        print("\n[WARN] No data files found in raw/. The raw data may need to be downloaded first.")
        print("       Expected: HuggingFace dataset, JSONL files (conversations.jsonl, utterances.jsonl),")
        print("       or CSV files with participant ratings.")
        print("\n       Creating placeholder outputs...")

        # Save a placeholder summary
        pd.DataFrame({"status": ["no_raw_data"]}).to_csv(
            PROCESSED_DIR / "summary_statistics.csv", index=False
        )
        return

    # ------------------------------------------------------------------
    # 2. Explore data structure
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 70)
    print("DATA STRUCTURE")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nColumn types:")
    for col in df.columns:
        nunique = df[col].nunique()
        print(f"  {col}: dtype={df[col].dtype}, nunique={nunique}, null={df[col].isna().sum()}")
    print(f"\nFirst 3 rows (truncated):")
    for i in range(min(3, len(df))):
        print(f"  [{i}]:")
        for k in df.columns:
            val = str(df.iloc[i][k])[:100]
            print(f"    {k}: {val}")

    # ------------------------------------------------------------------
    # 3. Detect key columns
    # ------------------------------------------------------------------
    col_participant = None
    for cand in ["participant_id", "user_id", "annotator_id", "rater_id", "worker_id", "participant"]:
        if cand in df.columns:
            col_participant = cand
            break

    col_model = None
    for cand in ["model_name", "model", "model_id", "system", "bot_name", "agent"]:
        if cand in df.columns:
            col_model = cand
            break

    col_rating = None
    for cand in ["rating", "score", "overall_rating", "preference", "quality", "likert"]:
        if cand in df.columns:
            col_rating = cand
            break

    col_conv = None
    for cand in ["conversation_id", "conv_id", "dialogue_id", "session_id"]:
        if cand in df.columns:
            col_conv = cand
            break

    print(f"\nDetected columns:")
    print(f"  participant: {col_participant}")
    print(f"  model:       {col_model}")
    print(f"  rating:      {col_rating}")
    print(f"  conversation: {col_conv}")

    # ------------------------------------------------------------------
    # 4. Build participant x model rating matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING RESPONSE MATRIX")
    print("=" * 70)

    if col_participant and col_model and col_rating:
        # Convert rating to numeric
        df["_rating_numeric"] = pd.to_numeric(df[col_rating], errors="coerce")
        valid = df.dropna(subset=["_rating_numeric"])
        print(f"\n  Valid ratings: {len(valid)} / {len(df)}")
        print(f"  Rating range: [{valid['_rating_numeric'].min()}, {valid['_rating_numeric'].max()}]")
        print(f"  Rating mean: {valid['_rating_numeric'].mean():.3f}")

        # Pivot: participant x model, values = mean rating
        pivot = valid.pivot_table(
            index=col_participant,
            columns=col_model,
            values="_rating_numeric",
            aggfunc="mean",
        )
        print(f"\n  Participant x Model matrix: {pivot.shape}")
        print(f"  Missing values: {pivot.isna().sum().sum()} ({100*pivot.isna().sum().sum()/(pivot.shape[0]*pivot.shape[1]):.1f}%)")
        print(f"  Participants: {pivot.shape[0]}, Models: {pivot.shape[1]}")

        out_path = PROCESSED_DIR / "response_matrix.csv"
        pivot.to_csv(out_path)
        print(f"  Saved to: {out_path}")

        # Also save count matrix
        count_pivot = valid.pivot_table(
            index=col_participant,
            columns=col_model,
            values="_rating_numeric",
            aggfunc="count",
        )
        out_path = PROCESSED_DIR / "interaction_counts.csv"
        count_pivot.to_csv(out_path)
        print(f"\n  Interaction count matrix: {count_pivot.shape}")
        print(f"  Saved to: {out_path}")

    elif col_participant and col_rating:
        print("\n  No model column found; building participant-level summary.")
        df["_rating_numeric"] = pd.to_numeric(df[col_rating], errors="coerce")
        summary = df.groupby(col_participant)["_rating_numeric"].agg(["mean", "std", "count"])
        out_path = PROCESSED_DIR / "participant_ratings.csv"
        summary.to_csv(out_path)
        print(f"  Saved participant rating summary: {summary.shape}")
        print(f"  Saved to: {out_path}")

    elif col_model and col_rating:
        print("\n  No participant column found; building model-level summary.")
        df["_rating_numeric"] = pd.to_numeric(df[col_rating], errors="coerce")
        summary = df.groupby(col_model)["_rating_numeric"].agg(["mean", "std", "count"])
        out_path = PROCESSED_DIR / "model_ratings.csv"
        summary.to_csv(out_path)
        print(f"  Saved model rating summary: {summary.shape}")
        print(f"  Saved to: {out_path}")

    else:
        print("\n  [WARN] Could not identify participant/model/rating columns.")
        print("         Saving raw data as CSV for manual inspection.")
        out_path = PROCESSED_DIR / "prism_data.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    stats = {"metric": [], "value": []}
    stats["metric"].append("total_records")
    stats["value"].append(len(df))

    if col_participant:
        stats["metric"].append("unique_participants")
        stats["value"].append(df[col_participant].nunique())
    if col_model:
        stats["metric"].append("unique_models")
        stats["value"].append(df[col_model].nunique())
        for model in sorted(df[col_model].unique()):
            stats["metric"].append(f"records_model_{model}")
            stats["value"].append(int((df[col_model] == model).sum()))
    if col_rating:
        stats["metric"].append("mean_rating")
        stats["value"].append(df["_rating_numeric"].mean() if "_rating_numeric" in df.columns else np.nan)
    if col_conv:
        stats["metric"].append("unique_conversations")
        stats["value"].append(df[col_conv].nunique())

    stats_df = pd.DataFrame(stats)
    out_path = PROCESSED_DIR / "summary_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\n  Summary statistics:")
    print(stats_df.to_string(index=False))
    print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("PRISM processing complete.")
    print(f"Outputs in: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
