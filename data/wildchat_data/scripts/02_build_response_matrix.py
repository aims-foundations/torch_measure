"""
Explore and process WildChat monitoring data.

Source: https://huggingface.co/datasets/allenai/WildChat-1M
Paper: Zhao et al., "WildChat: 1M ChatGPT Interaction Logs in the Wild", 2024

Structure: real user conversations with ChatGPT, with moderation flags per message.
This is production monitoring data — the "items" are real user prompts and the "responses"
include toxicity/safety flags from OpenAI's moderation API.

Output:
  - wildchat_summary.csv: summary statistics
  - model_x_toxicity.csv: model x toxicity-flag cross-tabulation
  - language_distribution.csv: conversation counts by language
"""

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Try parquet sample first
    parquet_path = RAW_DIR / "wildchat_100k_sample.parquet"
    dataset_path = RAW_DIR / "dataset"

    if parquet_path.exists():
        print(f"Loading {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
    elif dataset_path.exists():
        print("Loading from HuggingFace dataset...")
        from datasets import load_from_disk
        ds = load_from_disk(str(dataset_path))
        df = ds["train"].to_pandas() if "train" in ds else ds.to_pandas()
    else:
        print(f"No data found in {RAW_DIR}. Run 01_download_raw.sh first.")
        return

    print(f"Data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Summary
    summary = {"n_conversations": len(df)}

    # Model distribution
    model_col = next((c for c in df.columns if "model" in c.lower()), None)
    if model_col:
        model_counts = df[model_col].value_counts()
        print(f"\nModel distribution ({model_col}):")
        for m, n in model_counts.items():
            print(f"  {m}: {n}")
        summary["n_models"] = int(model_counts.count())

    # Toxicity
    toxic_col = next((c for c in df.columns if c.lower() == "toxic"), None)
    if toxic_col:
        toxic_rate = df[toxic_col].mean()
        print(f"\nToxicity rate: {toxic_rate:.4f} ({df[toxic_col].sum()} / {len(df)})")
        summary["toxic_rate"] = float(toxic_rate)
        summary["n_toxic"] = int(df[toxic_col].sum())

        # Model x toxicity
        if model_col:
            model_toxic = df.groupby(model_col)[toxic_col].agg(["mean", "sum", "count"])
            model_toxic.columns = ["toxic_rate", "n_toxic", "n_total"]
            model_toxic.to_csv(OUTPUT_DIR / "model_x_toxicity.csv")
            print(f"\nModel x Toxicity:")
            print(model_toxic.to_string())

    # Language distribution
    lang_col = next((c for c in df.columns if "language" in c.lower() or "lang" in c.lower()), None)
    if lang_col:
        lang_counts = df[lang_col].value_counts()
        lang_counts.to_csv(OUTPUT_DIR / "language_distribution.csv")
        print(f"\nTop 10 languages:")
        for l, n in lang_counts.head(10).items():
            print(f"  {l}: {n}")
        summary["n_languages"] = int(lang_counts.count())

    # Country distribution
    country_col = next((c for c in df.columns if "country" in c.lower()), None)
    if country_col:
        country_counts = df[country_col].value_counts()
        country_counts.to_csv(OUTPUT_DIR / "country_distribution.csv")
        print(f"\nTop 10 countries:")
        for c, n in country_counts.head(10).items():
            print(f"  {c}: {n}")

    # Save summary
    pd.Series(summary).to_csv(OUTPUT_DIR / "wildchat_summary.csv")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
