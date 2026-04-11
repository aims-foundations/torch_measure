"""
01_build_response_matrix.py — Download and process WildChat-1M monitoring data.

Source: https://huggingface.co/datasets/allenai/WildChat-1M
Paper: Zhao et al., "WildChat: 1M ChatGPT Interaction Logs in the Wild", 2024

This dataset contains ~1M real ChatGPT conversations collected via a browser extension.
Due to the large size, we save a 100K random sample as parquet by default.
Set WILDCHAT_FULL=1 environment variable to download the full dataset.

Structure: real user conversations with ChatGPT, with moderation flags per message.
This is production monitoring data — the "items" are real user prompts and the "responses"
include toxicity/safety flags from OpenAI's moderation API.

Output:
  - wildchat_summary.csv: summary statistics
  - model_x_toxicity.csv: model x toxicity-flag cross-tabulation
  - language_distribution.csv: conversation counts by language
"""

INFO = {
    'description': 'Download and process WildChat-1M monitoring data',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2405.01470',
    'data_source_url': 'https://huggingface.co/datasets/allenai/WildChat-1M',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'ODC-BY',
    'citation': """@misc{zhao2024wildchat1mchatgptinteraction,
      title={WildChat: 1M ChatGPT Interaction Logs in the Wild}, 
      author={Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie and Yejin Choi and Yuntian Deng},
      year={2024},
      eprint={2405.01470},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.01470}, 
}""",
    'tags': ['pending'],
}


import sys
import os
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download():
    """Download WildChat-1M from HuggingFace, with streaming fallback for large datasets."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "dataset"
    sample_file = RAW_DIR / "wildchat_100k_sample.parquet"
    full_download = os.environ.get("WILDCHAT_FULL", "0") == "1"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"[SKIP] Dataset already exists at: {dataset_dir}")
        return

    if sample_file.exists() and not full_download:
        print(f"[SKIP] Sample parquet already exists at: {sample_file}")
        return

    from datasets import load_dataset

    print("[INFO] Loading WildChat-1M from HuggingFace Hub...")
    print("[INFO] This is a large dataset (~1M conversations). This may take a while.")

    try:
        ds = load_dataset("allenai/WildChat-1M", trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Failed to load full dataset: {e}")
        print("[INFO] Trying streaming mode to get a sample...")

        ds = load_dataset("allenai/WildChat-1M", streaming=True, trust_remote_code=True)
        records = []
        for i, example in enumerate(ds["train"]):
            if i >= 100_000:
                break
            records.append(example)
            if (i + 1) % 10000 == 0:
                print(f"[INFO] Streamed {i + 1} examples...")
        df = pd.DataFrame(records)
        df.to_parquet(sample_file)
        print(f"[INFO] Saved 100K sample to {sample_file}")
        return

    if full_download:
        print("[INFO] Saving full dataset to disk...")
        ds.save_to_disk(str(dataset_dir))
        print(f"[INFO] Full dataset saved to: {dataset_dir}")
    else:
        print("[INFO] Sampling 100K examples from the train split...")
        train = ds["train"]
        n = len(train)
        print(f"[INFO] Full dataset has {n} examples.")
        sample_size = min(100_000, n)
        sample = train.shuffle(seed=42).select(range(sample_size))
        sample.to_parquet(str(sample_file))
        print(f"[INFO] Saved {sample_size} sample to: {sample_file}")

    print("[INFO] WildChat download complete.")


def main():
    download()

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
        print(f"No data found in {RAW_DIR}.")
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
