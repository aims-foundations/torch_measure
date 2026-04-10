"""
01_download_raw.py — Download Anthropic alignment-faking-rl dataset (~2.14M RL transcripts)

Source: https://huggingface.co/datasets/Anthropic/alignment-faking-rl
Transcripts studying alignment-faking behavior during RL training.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "alignment_faking_rl"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading Anthropic/alignment-faking-rl...")
    ds = load_dataset("Anthropic/alignment-faking-rl", trust_remote_code=True)

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
