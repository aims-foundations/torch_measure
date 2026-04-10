"""
01_download_raw.py — Download reasoning safety behaviours dataset (CoT safety labels)

Source: https://huggingface.co/datasets/AISafety-Student/reasoning-safety-behaviours
Labels for safety-relevant behaviors observed in chain-of-thought reasoning.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "reasoning_safety_behaviours"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading AISafety-Student/reasoning-safety-behaviours...")
    ds = load_dataset("AISafety-Student/reasoning-safety-behaviours", trust_remote_code=True)

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
