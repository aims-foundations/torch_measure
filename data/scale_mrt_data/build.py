"""
01_download_raw.py — Download ScaleAI MRT dataset (agent monitoring trajectories)

Source: https://huggingface.co/datasets/ScaleAI/mrt
Monitoring and Red-Teaming trajectories for evaluating AI agent safety.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "mrt"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading ScaleAI/mrt...")
    ds = load_dataset("ScaleAI/mrt", trust_remote_code=True)

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
