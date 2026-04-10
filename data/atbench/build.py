"""
01_download_raw.py — Download AI45Research ATBench dataset (agent trajectory safety)

Source: https://huggingface.co/datasets/AI45Research/ATBench
Benchmark for evaluating safety of agent trajectories across diverse tasks.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "atbench"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading AI45Research/ATBench...")
    ds = load_dataset("AI45Research/ATBench", trust_remote_code=True)

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
