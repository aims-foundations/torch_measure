"""
01_download_raw.py — Download UK AISI AgentHarm dataset (harmful agent tasks)

Source: https://huggingface.co/datasets/ai-safety-institute/AgentHarm
Benchmark from the UK AI Safety Institute for evaluating agent refusal of harmful tasks.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "agentharm"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading ai-safety-institute/AgentHarm...")
    ds = load_dataset("ai-safety-institute/AgentHarm", trust_remote_code=True)

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
