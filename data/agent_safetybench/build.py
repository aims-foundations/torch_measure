"""
01_download_raw.py — Download Agent-SafetyBench dataset.

Benchmark for evaluating safety of LLM agents in interactive environments.
Source: https://huggingface.co/datasets/thu-coai/Agent-SafetyBench
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "agent_safetybench"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading thu-coai/Agent-SafetyBench...")
    ds = load_dataset("thu-coai/Agent-SafetyBench", trust_remote_code=True)
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
