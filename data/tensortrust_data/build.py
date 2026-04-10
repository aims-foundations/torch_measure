"""
01_download_raw.py — Download Tensor Trust dataset (~563K prompt injection attacks)

Source: https://huggingface.co/datasets/qxcv/tensor-trust
Large-scale dataset of prompt injection attacks from the Tensor Trust game.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "tensor_trust"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading qxcv/tensor-trust...")
    ds = load_dataset("qxcv/tensor-trust", trust_remote_code=True)

    print(f"Saving to {dataset_dir}...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Done: {ds}")


if __name__ == "__main__":
    main()
