"""
01_download_raw.py — Download School of Reward Hacks dataset.

Dataset of reward hacking behaviors in LLM agents.
Source: https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    save_dir = RAW_DIR / "school-of-reward-hacks"

    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"school-of-reward-hacks already exists at {save_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading longtermrisk/school-of-reward-hacks...")
    save_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("longtermrisk/school-of-reward-hacks")
    for split_name, split_ds in ds.items():
        out_path = save_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))
        print(f"Saved {split_name} ({len(split_ds)} rows) to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
