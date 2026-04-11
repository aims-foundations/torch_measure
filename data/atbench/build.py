"""
01_download_raw.py — Download AI45Research ATBench dataset (agent trajectory safety)

Source: https://huggingface.co/datasets/AI45Research/ATBench
Benchmark for evaluating safety of agent trajectories across diverse tasks.
"""

INFO = {
    'description': 'Download AI45Research ATBench dataset (agent trajectory safety)',
    'testing_condition': '',
    'paper_url': '',
    'data_source_url': 'https://huggingface.co/datasets/AI45Research/ATBench',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'unknown',
    'citation': '@misc{atbench,\n  title={Atbench},\n  howpublished={\\url{https://huggingface.co/datasets/AI45Research/ATBench}},\n}',
    'tags': ['pending'],
}


import sys
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
