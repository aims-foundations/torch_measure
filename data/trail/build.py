"""
01_download_raw.py — Download PatronusAI TRAIL dataset (OpenTelemetry agent traces)

Source: https://huggingface.co/datasets/PatronusAI/TRAIL
Structured agent traces in OpenTelemetry format for evaluating agentic systems.
"""

INFO = {
    'description': 'Download PatronusAI TRAIL dataset (OpenTelemetry agent traces)',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2505.08638',
    'data_source_url': 'https://huggingface.co/datasets/PatronusAI/TRAIL',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'unknown',
    'citation': """@misc{deshpande2025trailtracereasoningagentic,
      title={TRAIL: Trace Reasoning and Agentic Issue Localization}, 
      author={Darshan Deshpande and Varun Gangal and Hersh Mehta and Jitin Krishnan and Anand Kannappan and Rebecca Qian},
      year={2025},
      eprint={2505.08638},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.08638}, 
}""",
    'tags': ['pending'],
}


import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = RAW_DIR / "trail"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading PatronusAI/TRAIL...")
    ds = load_dataset("PatronusAI/TRAIL", trust_remote_code=True)

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
