"""
01_download_raw.py — Download Anthropic alignment-faking-rl dataset (~2.14M RL transcripts)

Source: https://huggingface.co/datasets/Anthropic/alignment-faking-rl
Transcripts studying alignment-faking behavior during RL training.
"""

INFO = {
    'description': 'Download Anthropic alignment-faking-rl dataset (~2.14M RL transcripts)',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2412.14093',
    'data_source_url': 'https://huggingface.co/datasets/Anthropic/alignment-faking-rl',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{greenblatt2024alignmentfakinglargelanguage,
      title={Alignment faking in large language models}, 
      author={Ryan Greenblatt and Carson Denison and Benjamin Wright and Fabien Roger and Monte MacDiarmid and Sam Marks and Johannes Treutlein and Tim Belonax and Jack Chen and David Duvenaud and Akbir Khan and Julian Michael and Sören Mindermann and Ethan Perez and Linda Petrini and Jonathan Uesato and Jared Kaplan and Buck Shlegeris and Samuel R. Bowman and Evan Hubinger},
      year={2024},
      eprint={2412.14093},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.14093}, 
}""",
    'tags': ['pending'],
}


import sys
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
