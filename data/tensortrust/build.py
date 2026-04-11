"""
01_download_raw.py — Download Tensor Trust dataset (~563K prompt injection attacks)

Source: https://huggingface.co/datasets/qxcv/tensor-trust
Large-scale dataset of prompt injection attacks from the Tensor Trust game.
"""

INFO = {
    'description': 'Download Tensor Trust dataset (~563K prompt injection attacks)',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2311.01011',
    'data_source_url': 'https://huggingface.co/datasets/qxcv/tensor-trust',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-4.0',
    'citation': """@misc{toyer2023tensortrustinterpretableprompt,
      title={Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game}, 
      author={Sam Toyer and Olivia Watkins and Ethan Adrian Mendes and Justin Svegliato and Luke Bailey and Tiffany Wang and Isaac Ong and Karim Elmaaroufi and Pieter Abbeel and Trevor Darrell and Alan Ritter and Stuart Russell},
      year={2023},
      eprint={2311.01011},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.01011}, 
}""",
    'tags': ['pending'],
}


import sys
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
