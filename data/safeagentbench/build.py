"""
01_download_raw.py — Download SafeAgentBench dataset.

Benchmark for evaluating safety of LLM-based agents.
Source: https://huggingface.co/datasets/safeagentbench/SafeAgentBench
"""

INFO = {
    'description': 'Download SafeAgentBench dataset',
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2412.13178',
    'data_source_url': 'https://huggingface.co/datasets/safeagentbench/SafeAgentBench',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{yin2025safeagentbenchbenchmarksafetask,
      title={SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents}, 
      author={Sheng Yin and Xianghe Pang and Yuanzhuo Ding and Menglan Chen and Yutong Bi and Yichen Xiong and Wenhao Huang and Zhen Xiang and Jing Shao and Siheng Chen},
      year={2025},
      eprint={2412.13178},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2412.13178}, 
}""",
    'tags': ['pending'],
}


import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    save_dir = RAW_DIR / "SafeAgentBench"

    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"SafeAgentBench already exists at {save_dir}, skipping")
        return

    from datasets import load_dataset

    print("Downloading safeagentbench/SafeAgentBench...")
    save_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("safeagentbench/SafeAgentBench")
    for split_name, split_ds in ds.items():
        out_path = save_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))
        print(f"Saved {split_name} ({len(split_ds)} rows) to {out_path}")

    print("Done.")


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
