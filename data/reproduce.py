#!/usr/bin/env python3
"""
Reproduce all benchmark response matrices.

Usage:
    python reproduce.py                   # Run all ready benchmarks
    python reproduce.py bfcl swebench     # Run specific benchmarks
    python reproduce.py --list            # List available benchmarks
    python reproduce.py --pending         # Run pending benchmarks instead
    python reproduce.py --no-upload       # Skip uploading to HuggingFace Hub

Each benchmark has a `build.py` that downloads raw data and builds a
response matrix. This script orchestrates: run build.py, generate
visualizations, convert to .pt, and upload to HuggingFace Hub.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

GREEN, RED, YELLOW, NC = "\033[0;32m", "\033[0;31m", "\033[1;33m", "\033[0m"


# Benchmarks with item-level response matrices (models x items, >2 columns).
# Ready for IRT / psychometric analysis with torch_measure.
BENCHMARKS = [
    # Coding & software engineering
    "bfcl", "livecodebench", "swebench", "swebench_full", "swebench_java",
    "swebench_multilingual", "bigcodebench", "evalplus", "cruxeval",
    "swepolybench", "editbench", "aider",
    # Agent benchmarks
    "dpai", "agentdojo", "mlebench", "taubench", "cybench", "corebench",
    "paperbench", "visualwebarena", "appworld", "scienceagentbench",
    "agentbench", "androidworld", "browsergym", "toolbench", "workarena",
    "theagentcompany", "terminal_bench",
    # General knowledge & reasoning
    "mmlupro", "gaia", "hle", "livebench", "matharena", "osworld", "arcagi",
    "alpacaeval", "wildbench", "ultrafeedback", "rewardbench", "judgebench",
    "summeval", "prm800k", "wmt_mqm", "vl_rewardbench", "prism",
    # Vision-language
    "ai2d_test", "hallusionbench", "mathvista_mini", "mmbench_v11", "mme",
    "mmmu_dev_val",
    # Domain-specific
    "financebench", "igakuqa", "igakuqa119", "lawbench", "tumlu", "legaleval",
    # Multilingual
    "afrieval", "afrimedqa", "culturaleval", "bridging_gap", "sib200",
    "ko_leaderboard", "la_leaderboard", "pt_leaderboard", "thai_leaderboard",
    "helm_afr", "helm_cleva", "helm_thaiexam", "rakuda", "tengu",
    # Safety & red teaming
    "aegis", "bbq", "jailbreakbench", "ai_safety_index", "chatgpt_drift",
    # Monitoring & incidents
    "ca_dmv_disengagement", "nhtsa_sgo",
    # Intervention / treatment-response
    "collab_cxr", "metr_early2025", "metr_late2025", "haiid", "genai_learning",
]

# Benchmarks without item-level model responses yet — either no public
# per-item predictions, pairwise preference data, or complex pipelines.
# Run with: python reproduce.py --pending
BENCHMARKS_PENDING = [
    # No public per-item model predictions
    "ceval", "cmmlu", "fineval", "kmmlu",
    # Preference / pairwise datasets
    "arena_140k", "arena_hard", "mtbench", "nectar", "biggen",
    "preference_dissection", "indeterminacy", "hh_rlhf", "oasst", "helpsteer2",
    "shp2", "rewardbench2", "flask", "prometheus", "beavertails",
    "pku_saferlhf", "personalllm", "pickapic",
    # Medical benchmarks (questions only)
    "cmb", "cmexam", "frenchmedmcqa", "kormedmcqa", "medarabiq", "medexpqa",
    "medqa_chinese", "mmedbench", "permedcqa",
    # Complex pipelines
    "webarena",
]


def run_benchmark(name: str, no_upload: bool) -> None:
    """Run a single benchmark's build.py (which handles download → build → visualize → upload)."""
    print(f"\n{GREEN}[INFO]{NC} ========== {name} ==========")

    d = BASE_DIR / name
    if not d.exists():
        raise FileNotFoundError(f"Unknown benchmark: {name}")

    build_script = d / "build.py"
    if not build_script.exists():
        raise FileNotFoundError(f"No build.py found for {name}")

    env = os.environ.copy()
    if no_upload:
        env["NO_UPLOAD"] = "1"
    subprocess.run([sys.executable, str(build_script)], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Reproduce all benchmark response matrices.")
    parser.add_argument("benchmarks", nargs="*", help="Benchmark names (default: all ready)")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--pending", action="store_true",
                        help="Run BENCHMARKS_PENDING instead of BENCHMARKS")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading .pt files to HuggingFace Hub")
    args = parser.parse_args()

    if args.list:
        print(f"Ready benchmarks ({len(BENCHMARKS)}):")
        for b in BENCHMARKS:
            print(f"  {b}")
        print(f"\nPending benchmarks ({len(BENCHMARKS_PENDING)}):")
        for b in BENCHMARKS_PENDING:
            print(f"  {b}")
        return

    default_list = BENCHMARKS_PENDING if args.pending else BENCHMARKS
    targets = args.benchmarks or default_list

    print(f"{GREEN}[INFO]{NC} Reproducing {len(targets)} benchmarks from {BASE_DIR}")

    succeeded, failed = [], []
    for name in targets:
        try:
            run_benchmark(name, no_upload=args.no_upload)
            succeeded.append(name)
        except Exception as e:
            print(f"{RED}[ERROR]{NC} {name}: {e}")
            failed.append(name)

    print(f"\n{'=' * 40}")
    print(f"{GREEN}Succeeded ({len(succeeded)}):{NC} {' '.join(succeeded) or 'none'}")
    if failed:
        print(f"{RED}Failed ({len(failed)}):{NC} {' '.join(failed)}")


if __name__ == "__main__":
    main()
