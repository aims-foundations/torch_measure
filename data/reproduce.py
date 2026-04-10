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
    "swepolybench", "editbench",
    # Agent benchmarks
    "dpai", "agentdojo", "mlebench", "taubench", "cybench", "corebench",
    "paperbench", "visualwebarena", "appworld",
    "androidworld", "toolbench", "workarena",
    "theagentcompany", "terminal_bench",
    # General knowledge & reasoning
    "mmlupro", "gaia", "hle", "livebench", "matharena", "osworld", "arcagi",
    "alpacaeval", "wildbench", "ultrafeedback", "rewardbench", "judgebench",
    "summeval", "prm800k", "wmt_mqm", "vl_rewardbench", "prism",
    # Reward / judge / preference benchmarks
    "rewardbench2", "indeterminacy", "flask", "prometheus", "personalllm",
    "helpsteer2",
    # Preference / pairwise datasets with per-item response matrices
    "arena_140k", "arena_hard", "mtbench", "nectar", "biggen",
    "preference_dissection", "oasst", "shp2", "pickapic",
    # Vision-language
    "ai2d_test", "hallusionbench", "mathvista_mini", "mmbench_v11", "mme",
    "mmmu_dev_val",
    # Domain-specific
    "financebench", "igakuqa", "igakuqa119", "lawbench", "tumlu", "legaleval",
    # Multilingual
    "afrieval", "afrimedqa", "culturaleval", "bridging_gap", "sib200",
    "helm_afr", "helm_cleva", "helm_thaiexam", "rakuda", "tengu",
    "kmmlu", "kormedmcqa",
    # Safety & red teaming
    "aegis", "bbq", "jailbreakbench", "chatgpt_drift",
    "beavertails", "pku_saferlhf",
    "faithcot", "llmail_inject",
    # Intervention / treatment-response
    "collab_cxr", "metr_early2025", "metr_late2025", "haiid", "genai_learning",
    # Agent benchmarks
    "clinebench", "webarena",
    # Safety red teaming
    "machiavelli",
]

# Benchmarks with aggregate-only model data (no per-item cells).
# These have multi-model data but at the level of condition/category/metric
# rates, not individual items. Cells are already averages across trials or
# sub-benchmarks. They can still be used for model-level comparisons but
# aren't proper IRT response matrices.
# Run with: python reproduce.py --aggregate
BENCHMARKS_AGGREGATE = [
    # Safety / red teaming (extracted from paper tables)
    "agent_safetybench",       # 16 models x 18 categories (from paper Tables 5+6)
    "agentharm",               # 15 models x 9 conditions (from paper Table 9)
    "agentic_misalignment",    # 18 models x 18 scenario conditions (from appendix)
    # Model × sub-benchmark aggregate scores
    "aider",                   # 178 models x 6 aider benchmarks (pass_rate_2 aggregates)
    "agentbench",              # 29 models x 8 environment types (OS, DB, KG, ...)
    "browsergym",              # 18 agents x 8 sub-benchmarks (MiniWoB, WebArena, ...)
    "ko_leaderboard",          # 1159 models x 9 Korean benchmarks
    "la_leaderboard",          # 69 models x 70 Latin/Iberian benchmarks
    "pt_leaderboard",          # 1148 models x 10 Portuguese benchmarks
    "thai_leaderboard",        # 72 models x 19 Thai benchmarks
    # Governance / safety meta-analytic data (rows are companies/manufacturers)
    "ai_safety_index",         # 8 companies x 6 policy domains
    "ca_dmv_disengagement",    # 16 manufacturers x 7 location types
    "nhtsa_sgo",               # 27 manufacturers x 17 vehicle types
    "scienceagentbench",       # 57 model configs x 4 aggregate metrics (aggregate_results.csv)
]

# Benchmarks without any multi-model evaluation data yet — questions-only
# datasets, catalogs, conversation logs, or pipelines not yet wired up.
# Run with: python reproduce.py --pending
BENCHMARKS_PENDING = [
    # No public per-item model predictions
    "ceval", "cmmlu", "fineval",
    # Preference datasets without model identities
    # (hh_rlhf is chosen/rejected text without per-model labels)
    "hh_rlhf",
    # Medical benchmarks (questions only)
    "cmb", "cmexam", "frenchmedmcqa", "medarabiq", "medexpqa",
    "medqa_chinese", "mmedbench", "permedcqa",
    # Safety / red teaming (no per-item response matrices)
    "apollo_deception", "cot_safety_behaviors", "cot_unfaithfulness",
    "gandalf", "lmsys_toxicchat",
    "reward_hacks", "safeagentbench", "sycophancy_subterfuge",
    "tensortrust", "atbench", "bells", "odcv_bench", "scale_mrt", "trail",
    # AI governance / incidents / risk catalogs
    "aiid", "mit_airisk", "oecd_aim", "responsible_ai_measures",
    "alignment_faking",
    # Large-scale conversation/interaction logs (not per-item)
    "wildchat",
    # Multilingual (questions only, no model predictions)
    "agreval", "asiaeval", "iberbench",
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
    parser.add_argument("--aggregate", action="store_true",
                        help="Run BENCHMARKS_AGGREGATE (aggregate-only, non-IRT) instead of BENCHMARKS")
    parser.add_argument("--pending", action="store_true",
                        help="Run BENCHMARKS_PENDING (no model data) instead of BENCHMARKS")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading .pt files to HuggingFace Hub")
    args = parser.parse_args()

    if args.list:
        print(f"Ready benchmarks ({len(BENCHMARKS)}):")
        for b in BENCHMARKS:
            print(f"  {b}")
        print(f"\nAggregate-only benchmarks ({len(BENCHMARKS_AGGREGATE)}):")
        for b in BENCHMARKS_AGGREGATE:
            print(f"  {b}")
        print(f"\nPending benchmarks ({len(BENCHMARKS_PENDING)}):")
        for b in BENCHMARKS_PENDING:
            print(f"  {b}")
        return

    if args.aggregate:
        default_list = BENCHMARKS_AGGREGATE
    elif args.pending:
        default_list = BENCHMARKS_PENDING
    else:
        default_list = BENCHMARKS
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
