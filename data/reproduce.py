#!/usr/bin/env python3
"""
Reproduce all benchmark response matrices.

Usage:
    python reproduce.py              # Run everything
    python reproduce.py bfcl         # Run only BFCL
    python reproduce.py bfcl swebench  # Run several
    python reproduce.py --list       # List available benchmarks

Prerequisites:
    pip install pandas requests pyyaml datasets beautifulsoup4 lxml
    pip install matplotlib seaborn scipy   (for visualizations)
    pip install pdfplumber                 (for Cybench / PaperBench PDF extraction)
    pip install evalplus                   (only for evalplus benchmark)

Each benchmark:
    1. Downloads/clones raw data from original sources
    2. Runs the 01_build_response_matrix.py script
    3. Outputs to <benchmark>_data/processed/response_matrix.csv
    4. Generates standard visualizations (heatmap, accuracy, difficulty, correlation)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Colors ───────────────────────────────────────────────────────────────

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


def log_info(msg):
    print(f"{GREEN}[INFO]{NC} {msg}")


def log_warn(msg):
    print(f"{YELLOW}[WARN]{NC} {msg}")


def log_error(msg):
    print(f"{RED}[ERROR]{NC} {msg}")


# ── Helpers ──────────────────────────────────────────────────────────────

def _run(cmd, **kwargs):
    """Run a command, raising on failure unless check=False."""
    return subprocess.run(cmd, **kwargs)


def git_clone(url, dest, *, branch=None, depth=1, lfs_include=None):
    """Clone a git repo if it doesn't already exist."""
    dest = Path(dest)
    if dest.exists():
        log_info(f"  Repo already exists at {dest}, pulling...")
        _run(["git", "-C", str(dest), "pull", "--ff-only"], capture_output=True)
        return
    cmd = ["git", "clone"]
    if depth:
        cmd += ["--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    env = dict(os.environ)
    if lfs_include is not None:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    _run(cmd, check=True, env=env)
    if lfs_include is not None:
        _run(["git", "lfs", "pull", f"--include={lfs_include}"],
             cwd=str(dest), check=False)


def curl_download(url, dest, *, user_agent=None):
    """Download a file if it doesn't already exist."""
    dest = Path(dest)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
    except Exception as e:
        log_warn(f"  Download failed: {e}")


def ensure_dirs(bench_dir, *subdirs):
    """Create standard subdirectories for a benchmark."""
    for sub in subdirs:
        (bench_dir / sub).mkdir(parents=True, exist_ok=True)


def run_script(script_path, *args):
    """Run a Python script."""
    _run([sys.executable, str(script_path)] + list(args), check=True)


def visualize(bench_name):
    """Run the centralized visualization script for a benchmark."""
    viz_script = BASE_DIR / "scripts" / "visualize_response_matrix.py"
    _run([sys.executable, str(viz_script), bench_name], check=False)


# ── Benchmark registry ──────────────────────────────────────────────────
#
# Most benchmarks just need:  mkdir + python3 01_build_response_matrix.py
# Only benchmarks with extra setup steps need a dedicated function.
# The default behavior is applied to any benchmark not in SETUP_FUNCS.

# Benchmarks with item-level response matrices (models x items, >2 columns).
# These are ready for IRT / psychometric analysis with torch_measure.
BENCHMARKS = [
    # Coding & software engineering
    "bfcl",
    "livecodebench",
    "swebench",
    "swebench_full",
    "swebench_java",
    "swebench_multilingual",
    "bigcodebench",
    "evalplus",
    "cruxeval",
    "swepolybench",
    "editbench",
    "aider",
    # Agent benchmarks
    "dpai",
    "agentdojo",
    "mlebench",
    "taubench",
    "cybench",
    "corebench",
    "paperbench",
    "visualwebarena",
    "appworld",
    "scienceagentbench",
    "agentbench",
    "androidworld",
    "browsergym",
    "toolbench",
    "workarena",
    "theagentcompany",
    # General knowledge & reasoning
    "mmlupro",
    "gaia",
    "hle",
    "livebench",
    "matharena",
    "osworld",
    "arcagi",
    "alpacaeval",
    "wildbench",
    "ultrafeedback",
    "rewardbench",
    "judgebench",
    "summeval",
    "prm800k",
    "wmt_mqm",
    "vl_rewardbench",
    "prism",
    # Vision-language
    "ai2d_test",
    "hallusionbench",
    "mathvista_mini",
    "mmbench_v11",
    "mme",
    "mmmu_dev_val",
    # Domain-specific
    "financebench",
    "igakuqa",
    "igakuqa119",
    "lawbench",
    "tumlu",
    "legaleval",
    # Multilingual
    "afrieval",
    "afrimedqa",
    "culturaleval",
    "bridging_gap",
    "sib200",
    "ko_leaderboard",
    "la_leaderboard",
    "pt_leaderboard",
    "thai_leaderboard",
    "helm_afr",
    "helm_cleva",
    "helm_thaiexam",
    "rakuda",
    "tengu",
    # Safety & red teaming
    "aegis",
    "bbq",
    "jailbreakbench",
    "ai_safety_index",
    "chatgpt_drift",
    # Monitoring & incidents
    "ca_dmv_disengagement",
    "nhtsa_sgo",
    # Intervention / treatment-response
    "collab_cxr",
    "metr_early2025",
    "metr_late2025",
    "haiid",
    "genai_learning",
    # Agent benchmarks (complex pipelines)
    "terminal_bench",
]

# Benchmarks without item-level model responses yet.
# These have raw data or item catalogs but no per-item response matrix
# (either no model predictions are publicly available, or the build
# script only produces aggregate/placeholder data).
# Run with:  python reproduce.py --pending
BENCHMARKS_PENDING = [
    # No per-item model predictions available
    "ceval",              # C-Eval: test set requires server submission
    "cmmlu",              # CMMLU: OpenCompass data is gated
    "fineval",            # FinEval: no public per-item results
    "kmmlu",              # KMMLU: no public per-item results
    # Preference / pairwise datasets (not standard response matrices)
    "arena_140k",
    "arena_hard",
    "mtbench",
    "nectar",
    "biggen",
    "preference_dissection",
    "indeterminacy",
    "hh_rlhf",
    "oasst",
    "helpsteer2",
    "shp2",
    "rewardbench2",
    "flask",
    "prometheus",
    "beavertails",
    "pku_saferlhf",
    "personalllm",
    "pickapic",
    # Medical benchmarks (questions only, no per-item model predictions)
    "cmb",
    "cmexam",
    "frenchmedmcqa",
    "kormedmcqa",
    "medarabiq",
    "medexpqa",
    "medqa_chinese",
    "mmedbench",
    "permedcqa",
    # Complex pipelines not yet producing matrices
    "webarena",
]


# ── Setup functions for benchmarks that need extra steps ─────────────────

def setup_bfcl(d):
    ensure_dirs(d, "raw", "processed")
    clone_dir = d / "raw/BFCL-Result"
    if not clone_dir.exists():
        log_info("Cloning BFCL-Result repo...")
        git_clone("https://github.com/HuanzhiMao/BFCL-Result.git", clone_dir, depth=None)
    else:
        log_info("BFCL-Result repo already exists, pulling latest...")
        _run(["git", "-C", str(clone_dir), "pull"], check=False)
    log_info("Building BFCL response matrix...")
    run_script(d / "build.py")


def setup_livecodebench(d):
    ensure_dirs(d, "raw", "processed")
    git_clone("https://github.com/LiveCodeBench/submissions.git", d / "raw/submissions", depth=None)
    git_clone("https://github.com/LiveCodeBench/livecodebench.github.io.git",
              d / "raw/livecodebench.github.io", depth=None)
    log_info("Building LiveCodeBench response matrix...")
    run_script(d / "build.py")


def setup_swebench(d):
    ensure_dirs(d, "raw/results_json", "processed")
    raw_dir = d / "raw/results_json"

    log_info("Downloading SWE-bench submission results via GitHub API...")
    api_url = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        submissions = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        log_warn(f"GitHub API call failed: {e}. Using existing raw data if available.")
        submissions = []

    downloaded = 0
    for sub in submissions:
        if sub.get("type") != "dir":
            continue
        name = sub["name"]
        out_file = raw_dir / f"{name}.json"
        if out_file.exists():
            continue
        results_url = (
            f"https://raw.githubusercontent.com/SWE-bench/experiments/main/"
            f"evaluation/verified/{name}/results/results.json"
        )
        try:
            req2 = urllib.request.Request(results_url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req2, timeout=15).read()
            out_file.write_bytes(data)
            downloaded += 1
            if downloaded % 20 == 0:
                time.sleep(1)
        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"Downloaded {downloaded} new results files. "
          f"Total: {len(list(raw_dir.glob('*.json')))}")

    log_info("Building SWE-bench response matrix...")
    run_script(d / "build.py")


def setup_evalplus(d):
    ensure_dirs(d, "raw/humaneval_samples", "raw/mbpp_samples", "raw/eval_results", "processed")

    log_info("Downloading EvalPlus code samples from GitHub releases...")
    raw = d / "raw"
    releases_url = "https://api.github.com/repos/evalplus/evalplus/releases"
    try:
        req = urllib.request.Request(releases_url, headers={"User-Agent": "Mozilla/5.0"})
        releases = json.loads(urllib.request.urlopen(req, timeout=30).read())
        for rel in releases:
            for asset in rel.get("assets", []):
                name = asset["name"]
                if not name.endswith(".zip"):
                    continue
                dest = raw / "humaneval_samples" if "humaneval" in name.lower() else raw / "mbpp_samples"
                dest_file = dest / name
                if dest_file.exists():
                    continue
                print(f"Downloading {name}...")
                urllib.request.urlretrieve(asset["browser_download_url"], dest_file)
        print("EvalPlus samples downloaded.")
    except Exception as e:
        log_warn(f"{e}. Using existing data.")

    log_info("Building EvalPlus response matrices...")
    try:
        run_script(d / "build.py", "--skip-eval")
    except subprocess.CalledProcessError:
        run_script(d / "build.py")


def setup_mmlupro(d):
    ensure_dirs(d, "raw/eval_results", "processed")
    raw_dir = d / "raw"
    eval_dir = raw_dir / "eval_results"

    log_info("Downloading MMLU-Pro eval results from GitHub...")
    api_url = "https://api.github.com/repos/TIGER-AI-Lab/MMLU-Pro/contents/eval_results"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        listing = json.loads(urllib.request.urlopen(req, timeout=30).read())
        zips = [f for f in listing if f["name"].endswith(".zip")]
        print(f"Found {len(zips)} eval result zip files")
        for z in zips:
            dest = eval_dir / z["name"]
            if dest.exists():
                continue
            print(f"  Downloading {z['name']}...")
            urllib.request.urlretrieve(z["download_url"], dest)
            time.sleep(0.5)
    except Exception as e:
        log_warn(f"{e}. Using existing data.")

    lb_path = raw_dir / "leaderboard_results.csv"
    if not lb_path.exists():
        try:
            url = ("https://huggingface.co/datasets/TIGER-Lab/mmlu_pro_leaderboard_submission/"
                   "resolve/main/leaderboard_results.csv")
            urllib.request.urlretrieve(url, lb_path)
            print("Leaderboard CSV downloaded.")
        except Exception as e:
            log_warn(f"Leaderboard download failed: {e}")

    log_info("Building MMLU-Pro response matrix...")
    run_script(d / "build.py")


def setup_cruxeval(d):
    ensure_dirs(d, "raw", "processed")
    git_clone("https://github.com/facebookresearch/cruxeval.git", d / "raw/cruxeval", depth=None)
    log_info("Building CRUXEval response matrix...")
    run_script(d / "build.py")


def setup_webarena(d):
    ensure_dirs(d, "raw", "processed")

    log_info("Downloading WebArena traces...")
    for folder_id, subdir in [
        ("https://drive.google.com/drive/folders/1tnFTtNBEfJmPQR0x8arHxINGR2bZFvVm",
         d / "raw/072023_release_v1"),
        ("https://drive.google.com/drive/folders/1eFAZkRp6jEbgqJ2LJmhayc1LLHvj1sXH",
         d / "raw/102023_release_v2"),
    ]:
        if not subdir.exists():
            subdir.mkdir(parents=True, exist_ok=True)
            try:
                import gdown
                gdown.download_folder(folder_id, output=str(subdir), quiet=False)
            except Exception:
                log_warn("gdown not installed or Google Drive download failed. "
                         "Install: pip install gdown")

    git_clone("https://github.com/anthropics/agent-evals.git",
              d / "raw/agent-evals", depth=None)

    log_info("Building WebArena response matrix...")
    run_script(d / "build.py")



def setup_agentdojo(d):
    ensure_dirs(d, "processed")
    clone_dir = Path("/tmp/agentdojo_repo")
    git_clone("https://github.com/ethz-spylab/agentdojo.git", clone_dir, depth=None)
    log_info("Building AgentDojo response matrix...")
    run_script(d / "build.py", "--runs-dir", str(clone_dir / "runs"))


def setup_mlebench(d):
    ensure_dirs(d, "processed")
    clone_dir = Path("/tmp/mle-bench")
    if not clone_dir.exists():
        log_info("Cloning MLE-bench repo (with Git LFS)...")
        git_clone("https://github.com/openai/mle-bench.git", clone_dir,
                  depth=None, lfs_include="runs/**")
    log_info("Building MLE-bench response matrix...")
    run_script(d / "build.py")


def setup_editbench(d):
    ensure_dirs(d, "raw_results", "processed")
    clone_dir = Path("/tmp/editbench_repo")
    git_clone("https://github.com/waynchi/editbench.git", clone_dir, depth=None)
    # Copy result JSONs to raw_results/
    results_dir = clone_dir / "results/whole_file"
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            shutil.copy2(f, d / "raw_results" / f.name)
    log_info("Building EDIT-Bench response matrix...")
    run_script(d / "build.py")


def setup_taubench(d):
    ensure_dirs(d, "raw", "processed")
    clone_v1 = Path("/tmp/tau-bench")
    clone_v2 = Path("/tmp/tau2-bench")
    git_clone("https://github.com/sierra-research/tau-bench.git", clone_v1, depth=None)
    git_clone("https://github.com/sierra-research/tau2-bench.git", clone_v2, depth=None)

    log_info("Copying trajectory data...")
    traj_dir = clone_v1 / "trajectories"
    if traj_dir.exists():
        for f in traj_dir.glob("*.json"):
            shutil.copy2(f, d / "raw" / f.name)
    tau2_results = clone_v2 / "results"
    tau2_dest = d / "raw/tau2_results"
    tau2_dest.mkdir(parents=True, exist_ok=True)
    if tau2_results.exists():
        for f in tau2_results.glob("*.json"):
            shutil.copy2(f, tau2_dest / f.name)

    log_info("Building TAU-bench response matrices...")
    run_script(d / "build.py")


def setup_cybench(d):
    ensure_dirs(d, "raw", "processed")
    clone_dir = Path("/tmp/cybench_repo")
    git_clone("https://github.com/andyzorigin/cybench.git", clone_dir, depth=None)
    curl_download("https://cybench.github.io/data/leaderboard.csv", d / "raw/leaderboard.csv")
    curl_download("https://arxiv.org/pdf/2408.08926", d / "raw/cybench_paper.pdf")
    log_info("Building Cybench response matrix (extracting Tables 10-12 from PDF)...")
    run_script(d / "build.py")


def setup_corebench(d):
    ensure_dirs(d, "processed")
    git_clone("https://github.com/siegelz/core-bench.git", Path("/tmp/core-bench"), depth=None)
    log_info("Building CORE-Bench response matrix...")
    run_script(d / "build.py")


def setup_paperbench(d):
    ensure_dirs(d, "raw", "processed")
    curl_download("https://arxiv.org/pdf/2504.01848", d / "raw/paperbench_paper.pdf")
    log_info("Building PaperBench response matrix (extracting Tables 10-18 from PDF)...")
    run_script(d / "build.py")


def setup_aider(d):
    ensure_dirs(d, "raw", "processed")
    log_info("Downloading Aider leaderboard YAML files...")
    for yml in ["edit_leaderboard", "polyglot_leaderboard",
                "o1_polyglot_leaderboard", "refactor_leaderboard"]:
        url = (f"https://raw.githubusercontent.com/Aider-AI/aider/main/"
               f"aider/website/_data/{yml}.yml")
        curl_download(url, d / f"raw/{yml}.yml")
    log_info("Building Aider response matrix (aggregate only)...")
    run_script(d / "build.py")


def setup_swepolybench(d):
    ensure_dirs(d, "raw", "processed")
    git_clone("https://github.com/amazon-science/SWE-PolyBench.git",
              Path("/tmp/swepolybench_repo"), branch="submission", depth=None)
    log_info("Building SWE-PolyBench response matrix...")
    run_script(d / "build.py")


def setup_arcagi(d):
    ensure_dirs(d, "raw", "processed")
    git_clone("https://github.com/fchollet/ARC-AGI.git", d / "raw/arc-agi-1", depth=None)
    log_info("Building ARC-AGI response matrix (downloads from HuggingFace)...")
    run_script(d / "build.py")




# Map benchmark names to their setup functions.
# Benchmarks NOT listed here use the default: mkdir + 01_build_response_matrix.py
SETUP_FUNCS = {
    "bfcl": setup_bfcl,
    "livecodebench": setup_livecodebench,
    "swebench": setup_swebench,
    "evalplus": setup_evalplus,
    "mmlupro": setup_mmlupro,
    "cruxeval": setup_cruxeval,
    "webarena": setup_webarena,
    "agentdojo": setup_agentdojo,
    "mlebench": setup_mlebench,
    "editbench": setup_editbench,
    "taubench": setup_taubench,
    "cybench": setup_cybench,
    "corebench": setup_corebench,
    "paperbench": setup_paperbench,
    "aider": setup_aider,
    "swepolybench": setup_swepolybench,
    "arcagi": setup_arcagi,
}


# ── Main logic ───────────────────────────────────────────────────────────

def run_benchmark(name: str) -> bool:
    """Run a single benchmark. Returns True on success."""
    log_info(f"========== {name} ==========")

    d = BASE_DIR / f"{name}_data"
    if not d.exists() and name not in SETUP_FUNCS:
        log_error(f"Unknown benchmark: {name}")
        return False

    setup_fn = SETUP_FUNCS.get(name)
    if setup_fn:
        setup_fn(d)
    else:
        # Default: mkdir + run build script
        ensure_dirs(d, "raw", "processed")
        build_script = d / "build.py"
        if build_script.exists():
            log_info(f"Building {name} response matrix...")
            run_script(build_script)
        else:
            log_warn(f"No build script found for {name}")

    # Generate visualizations
    log_info(f"Generating {name} visualizations...")
    visualize(name)

    return True


def main():
    parser = argparse.ArgumentParser(description="Reproduce all benchmark response matrices.")
    parser.add_argument("benchmarks", nargs="*", help="Benchmark names (default: all ready)")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--pending", action="store_true",
                        help="Run pending benchmarks (no item-level data yet) instead of ready ones")
    args = parser.parse_args()

    if args.list:
        print(f"Ready benchmarks ({len(BENCHMARKS)}):")
        for b in BENCHMARKS:
            print(f"  {b}")
        print(f"\nPending benchmarks ({len(BENCHMARKS_PENDING)}):")
        for b in BENCHMARKS_PENDING:
            print(f"  {b}")
        return

    if args.pending:
        targets = args.benchmarks if args.benchmarks else BENCHMARKS_PENDING
    else:
        targets = args.benchmarks if args.benchmarks else BENCHMARKS

    succeeded = []
    failed = []

    log_info(f"Starting reproduction of {len(targets)} benchmarks...")
    log_info(f"Base directory: {BASE_DIR}")
    print()

    for name in targets:
        try:
            if run_benchmark(name):
                succeeded.append(name)
                log_info(f"{name}: SUCCESS")
            else:
                failed.append(name)
                log_error(f"{name}: FAILED")
        except Exception as e:
            failed.append(name)
            log_error(f"{name}: FAILED ({e})")
        print()

    # Summary
    print()
    log_info("=" * 30)
    log_info("  Reproduction Summary")
    log_info("=" * 30)
    print(f"{GREEN}Succeeded ({len(succeeded)}):{NC} {' '.join(succeeded) or 'none'}")
    if failed:
        print(f"{RED}Failed ({len(failed)}):{NC} {' '.join(failed)}")
    print()
    log_info("Output matrices are in: <benchmark>_data/processed/response_matrix.csv")
    log_info("Visualizations are in:  <benchmark>_data/figures/")


if __name__ == "__main__":
    main()
