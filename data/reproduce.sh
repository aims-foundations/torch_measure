#!/usr/bin/env bash
# reproduce.sh — Re-download all raw data and rebuild all response matrices.
#
# Usage:
#   bash reproduce.sh           # Run everything
#   bash reproduce.sh bfcl      # Run only the BFCL benchmark
#   bash reproduce.sh --list    # List all available benchmarks
#
# Prerequisites:
#   conda activate info-gathering   (or any Python 3.11+ env)
#   pip install pandas requests pyyaml datasets beautifulsoup4 lxml
#   pip install matplotlib seaborn     (needed for visualization scripts)
#   pip install pdfplumber            (needed for Cybench and PaperBench PDF extraction)
#   pip install evalplus             (only needed for evalplus benchmark)
#
# Each benchmark section:
#   1. Downloads/clones raw data from original sources
#   2. Runs the 01_build_response_matrix.py script
#   3. Outputs to <benchmark>_data/processed/response_matrix.csv
#   4. Generates visualizations via 02_visualize_response_matrix.py
#      (heatmaps, bar charts, etc. saved to <benchmark>_data/figures/)
#
# Data source types:
#   - Git clone:     Clone GitHub repo with per-task results (most benchmarks)
#   - API download:  Fetch from GitHub API, TeamCity API, or HuggingFace
#   - Google Drive:  Download execution traces via gdown (WebArena)
#   - Paper PDF:     Per-task data extracted programmatically from paper PDFs using pdfplumber
#                    (Cybench arXiv:2408.08926, PaperBench arXiv:2504.01848)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

BENCHMARKS=(
    terminal_bench
    bfcl
    livecodebench
    swebench
    swebench_full
    swebench_java
    swebench_multilingual
    bigcodebench
    evalplus
    mmlupro
    cruxeval
    webarena
    dpai
    agentdojo
    mlebench
    editbench
    taubench
    cybench
    corebench
    paperbench
    aider
    swepolybench
    visualwebarena
    appworld
    scienceagentbench
    agentbench
    alpacaeval
    androidworld
    arcagi
    browsergym
    gaia
    hle
    livebench
    matharena
    osworld
    theagentcompany
    toolbench
    wildbench
    workarena
    arena_140k
    arena_hard
    mtbench
    ultrafeedback
    nectar
    biggen
    preference_dissection
    rewardbench
    indeterminacy
    hh_rlhf
    oasst
    helpsteer2
    shp2
    rewardbench2
    summeval
    flask
    prometheus
    judgebench
    prm800k
    beavertails
    pku_saferlhf
    prism
    personalllm
    wmt_mqm
    pickapic
    vl_rewardbench
    ai2d_test
    hallusionbench
    mathvista_mini
    mmbench_v11
    mme
    mmmu_dev_val
    financebench
    igakuqa
    lawbench
    tumlu
)

if [[ "${1:-}" == "--list" ]]; then
    echo "Available benchmarks:"
    for b in "${BENCHMARKS[@]}"; do echo "  $b"; done
    exit 0
fi

# If specific benchmarks requested, filter to those
if [[ $# -gt 0 && "$1" != "--list" ]]; then
    BENCHMARKS=("$@")
fi

SUCCEEDED=()
FAILED=()
SKIPPED=()

run_benchmark() {
    local name="$1"
    log_info "========== $name =========="

    case "$name" in

    # ─────────────────────────────────────────────────────────────────────
    # 1. Terminal-Bench 2.0 (multi-step pipeline)
    # ─────────────────────────────────────────────────────────────────────
    terminal_bench)
        DIR="$BASE_DIR/terminal_bench_data"
        mkdir -p "$DIR/raw" "$DIR/processed" "$DIR/scripts"

        # Step 1: Download task metadata from HuggingFace
        log_info "Downloading Terminal-Bench task metadata..."
        python3 "$DIR/scripts/01_download_task_metadata.py" || true

        # Step 5: Query Supabase database for trial-level data
        log_info "Querying Terminal-Bench database..."
        python3 "$DIR/scripts/05_query_database.py" || true

        # Step 6: Build response matrix
        log_info "Building Terminal-Bench response matrix..."
        python3 "$DIR/scripts/06_build_response_matrix.py"

        # Step 8: Merge metadata
        python3 "$DIR/scripts/08_merge_metadata.py" || true
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 2. BFCL v3 (Berkeley Function Calling Leaderboard)
    # ─────────────────────────────────────────────────────────────────────
    bfcl)
        DIR="$BASE_DIR/bfcl_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Clone the BFCL-Result repo (pre-computed scores for all models)
        if [[ ! -d "$DIR/raw/BFCL-Result" ]]; then
            log_info "Cloning BFCL-Result repo..."
            git clone https://github.com/HuanzhiMao/BFCL-Result.git "$DIR/raw/BFCL-Result"
        else
            log_info "BFCL-Result repo already exists, pulling latest..."
            git -C "$DIR/raw/BFCL-Result" pull || true
        fi

        log_info "Building BFCL response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 3. LiveCodeBench
    # ─────────────────────────────────────────────────────────────────────
    livecodebench)
        DIR="$BASE_DIR/livecodebench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Clone submissions repo (per-model eval_all.json files)
        if [[ ! -d "$DIR/raw/submissions" ]]; then
            log_info "Cloning LiveCodeBench submissions repo..."
            git clone https://github.com/LiveCodeBench/submissions.git "$DIR/raw/submissions"
        else
            log_info "LiveCodeBench submissions repo exists, pulling..."
            git -C "$DIR/raw/submissions" pull || true
        fi

        # Clone website repo (leaderboard JSON data)
        if [[ ! -d "$DIR/raw/livecodebench.github.io" ]]; then
            log_info "Cloning LiveCodeBench website repo..."
            git clone https://github.com/LiveCodeBench/livecodebench.github.io.git "$DIR/raw/livecodebench.github.io"
        else
            git -C "$DIR/raw/livecodebench.github.io" pull || true
        fi

        log_info "Building LiveCodeBench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 4. SWE-bench Verified
    # ─────────────────────────────────────────────────────────────────────
    swebench)
        DIR="$BASE_DIR/swebench_data"
        mkdir -p "$DIR/raw/results_json" "$DIR/processed"

        # Download all results.json files from SWE-bench/experiments repo via GitHub API
        log_info "Downloading SWE-bench submission results via GitHub API..."
        python3 -c "
import json, os, urllib.request, time
from pathlib import Path

raw_dir = Path('$DIR/raw/results_json')
api_url = 'https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified'
req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    resp = urllib.request.urlopen(req, timeout=30)
    submissions = json.loads(resp.read())
except Exception as e:
    print(f'Warning: GitHub API call failed: {e}')
    print('Using existing raw data if available.')
    submissions = []

downloaded = 0
for sub in submissions:
    if sub.get('type') != 'dir':
        continue
    name = sub['name']
    results_url = f'https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified/{name}/results/results.json'
    out_file = raw_dir / f'{name}.json'
    if out_file.exists():
        continue
    try:
        req2 = urllib.request.Request(results_url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req2, timeout=15).read()
        out_file.write_bytes(data)
        downloaded += 1
        if downloaded % 20 == 0:
            time.sleep(1)
    except Exception as e:
        print(f'  Skip {name}: {e}')

print(f'Downloaded {downloaded} new results files. Total: {len(list(raw_dir.glob(\"*.json\")))}')
"

        log_info "Building SWE-bench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 5. BigCodeBench
    # ─────────────────────────────────────────────────────────────────────
    bigcodebench)
        DIR="$BASE_DIR/bigcodebench_data"
        mkdir -p "$DIR/processed"

        # Script downloads directly from HuggingFace bigcode/bigcodebench-perf
        log_info "Building BigCodeBench response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 6. EvalPlus (HumanEval+ / MBPP+)
    # ─────────────────────────────────────────────────────────────────────
    evalplus)
        DIR="$BASE_DIR/evalplus_data"
        mkdir -p "$DIR/raw/humaneval_samples" "$DIR/raw/mbpp_samples" "$DIR/raw/eval_results" "$DIR/processed"

        # Download pre-generated code samples from EvalPlus GitHub releases
        log_info "Downloading EvalPlus code samples from GitHub releases..."
        python3 -c "
import urllib.request, json, os, zipfile, io
from pathlib import Path

raw = Path('$DIR/raw')
releases_url = 'https://api.github.com/repos/evalplus/evalplus/releases'
req = urllib.request.Request(releases_url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    releases = json.loads(urllib.request.urlopen(req, timeout=30).read())
    for rel in releases:
        for asset in rel.get('assets', []):
            name = asset['name']
            if not name.endswith('.zip'):
                continue
            dest = raw / 'humaneval_samples' if 'humaneval' in name.lower() else raw / 'mbpp_samples'
            dest_file = dest / name
            if dest_file.exists():
                continue
            print(f'Downloading {name}...')
            data = urllib.request.urlopen(asset['browser_download_url'], timeout=60).read()
            dest_file.write_bytes(data)
    print('EvalPlus samples downloaded.')
except Exception as e:
    print(f'Warning: {e}. Using existing data.')
"

        log_info "Building EvalPlus response matrices..."
        python3 "$DIR/scripts/01_build_response_matrix.py" --skip-eval || \
            python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 7. MMLU-Pro
    # ─────────────────────────────────────────────────────────────────────
    mmlupro)
        DIR="$BASE_DIR/mmlupro_data"
        mkdir -p "$DIR/raw/eval_results" "$DIR/processed"

        # Download per-question eval results from TIGER-AI-Lab GitHub
        log_info "Downloading MMLU-Pro eval results from GitHub..."
        python3 -c "
import json, os, urllib.request, zipfile, io, time
from pathlib import Path

raw_dir = Path('$DIR/raw')
eval_dir = raw_dir / 'eval_results'

# Get listing of eval_results zip files from GitHub API
api_url = 'https://api.github.com/repos/TIGER-AI-Lab/MMLU-Pro/contents/eval_results'
try:
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    listing = json.loads(urllib.request.urlopen(req, timeout=30).read())
    zips = [f for f in listing if f['name'].endswith('.zip')]
    print(f'Found {len(zips)} eval result zip files')
    for z in zips:
        dest = eval_dir / z['name']
        if dest.exists():
            continue
        print(f'  Downloading {z[\"name\"]}...')
        data = urllib.request.urlopen(z['download_url'], timeout=60).read()
        dest.write_bytes(data)
        time.sleep(0.5)
except Exception as e:
    print(f'Warning: {e}. Using existing data.')

# Download leaderboard CSV from HuggingFace
lb_path = raw_dir / 'leaderboard_results.csv'
if not lb_path.exists():
    try:
        url = 'https://huggingface.co/datasets/TIGER-Lab/mmlu_pro_leaderboard_submission/resolve/main/leaderboard_results.csv'
        urllib.request.urlretrieve(url, lb_path)
        print('Leaderboard CSV downloaded.')
    except Exception as e:
        print(f'Leaderboard download failed: {e}')
"

        log_info "Building MMLU-Pro response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 8. CRUXEval
    # ─────────────────────────────────────────────────────────────────────
    cruxeval)
        DIR="$BASE_DIR/cruxeval_data"
        CLONE_DIR="$DIR/raw/cruxeval"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Clone the Facebook Research CRUXEval repo (model generations + evaluations)
        if [[ ! -d "$CLONE_DIR" ]]; then
            log_info "Cloning CRUXEval repo..."
            git clone https://github.com/facebookresearch/cruxeval.git "$CLONE_DIR"
        else
            log_info "CRUXEval repo exists, pulling..."
            git -C "$CLONE_DIR" pull || true
        fi

        log_info "Building CRUXEval response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 9. WebArena
    # ─────────────────────────────────────────────────────────────────────
    webarena)
        DIR="$BASE_DIR/webarena_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Download execution traces from Google Drive (original paper models)
        log_info "Downloading WebArena traces..."
        # v1 traces (July 2023 release)
        if [[ ! -d "$DIR/raw/072023_release_v1" ]]; then
            log_info "  Downloading v1 traces from Google Drive..."
            python3 -c "
import gdown, os
os.makedirs('$DIR/raw/072023_release_v1', exist_ok=True)
# Google Drive folder ID for v1 traces
gdown.download_folder('https://drive.google.com/drive/folders/1tnFTtNBEfJmPQR0x8arHxINGR2bZFvVm', output='$DIR/raw/072023_release_v1/', quiet=False)
" 2>/dev/null || log_warn "  gdown not installed or Google Drive download failed. Install: pip install gdown"
        fi

        # v2 traces (Oct 2023 release)
        if [[ ! -d "$DIR/raw/102023_release_v2" ]]; then
            log_info "  Downloading v2 traces from Google Drive..."
            python3 -c "
import gdown, os
os.makedirs('$DIR/raw/102023_release_v2', exist_ok=True)
gdown.download_folder('https://drive.google.com/drive/folders/1eFAZkRp6jEbgqJ2LJmhayc1LLHvj1sXH', output='$DIR/raw/102023_release_v2/', quiet=False)
" 2>/dev/null || log_warn "  gdown not installed or Google Drive download failed."
        fi

        # Clone third-party agent result repos
        if [[ ! -d "$DIR/raw/agent-evals" ]]; then
            git clone https://github.com/anthropics/agent-evals.git "$DIR/raw/agent-evals" 2>/dev/null || true
        fi

        log_info "Building WebArena response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 10. DPAI Arena
    # ─────────────────────────────────────────────────────────────────────
    dpai)
        DIR="$BASE_DIR/dpai_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Download score reports from TeamCity REST API
        log_info "Downloading DPAI score reports from TeamCity API..."
        python3 -c "
import json, urllib.request, csv, os
from pathlib import Path

raw_dir = Path('$DIR/raw')
processed_dir = Path('$DIR/processed')

# Known build IDs and their agent names (from dpaia.dev leaderboard)
agents = {
    'junie_cli_claude_opus_4.5': '10697',
    'junie_cli_sonnet_4.5': '10696',
    'junie_cli_gemini_3.0': '10698',
    'claude_code_opus_4.5': '10699',
    'claude_code_sonnet_4.5_auto': '10700',
    'claude_code_sonnet_4.5_explicit': '10701',
    'codex_cli_gpt5_codex': '10702',
    'gemini_cli_gemini_3.0_preview': '10703',
    'gemini_cli_gemini_2.5_pro': '10704',
}

for agent_name, build_id in agents.items():
    out_file = raw_dir / f'{agent_name}_score_report.json'
    if out_file.exists():
        print(f'  {agent_name}: already downloaded')
        continue
    url = f'https://dpaia.teamcity.com/guestAuth/app/rest/builds/id:{build_id}/artifacts/content/aggregated_total_score_report.json'
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        data = urllib.request.urlopen(req, timeout=30).read()
        out_file.write_bytes(data)
        print(f'  {agent_name}: downloaded')
    except Exception as e:
        print(f'  {agent_name}: FAILED ({e})')

# Build response matrix from score reports
rows = {}
all_tasks = set()
for f in sorted(raw_dir.glob('*_score_report.json')):
    agent = f.stem.replace('_score_report', '')
    with open(f) as fh:
        data = json.load(fh)
    for task in data:
        tid = task.get('instance_id', task.get('task_id', ''))
        score = task.get('total_score', task.get('score', 0))
        all_tasks.add(tid)
        if tid not in rows:
            rows[tid] = {}
        rows[tid][agent] = score

agents_list = sorted({a for r in rows.values() for a in r})
tasks_sorted = sorted(rows.keys())

with open(processed_dir / 'response_matrix_total_score.csv', 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['instance_id'] + agents_list)
    for tid in tasks_sorted:
        row = [tid] + [rows[tid].get(a, '') for a in agents_list]
        writer.writerow(row)

print(f'DPAI matrix: {len(tasks_sorted)} tasks x {len(agents_list)} agents')
" || log_warn "DPAI download may require network access to dpaia.teamcity.com"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 11. AgentDojo
    # ─────────────────────────────────────────────────────────────────────
    agentdojo)
        DIR="$BASE_DIR/agentdojo_data"
        CLONE_DIR="/tmp/agentdojo_repo"
        mkdir -p "$DIR/processed"

        # Clone the AgentDojo repo (contains runs/ directory with per-task JSONs)
        if [[ ! -d "$CLONE_DIR/runs" ]]; then
            log_info "Cloning AgentDojo repo..."
            git clone https://github.com/ethz-spylab/agentdojo.git "$CLONE_DIR"
        else
            log_info "AgentDojo repo exists, pulling..."
            git -C "$CLONE_DIR" pull || true
        fi

        log_info "Building AgentDojo response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py" --runs-dir "$CLONE_DIR/runs"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 12. MLE-bench
    # ─────────────────────────────────────────────────────────────────────
    mlebench)
        DIR="$BASE_DIR/mlebench_data"
        CLONE_DIR="/tmp/mle-bench"
        mkdir -p "$DIR/processed"

        # Clone with Git LFS for the runs/ directory
        if [[ ! -d "$CLONE_DIR" ]]; then
            log_info "Cloning MLE-bench repo (with Git LFS)..."
            GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/openai/mle-bench.git "$CLONE_DIR"
            cd "$CLONE_DIR"
            git lfs pull --include="runs/**" || log_warn "Git LFS pull failed; data may be pointers"
            cd "$BASE_DIR"
        else
            log_info "MLE-bench repo exists."
        fi

        log_info "Building MLE-bench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 13. EDIT-Bench
    # ─────────────────────────────────────────────────────────────────────
    editbench)
        DIR="$BASE_DIR/editbench_data"
        CLONE_DIR="/tmp/editbench_repo"
        mkdir -p "$DIR/raw_results" "$DIR/processed"

        # Clone the editbench repo (contains results/whole_file/ JSON files)
        if [[ ! -d "$CLONE_DIR" ]]; then
            log_info "Cloning editbench repo..."
            git clone https://github.com/waynchi/editbench.git "$CLONE_DIR"
        else
            git -C "$CLONE_DIR" pull || true
        fi

        # Copy result JSONs to raw_results/
        cp "$CLONE_DIR"/results/whole_file/*.json "$DIR/raw_results/" 2>/dev/null || true

        log_info "Building EDIT-Bench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 14. TAU-bench
    # ─────────────────────────────────────────────────────────────────────
    taubench)
        DIR="$BASE_DIR/taubench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Clone tau-bench v1 for historical trajectories
        CLONE_V1="/tmp/tau-bench"
        if [[ ! -d "$CLONE_V1" ]]; then
            log_info "Cloning tau-bench v1..."
            git clone https://github.com/sierra-research/tau-bench.git "$CLONE_V1"
        fi

        # Clone tau2-bench v2
        CLONE_V2="/tmp/tau2-bench"
        if [[ ! -d "$CLONE_V2" ]]; then
            log_info "Cloning tau2-bench v2..."
            git clone https://github.com/sierra-research/tau2-bench.git "$CLONE_V2"
        fi

        # Copy trajectory files to raw/
        log_info "Copying trajectory data..."
        for f in "$CLONE_V1"/trajectories/*.json; do
            [[ -f "$f" ]] && cp "$f" "$DIR/raw/" 2>/dev/null || true
        done
        mkdir -p "$DIR/raw/tau2_results"
        for f in "$CLONE_V2"/results/*.json; do
            [[ -f "$f" ]] && cp "$f" "$DIR/raw/tau2_results/" 2>/dev/null || true
        done

        log_info "Building TAU-bench response matrices..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 15. Cybench
    # ─────────────────────────────────────────────────────────────────────
    cybench)
        DIR="$BASE_DIR/cybench_data"
        CLONE_DIR="/tmp/cybench_repo"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Clone repo for task metadata + leaderboard CSV
        if [[ ! -d "$CLONE_DIR" ]]; then
            log_info "Cloning Cybench repo..."
            git clone https://github.com/andyzorigin/cybench.git "$CLONE_DIR"
        else
            git -C "$CLONE_DIR" pull || true
        fi

        # Download leaderboard aggregate scores
        curl -sL "https://cybench.github.io/data/leaderboard.csv" -o "$DIR/raw/leaderboard.csv" 2>/dev/null || true

        # Download paper PDF for programmatic table extraction
        if [[ ! -f "$DIR/raw/cybench_paper.pdf" ]]; then
            log_info "Downloading Cybench paper PDF (arXiv:2408.08926)..."
            curl -sL "https://arxiv.org/pdf/2408.08926" -o "$DIR/raw/cybench_paper.pdf"
        fi

        log_info "Building Cybench response matrix (extracting Tables 10-12 from PDF)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 16. CORE-Bench
    # ─────────────────────────────────────────────────────────────────────
    corebench)
        DIR="$BASE_DIR/corebench_data"
        CLONE_DIR="/tmp/core-bench"
        mkdir -p "$DIR/processed"

        if [[ ! -d "$CLONE_DIR" ]]; then
            log_info "Cloning CORE-Bench repo..."
            git clone https://github.com/siegelz/core-bench.git "$CLONE_DIR"
        else
            git -C "$CLONE_DIR" pull || true
        fi

        log_info "Building CORE-Bench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 17. PaperBench
    # ─────────────────────────────────────────────────────────────────────
    paperbench)
        DIR="$BASE_DIR/paperbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Download paper PDF for programmatic table extraction
        if [[ ! -f "$DIR/raw/paperbench_paper.pdf" ]]; then
            log_info "Downloading PaperBench paper PDF (arXiv:2504.01848)..."
            curl -sL "https://arxiv.org/pdf/2504.01848" -o "$DIR/raw/paperbench_paper.pdf"
        fi

        log_info "Building PaperBench response matrix (extracting Tables 10-18 from PDF)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 18. Aider (aggregate only — per-exercise data not public)
    # ─────────────────────────────────────────────────────────────────────
    aider)
        DIR="$BASE_DIR/aider_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Download leaderboard YAML files from aider.chat website
        log_info "Downloading Aider leaderboard YAML files..."
        for yml in edit_leaderboard polyglot_leaderboard o1_polyglot_leaderboard refactor_leaderboard; do
            url="https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/${yml}.yml"
            out="$DIR/raw/${yml}.yml"
            if [[ ! -f "$out" ]]; then
                curl -sL "$url" -o "$out" 2>/dev/null || true
            fi
        done

        log_info "Building Aider response matrix (aggregate only)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 19. SWE-PolyBench (only 3 models — sparse)
    # ─────────────────────────────────────────────────────────────────────
    swepolybench)
        DIR="$BASE_DIR/swepolybench_data"
        CLONE_DIR="/tmp/swepolybench_repo"
        mkdir -p "$DIR/raw" "$DIR/processed"

        if [[ ! -d "$CLONE_DIR" ]]; then
            log_info "Cloning SWE-PolyBench repo (submission branch)..."
            git clone --branch submission https://github.com/amazon-science/SWE-PolyBench.git "$CLONE_DIR"
        fi

        log_info "Building SWE-PolyBench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 20. VisualWebArena (sparse — only 100/910 tasks from AgentRewardBench)
    # ─────────────────────────────────────────────────────────────────────
    visualwebarena)
        DIR="$BASE_DIR/visualwebarena_data"
        mkdir -p "$DIR/processed"

        # Script downloads from HuggingFace AgentRewardBench dataset
        log_info "Building VisualWebArena response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 21. AppWorld (aggregate only — per-task encrypted)
    # ─────────────────────────────────────────────────────────────────────
    appworld)
        DIR="$BASE_DIR/appworld_data"
        mkdir -p "$DIR/processed"

        # Script downloads leaderboard JSON from appworld.dev + HF dataset
        log_info "Building AppWorld response matrix (aggregate only)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 22. ScienceAgentBench (aggregate only — per-task not published)
    # ─────────────────────────────────────────────────────────────────────
    scienceagentbench)
        DIR="$BASE_DIR/scienceagentbench_data"
        mkdir -p "$DIR/processed"

        # Script has hardcoded aggregate data from paper + HAL leaderboard
        log_info "Building ScienceAgentBench response matrix (aggregate only)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 23. SWE-bench Full (2,294 instances)
    # ─────────────────────────────────────────────────────────────────────
    swebench_full)
        DIR="$BASE_DIR/swebench_full_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building SWE-bench Full response matrix (downloads via GitHub API)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 24. SWE-bench Java (170 instances)
    # ─────────────────────────────────────────────────────────────────────
    swebench_java)
        DIR="$BASE_DIR/swebench_java_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building SWE-bench Java response matrix (downloads via GitHub API)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 25. SWE-bench Multilingual (301 instances, 9 languages)
    # ─────────────────────────────────────────────────────────────────────
    swebench_multilingual)
        DIR="$BASE_DIR/swebench_multilingual_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building SWE-bench Multilingual response matrix (downloads via GitHub API)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 26. AgentBench (data embedded in script from paper Table 3)
    # ─────────────────────────────────────────────────────────────────────
    agentbench)
        DIR="$BASE_DIR/agentbench_data"
        mkdir -p "$DIR/processed"

        log_info "Building AgentBench response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 27. AlpacaEval 2.0 (downloads from GitHub)
    # ─────────────────────────────────────────────────────────────────────
    alpacaeval)
        DIR="$BASE_DIR/alpacaeval_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building AlpacaEval response matrix (downloads from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 28. AndroidWorld (downloads from Google Sheets + web)
    # ─────────────────────────────────────────────────────────────────────
    androidworld)
        DIR="$BASE_DIR/androidworld_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building AndroidWorld response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 29. ARC-AGI (downloads from HuggingFace + GitHub)
    # ─────────────────────────────────────────────────────────────────────
    arcagi)
        DIR="$BASE_DIR/arcagi_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        # Clone ground truth for scoring
        if [[ ! -d "$DIR/raw/arc-agi-1" ]]; then
            log_info "Cloning ARC-AGI ground truth..."
            git clone https://github.com/fchollet/ARC-AGI.git "$DIR/raw/arc-agi-1" 2>/dev/null || true
        fi

        log_info "Building ARC-AGI response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 30. BrowserGym (downloads from HuggingFace Space)
    # ─────────────────────────────────────────────────────────────────────
    browsergym)
        DIR="$BASE_DIR/browsergym_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building BrowserGym response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 31. GAIA (HuggingFace — gated dataset, requires approval)
    # ─────────────────────────────────────────────────────────────────────
    gaia)
        DIR="$BASE_DIR/gaia_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building GAIA response matrix (requires HuggingFace gated dataset access)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 32. Humanity's Last Exam (downloads from GitHub + Scale AI)
    # ─────────────────────────────────────────────────────────────────────
    hle)
        DIR="$BASE_DIR/hle_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building HLE response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 33. LiveBench (downloads from HuggingFace datasets)
    # ─────────────────────────────────────────────────────────────────────
    livebench)
        DIR="$BASE_DIR/livebench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building LiveBench response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 34. MathArena (downloads from HuggingFace)
    # ─────────────────────────────────────────────────────────────────────
    matharena)
        DIR="$BASE_DIR/matharena_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building MathArena response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 35. OSWorld (downloads trajectories from HuggingFace)
    # ─────────────────────────────────────────────────────────────────────
    osworld)
        DIR="$BASE_DIR/osworld_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building OSWorld response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 36. TheAgentCompany (clones experiments repo)
    # ─────────────────────────────────────────────────────────────────────
    theagentcompany)
        DIR="$BASE_DIR/theagentcompany_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building TheAgentCompany response matrix (clones experiments repo)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 37. StableToolBench (downloads from HuggingFace)
    # ─────────────────────────────────────────────────────────────────────
    toolbench)
        DIR="$BASE_DIR/toolbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building ToolBench response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 38. WildBench (downloads from GitHub)
    # ─────────────────────────────────────────────────────────────────────
    wildbench)
        DIR="$BASE_DIR/wildbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building WildBench response matrix (downloads from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 39. WorkArena (downloads from HuggingFace + BrowserGym)
    # ─────────────────────────────────────────────────────────────────────
    workarena)
        DIR="$BASE_DIR/workarena_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building WorkArena response matrix..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 40. Arena 140K (pairwise human preferences from Chatbot Arena)
    # ─────────────────────────────────────────────────────────────────────
    arena_140k)
        DIR="$BASE_DIR/arena_140k_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Arena 140K comparison summary (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 41. Arena-Hard-Auto (GPT-4 judgments, 72 models x 500 prompts)
    # ─────────────────────────────────────────────────────────────────────
    arena_hard)
        DIR="$BASE_DIR/arena_hard_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Arena-Hard-Auto response matrix (clones HuggingFace Space)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 42. MT-Bench (GPT-4 single-answer judgments, 34 models x 80 questions)
    # ─────────────────────────────────────────────────────────────────────
    mtbench)
        DIR="$BASE_DIR/mtbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building MT-Bench response matrix (downloads JSONL from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 43. UltraFeedback (GPT-4 multi-aspect ratings, 17 models x 64K prompts)
    # ─────────────────────────────────────────────────────────────────────
    ultrafeedback)
        DIR="$BASE_DIR/ultrafeedback_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building UltraFeedback response matrix (streams from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 44. Nectar (GPT-4 rankings, 40 models x 183K prompts)
    # ─────────────────────────────────────────────────────────────────────
    nectar)
        DIR="$BASE_DIR/nectar_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Nectar response matrix (streams from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 45. BiGGen-Bench (multi-judge, 99 models x 695 items x 5 judges)
    # ─────────────────────────────────────────────────────────────────────
    biggen)
        DIR="$BASE_DIR/biggen_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building BiGGen-Bench response matrices (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 46. Preference Dissection (gated, 33 judges x 5,240 pairs)
    # ─────────────────────────────────────────────────────────────────────
    preference_dissection)
        DIR="$BASE_DIR/preference_dissection_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Preference Dissection response matrix (requires HF_TOKEN with gated access)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 47. RewardBench (151 judges x 2,985 items)
    # ─────────────────────────────────────────────────────────────────────
    rewardbench)
        DIR="$BASE_DIR/rewardbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building RewardBench response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 48. LLM Judge Indeterminacy (9 judges x 800 items)
    # ─────────────────────────────────────────────────────────────────────
    indeterminacy)
        DIR="$BASE_DIR/indeterminacy_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Indeterminacy response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 49. HH-RLHF (169K pairwise preferences)
    # ─────────────────────────────────────────────────────────────────────
    hh_rlhf)
        DIR="$BASE_DIR/hh_rlhf_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building HH-RLHF comparison data (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 50. OpenAssistant Conversations (16 rank tiers x 18,922 prompts)
    # ─────────────────────────────────────────────────────────────────────
    oasst)
        DIR="$BASE_DIR/oasst_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building OpenAssistant response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 51. HelpSteer2 (2 responses x 10,679 prompts x 6 attributes)
    # ─────────────────────────────────────────────────────────────────────
    helpsteer2)
        DIR="$BASE_DIR/helpsteer2_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building HelpSteer2 response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 52. SHP-2 (100K sampled pairwise preferences across 124 domains)
    # ─────────────────────────────────────────────────────────────────────
    shp2)
        DIR="$BASE_DIR/shp2_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building SHP-2 comparison data (streams from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 53. RewardBench 2 (188 judges x 1,865 items, 6 domains)
    # ─────────────────────────────────────────────────────────────────────
    rewardbench2)
        DIR="$BASE_DIR/rewardbench2_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building RewardBench 2 response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 54. SummEval (16 models x 100 docs x 4 quality dimensions)
    # ─────────────────────────────────────────────────────────────────────
    summeval)
        DIR="$BASE_DIR/summeval_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building SummEval response matrices (downloads from GCS)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 55. FLASK (15 models x 1,700 instructions x 12 skills)
    # ─────────────────────────────────────────────────────────────────────
    flask)
        DIR="$BASE_DIR/flask_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building FLASK response matrices (downloads from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 56. Prometheus (rubric-based multi-judge evaluation)
    # ─────────────────────────────────────────────────────────────────────
    prometheus)
        DIR="$BASE_DIR/prometheus_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Prometheus response matrices (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 57. JudgeBench (33 judges x 350 pairs)
    # ─────────────────────────────────────────────────────────────────────
    judgebench)
        DIR="$BASE_DIR/judgebench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building JudgeBench response matrix (downloads from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 58. PRM800K (step-level math verification, 75K solutions)
    # ─────────────────────────────────────────────────────────────────────
    prm800k)
        DIR="$BASE_DIR/prm800k_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building PRM800K response matrices (clones from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 59. BeaverTails (safety annotations, 330K+ QA pairs, 14 harm categories)
    # ─────────────────────────────────────────────────────────────────────
    beavertails)
        DIR="$BASE_DIR/beavertails_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building BeaverTails response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 60. PKU-SafeRLHF (dual safety+helpfulness preference, 330K+ pairs)
    # ─────────────────────────────────────────────────────────────────────
    pku_saferlhf)
        DIR="$BASE_DIR/pku_saferlhf_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building PKU-SafeRLHF comparison data (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 61. PRISM (personalized preferences, 1,500 participants x 68K utterances)
    # ─────────────────────────────────────────────────────────────────────
    prism)
        DIR="$BASE_DIR/prism_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building PRISM response matrices (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 62. PersonalLLM (10 reward models x 83K prompt-response pairs)
    # ─────────────────────────────────────────────────────────────────────
    personalllm)
        DIR="$BASE_DIR/personalllm_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building PersonalLLM response matrices (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 63. WMT MQM (expert MT evaluation, multi-rater segment scores)
    # ─────────────────────────────────────────────────────────────────────
    wmt_mqm)
        DIR="$BASE_DIR/wmt_mqm_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building WMT MQM response matrices (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 64. Pick-a-Pic (text-to-image preference, 500K+ judgments)
    # ─────────────────────────────────────────────────────────────────────
    pickapic)
        DIR="$BASE_DIR/pickapic_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building Pick-a-Pic comparison data (streams from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # 65. VL-RewardBench (vision-language reward model evaluation)
    # ─────────────────────────────────────────────────────────────────────
    vl_rewardbench)
        DIR="$BASE_DIR/vl_rewardbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building VL-RewardBench response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # VLM Benchmarks (from VLMEval/OpenVLMRecords on HuggingFace)
    # ─────────────────────────────────────────────────────────────────────
    ai2d_test)
        DIR="$BASE_DIR/ai2d_test_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building AI2D_TEST response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    hallusionbench)
        DIR="$BASE_DIR/hallusionbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building HallusionBench response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    mathvista_mini)
        DIR="$BASE_DIR/mathvista_mini_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building MathVista_MINI response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    mmbench_v11)
        DIR="$BASE_DIR/mmbench_v11_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building MMBench_V11 response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    mme)
        DIR="$BASE_DIR/mme_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building MME response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    mmmu_dev_val)
        DIR="$BASE_DIR/mmmu_dev_val_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building MMMU_DEV_VAL response matrix (downloads from HuggingFace)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    # ─────────────────────────────────────────────────────────────────────
    # Ported benchmarks (from textbook curation scripts)
    # ─────────────────────────────────────────────────────────────────────
    financebench)
        DIR="$BASE_DIR/financebench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building FinanceBench response matrix (clones from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    igakuqa)
        DIR="$BASE_DIR/igakuqa_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building IgakuQA response matrix (clones from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    lawbench)
        DIR="$BASE_DIR/lawbench_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building LawBench response matrix (clones from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    tumlu)
        DIR="$BASE_DIR/tumlu_data"
        mkdir -p "$DIR/raw" "$DIR/processed"

        log_info "Building TUMLU response matrix (clones from GitHub)..."
        python3 "$DIR/scripts/01_build_response_matrix.py"
        ;;

    *)
        log_error "Unknown benchmark: $name"
        SKIPPED+=("$name")
        return 1
        ;;
    esac

    # ── Generate visualizations (applies to all benchmarks) ──────────
    local viz_script="$BASE_DIR/${name}_data/scripts/02_visualize_response_matrix.py"
    if [[ -f "$viz_script" ]]; then
        log_info "Generating $name visualizations..."
        python3 "$viz_script" || log_warn "$name visualization failed"
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────

log_info "Starting reproduction of ${#BENCHMARKS[@]} benchmarks..."
log_info "Base directory: $BASE_DIR"
echo

for bench in "${BENCHMARKS[@]}"; do
    if run_benchmark "$bench"; then
        SUCCEEDED+=("$bench")
        log_info "$bench: SUCCESS"
    else
        FAILED+=("$bench")
        log_error "$bench: FAILED"
    fi
    echo
done

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────

echo
log_info "=============================="
log_info "  Reproduction Summary"
log_info "=============================="
echo -e "${GREEN}Succeeded (${#SUCCEEDED[@]}):${NC} ${SUCCEEDED[*]:-none}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}Failed (${#FAILED[@]}):${NC} ${FAILED[*]}"
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Skipped (${#SKIPPED[@]}):${NC} ${SKIPPED[*]}"
fi
echo
log_info "Output matrices are in: <benchmark>_data/processed/response_matrix.csv"
log_info "Visualizations are in:  <benchmark>_data/figures/"
