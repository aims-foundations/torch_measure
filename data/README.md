# Benchmark Data Collection

Curated response matrices from 40+ AI evaluation benchmarks, standardized as
(subjects x items) matrices for IRT analysis with `torch_measure`.

## Quick Start

```bash
# Reproduce all benchmarks from original sources
python reproduce.py

# Reproduce a single benchmark
python reproduce.py bfcl

# List available benchmarks
python reproduce.py --list
```

## Prerequisites

```bash
pip install pandas numpy requests pyyaml beautifulsoup4 lxml matplotlib seaborn
pip install pdfplumber   # for Cybench and PaperBench (PDF table extraction)
pip install gdown        # for WebArena (Google Drive traces)
pip install psycopg2     # for Terminal-Bench (Supabase database queries)
pip install datasets     # for HuggingFace-hosted benchmarks
pip install evalplus     # for EvalPlus (local code evaluation)
```

## Directory Structure

Each benchmark follows a consistent layout:

```
<benchmark>/
  build.py        # Downloads raw data and builds response matrix
  raw/            # Original data (cloned repos, API downloads, PDFs)
  processed/      # Analysis-ready outputs
    response_matrix.csv    # Primary output: subjects x items
    item_content.csv       # Item text content (item_id, content)
    model_summary.csv      # Per-subject aggregate statistics
    task_metadata.csv      # Item metadata (difficulty, category, etc.)
  figures/        # Visualizations (heatmaps, bar charts)
```

## Data Sources

| Benchmark | Source Type | Raw Data Origin |
|-----------|-----------|-----------------|
| agentbench | GitHub API | [THUDM/AgentBench](https://github.com/THUDM/AgentBench) |
| agentdojo | Git clone | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) (runs/) |
| aider | HTTP download | [Aider-AI/aider](https://github.com/Aider-AI/aider) leaderboard YAML |
| alpacaeval | HuggingFace | [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) |
| androidworld | HuggingFace | [google-research/android_world](https://github.com/google-research/android_world) |
| appworld | HTTP API | [appworld.dev](https://appworld.dev/) leaderboard JSON |
| arcagi | HuggingFace | [arcprize](https://huggingface.co/arcprize) per-model eval datasets |
| bfcl | Git clone | [HuanzhiMao/BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) |
| bigcodebench | HuggingFace | [bigcode/bigcodebench-perf](https://huggingface.co/datasets/bigcode/bigcodebench-perf) |
| browsergym | HuggingFace | [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym) |
| clinebench | GitHub API | [cline/cline](https://github.com/cline/cline) bench results |
| corebench | Git clone | [siegelz/core-bench](https://github.com/siegelz/core-bench) |
| cruxeval | Git clone | [facebookresearch/cruxeval](https://github.com/facebookresearch/cruxeval) |
| cybench | arXiv PDF | [arXiv:2408.08926](https://arxiv.org/abs/2408.08926) Tables 10-12 |
| dpai | TeamCity API | [dpaia.teamcity.com](https://dpaia.teamcity.com) guest REST API |
| editbench | Git clone | [waynchi/editbench](https://github.com/waynchi/editbench) |
| evalplus | GitHub releases | [evalplus/evalplus](https://github.com/evalplus/evalplus) ZIP samples |
| gaia | HuggingFace | [gaia-benchmark](https://huggingface.co/gaia-benchmark) |
| hle | HuggingFace | [lastexam.ai](https://lastexam.ai/) |
| livebench | Git clone | [livebench submissions](https://github.com/LiveBench) |
| livecodebench | Git clone | [LiveCodeBench/submissions](https://github.com/LiveCodeBench/submissions) |
| matharena | Git clone | [matharena.ai](https://matharena.ai/) GitHub data |
| mlebench | Git clone + LFS | [openai/mle-bench](https://github.com/openai/mle-bench) |
| mmlupro | GitHub + HuggingFace | [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) eval_results/ |
| osworld | Git clone | [os-world.github.io](https://os-world.github.io/) |
| paperbench | arXiv PDF | [arXiv:2504.01848](https://arxiv.org/abs/2504.01848) Tables 10-18 |
| scienceagentbench | Embedded data | Aggregated from paper + HAL leaderboard |
| swebench | GitHub API | [SWE-bench/experiments](https://github.com/SWE-bench/experiments) |
| swebench_full | GitHub API | [SWE-bench/experiments](https://github.com/SWE-bench/experiments) |
| swebench_java | GitHub API | [multi-swe-bench](https://github.com/multi-swe-bench) |
| swebench_multilingual | GitHub API | [multi-swe-bench](https://github.com/multi-swe-bench) |
| swepolybench | Git clone | [amazon-science/SWE-PolyBench](https://github.com/amazon-science/SWE-PolyBench) |
| taubench | Git clone | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) |
| terminal_bench | Supabase DB | [tbench.ai](https://www.tbench.ai/) public database |
| theagentcompany | HuggingFace | [TheAgentCompany/experiments](https://github.com/TheAgentCompany/experiments) |
| toolbench | HuggingFace | [THUNLP-MT/StableToolBench](https://github.com/THUNLP-MT/StableToolBench) |
| visualwebarena | HuggingFace | AgentRewardBench dataset |
| webarena | Google Drive + Git | [webarena.dev](https://webarena.dev/) traces |
| wildbench | GitHub | [allenai/WildBench](https://github.com/allenai/WildBench) |
| workarena | Git clone | [ServiceNow/WorkArena](https://github.com/ServiceNow/WorkArena) |

## Access Notes

Most benchmarks have fully public per-task data. Exceptions:
- **AppWorld**: Only aggregate leaderboard scores; per-task results are encrypted in `.bundle` files
- **Aider**: Only aggregate leaderboard; per-exercise pass/fail is not public
- **ScienceAgentBench**: Aggregate from paper; per-task results not separately published
- **VisualWebArena**: Sparse coverage (100/910 tasks from AgentRewardBench)
- **GAIA**: HuggingFace dataset is gated (requires manual approval)
- **Terminal-Bench**: Queries a live Supabase database (requires network access)

## Registered Datasets

After processing, data is uploaded to HuggingFace Hub (`aims-foundation/torch-measure-data`)
and registered in `src/torch_measure/datasets/bench.py`. Load with:

```python
from torch_measure.datasets import load, list_datasets

list_datasets()                 # See all available
rm = load("swebench")           # Download and load as ResponseMatrix
print(rm.data.shape)            # torch.Size([134, 500])
```

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `agentbench` | 29x8 | continuous | AgentBench — multi-environment agent evaluation |
| `agentdojo` | 29x132 | binary | AgentDojo — tool-use agent utility tasks |
| `agentdojo_security` | 28x949 | binary | AgentDojo — security evaluation |
| `agentdojo_utility_attack` | 28x97 | binary | AgentDojo — utility under attack |
| `alpacaeval` | 221x805 | binary | AlpacaEval — instruction following |
| `androidworld` | 3x116 | binary | AndroidWorld — mobile device automation |
| `appworld` | 18x24 | continuous | AppWorld — multi-app interaction |
| `arcagi` | 52x400 | binary | ARC-AGI v1 — abstract reasoning |
| `arcagi_v2` | 28x120 | binary | ARC-AGI v2 — abstract reasoning |
| `bfcl` | 93x4751 | binary | BFCL v3 — function calling |
| `bigcodebench` | 153x1140 | binary | BigCodeBench — code generation |
| `bigcodebench_hard_complete` | 199x148 | binary | BigCodeBench — hard tasks |
| `bigcodebench_hard_instruct` | 173x148 | binary | BigCodeBench — hard instruct |
| `bigcodebench_instruct` | 126x1140 | binary | BigCodeBench — instruct |
| `browsergym` | 18x8 | continuous | BrowserGym — web agent aggregates |
| `clinebench` | 3x12 | continuous | ClineBench — AI coding agent |
| `corebench` | 15x270 | binary | CORE-Bench — reproducibility |
| `corebench_scores` | 15x270 | continuous | CORE-Bench — scores |
| `cruxeval` | 38x800 | continuous | CRUXEval — code reasoning |
| `cruxeval_binary` | 38x800 | binary | CRUXEval — code reasoning (binary) |
| `cruxeval_input` | 20x800 | continuous | CRUXEval — input prediction |
| `cruxeval_input_binary` | 20x800 | binary | CRUXEval — input prediction (binary) |
| `cruxeval_output` | 18x800 | continuous | CRUXEval — output prediction |
| `cruxeval_output_binary` | 18x800 | binary | CRUXEval — output prediction (binary) |
| `cybench` | 8x40 | binary | Cybench — CTF tasks |
| `cybench_guided` | 8x40 | binary | Cybench — guided CTF tasks |
| `cybench_scores` | 8x40 | continuous | Cybench — subtask scores |
| `dpai` | 9x141 | continuous | DPAI Arena — total score |
| `dpai_binary` | 9x141 | binary | DPAI Arena — binary |
| `dpai_blind` | 9x141 | continuous | DPAI Arena — blind evaluation |
| `dpai_informed` | 9x141 | continuous | DPAI Arena — informed evaluation |
| `editbench` | 44x540 | continuous | EDIT-Bench — code editing |
| `editbench_binary` | 44x540 | binary | EDIT-Bench — code editing (binary) |
| `evalplus` | 31x164 | binary | EvalPlus — HumanEval+ and MBPP+ |
| `evalplus_humaneval_base` | 31x164 | binary | EvalPlus — HumanEval base |
| `evalplus_humaneval_plus` | 31x164 | binary | EvalPlus — HumanEval+ |
| `evalplus_mbpp_base` | 22x378 | binary | EvalPlus — MBPP base |
| `evalplus_mbpp_plus` | 22x378 | binary | EvalPlus — MBPP+ |
| `gaia` | 32x165 | binary | GAIA — general AI assistant |
| `hle` | 19x1792 | binary | Humanity's Last Exam |
| `livebench` | 195x494 | continuous | LiveBench — contamination-free |
| `livebench_binary` | 195x494 | binary | LiveBench — binary |
| `livecodebench` | 72x1055 | continuous | LiveCodeBench — competitive programming |
| `matharena` | 68x336 | continuous | MathArena — competition math |
| `matharena_aime2025` | 62x30 | binary | MathArena — AIME 2025 |
| `mlebench` | 30x75 | continuous | MLE-bench — ML engineering |
| `mlebench_above_median` | 30x75 | binary | MLE-bench — above-median |
| `mlebench_binary` | 30x75 | binary | MLE-bench — binary |
| `mlebench_scores` | 30x75 | continuous | MLE-bench — raw Kaggle scores |
| `mmlupro` | 48x12257 | continuous | MMLU-Pro — per-question |
| `mmlupro_category` | 247x14 | continuous | MMLU-Pro — per-category |
| `osworld` | 77x369 | continuous | OSWorld — desktop automation |
| `osworld_binary` | 77x369 | binary | OSWorld — binary |
| `paperbench` | 9x20 | continuous | PaperBench — paper reproduction |
| `paperbench_runs` | 7x180 | continuous | PaperBench — per-run |
| `swebench` | 134x500 | binary | SWE-bench Verified |
| `swebench_full` | 24x2294 | binary | SWE-bench Full |
| `swebench_java` | 52x170 | binary | SWE-bench Java |
| `swebench_multi_all` | 79x2132 | binary | SWE-bench Multi — all languages |
| `swebench_multilingual` | 13x301 | binary | SWE-bench Multilingual |
| `swepolybench_full` | 1x2110 | binary | SWE-PolyBench — full |
| `swepolybench_verified` | 3x382 | binary | SWE-PolyBench — verified |
| `taubench` | 32x329 | continuous | TAU-bench — all domains |
| `taubench_hal_airline` | 26x50 | continuous | TAU-bench — HAL airline |
| `taubench_retail` | 6x115 | continuous | TAU-bench — retail |
| `taubench_telecom` | 4x114 | continuous | TAU-bench — telecom |
| `taubench_v1_airline` | 28x50 | continuous | TAU-bench — airline v1 |
| `taubench_v2_airline` | 4x50 | continuous | TAU-bench — airline v2 |
| `terminalbench` | 128x89 | binary | Terminal-Bench — CLI tasks |
| `terminalbench_resolution` | 128x89 | continuous | Terminal-Bench — resolution rate |
| `theagentcompany` | 19x175 | continuous | TheAgentCompany — enterprise |
| `theagentcompany_binary` | 19x175 | binary | TheAgentCompany — binary |
| `toolbench` | 10x765 | binary | StableToolBench — tool-use |
| `visualwebarena` | 6x910 | continuous | VisualWebArena — multimodal web |
| `webarena` | 14x812 | continuous | WebArena — web agent tasks |
| `wildbench` | 63x1024 | continuous | WildBench — open-ended LLM eval |
| `workarena` | 4x118 | continuous | WorkArena — enterprise web tasks |
