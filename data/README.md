# Benchmark Data Collection

Curated response matrices from 146 AI evaluation benchmarks, standardized
as `(subjects × items)` matrices for IRT / psychometric analysis with
`torch_measure`. Each benchmark has a single self-contained `build.py`
that downloads raw data, builds the response matrix, generates a heatmap,
converts to `.pt`, and uploads to HuggingFace Hub.

- **98 ready benchmarks** (`BENCHMARKS`) — real per-(model, item) response matrices
- **13 aggregate-only benchmarks** (`BENCHMARKS_AGGREGATE`) — multi-model data but
  at the level of conditions/categories, not individual items
- **35 pending benchmarks** (`BENCHMARKS_PENDING`) — questions/catalogs with no
  multi-model evaluation data yet

## Statistics

Across the **98 ready benchmarks** (as of the latest run):

| Metric | Count |
|--------|-------|
| Unique items (largest variant per benchmark) | **1,037,045** |
| Total items (summed across all variants) | 1,278,791 |
| Total cells (subject × item values) | 250,392,642 |
| Response matrices (including variants) | 320 |
| Binary matrices | 171 |
| Continuous matrices | 145 |

**Top 10 benchmarks by item count:**

| Benchmark | Items |
|-----------|-------|
| `nectar` | 182,954 |
| `pku_saferlhf` | 164,236 |
| `shp2` | 100,000 |
| `personalllm` | 83,216 |
| `prism` | 68,371 |
| `ultrafeedback` | 63,966 |
| `bbq` | 58,492 |
| `pickapic` | 53,901 |
| `kmmlu` | 35,030 |
| `helm_afr` | 33,880 |

Regenerate these numbers after rebuilding any benchmark:

```bash
python data/scripts/dataset_stats.py              # all three lists
python data/scripts/dataset_stats.py --full       # every benchmark
python data/scripts/dataset_stats.py --list BENCHMARKS --top 20
```

## Quick Start

```bash
# Install dependencies (see requirements.txt)
pip install -r data/requirements.txt

# Run all ready benchmarks (download → build → visualize → upload to HF)
python data/reproduce.py

# Run a specific benchmark
python data/reproduce.py bfcl

# Run several benchmarks
python data/reproduce.py bfcl swebench cruxeval

# List all available benchmarks, grouped by list
python data/reproduce.py --list

# Run the aggregate-only benchmarks instead
python data/reproduce.py --aggregate

# Run the pending benchmarks (mostly for development/debugging)
python data/reproduce.py --pending

# Skip uploading to HuggingFace Hub (build locally only)
python data/reproduce.py --no-upload
```

## Running a Single Benchmark Directly

Each `build.py` is fully self-contained and can be run on its own without
going through `reproduce.py`:

```bash
# Full pipeline: download → build → visualize → upload to HF
python data/bfcl/build.py

# Skip the HF upload step (just builds local CSVs and heatmap)
NO_UPLOAD=1 python data/bfcl/build.py
```

`reproduce.py` is just a loop over benchmarks that forwards the
`NO_UPLOAD` env var to each build. It exists for reproducibility/sanity
runs; it's not required for the pipeline to function.

## Directory Structure

Each benchmark follows a consistent, flat layout:

```
<benchmark>/
  build.py                              # self-contained pipeline
  raw/                                  # original downloaded data
  processed/
    response_matrix.csv                 # primary output: subjects × items
    response_matrix.png                 # heatmap visualization
    response_matrix_<variant>.csv       # optional: extra variants
    response_matrix_<variant>.png       # optional: extra heatmaps
    item_content.csv                    # (item_id, content) — text of each item
    model_summary.csv                   # per-subject aggregate statistics
    task_metadata.csv                   # per-item metadata (category, difficulty, etc.)
    <benchmark>.pt                      # serialized torch payload (uploaded to HF)
```

Shared utilities live in `data/scripts/`:
- `visualize_response_matrix.py` — generates heatmap PNGs for each `response_matrix*.csv`
- `upload_to_hf.py` — converts CSVs to `.pt` and uploads to HuggingFace

## Gallery

A preview of all heatmaps is browsable in [`GALLERY.md`](GALLERY.md). This
is auto-generated from the tracked `response_matrix*.png` files, which
live next to each benchmark's CSV in `<benchmark>/processed/`.

To regenerate the heatmaps without rebuilding the raw data:

```bash
# Regenerate all heatmaps
python data/scripts/visualize_response_matrix.py

# Regenerate one benchmark
python data/scripts/visualize_response_matrix.py bfcl
```

## Registered Datasets

After processing, each `.pt` file is uploaded to HuggingFace Hub
(`aims-foundation/torch-measure-data`) at the repo root (flat structure,
no subdirectories). Load any benchmark in Python with:

```python
from torch_measure.datasets import load, list_datasets

list_datasets()                 # see all available
rm = load("swebench")           # downloads and loads as ResponseMatrix
print(rm.data.shape)            # torch.Size([134, 500])
print(rm.subject_ids[:5])       # model names
print(rm.item_ids[:5])          # item IDs
print(rm.item_contents[:1])     # actual question/task text
```

The `.pt` payload is a single dict with:

- `data` — `torch.Tensor` of shape `(n_subjects, n_items)`, float32
- `subject_ids` — list of subject identifiers (usually model names)
- `item_ids` — list of item identifiers
- `item_contents` — list of item text (aligned with `item_ids`)
- `subject_metadata` — optional dict of per-subject metadata (from `model_summary.csv`)

## Prerequisites

See [`requirements.txt`](requirements.txt) for the full list. Core deps:

```bash
pip install -r data/requirements.txt
```

The pipeline assumes `git` and `git-lfs` are available on the system for
benchmarks that clone source repos (most of them). HuggingFace uploads
require `HF_TOKEN` to be set in the environment or `huggingface-cli login`
to have been run.

## The Three Benchmark Lists

### BENCHMARKS (98)

These produce proper `(subjects × items)` matrices where each cell is a
single subject's response to a single item. Some benchmarks produce
multiple matrices (e.g. binary and continuous variants, or per-subset
splits), totaling ~320 response matrices.

### BENCHMARKS_AGGREGATE (13)

These have multi-model data but the cells are aggregate rates across
trials, conditions, or sub-benchmarks — not per-item responses. They're
useful for model-level comparisons but don't support IRT analysis.

| Benchmark | Shape | Why aggregate |
|-----------|-------|---------------|
| `agent_safetybench` | 16 × 18 | models × categories (from paper tables) |
| `agentharm` | 15 × 9 | models × (attack × metric) conditions |
| `agentic_misalignment` | 18 × 18 | models × scenario conditions |
| `aider` | 178 × 6 | models × Aider sub-benchmarks |
| `agentbench` | 29 × 8 | models × environment types |
| `browsergym` | 18 × 8 | models × sub-benchmarks |
| `ko_leaderboard` | 1159 × 9 | models × Korean benchmarks |
| `la_leaderboard` | 69 × 70 | models × Iberian benchmarks |
| `pt_leaderboard` | 1148 × 10 | models × Portuguese benchmarks |
| `thai_leaderboard` | 72 × 19 | models × Thai benchmarks |
| `ai_safety_index` | 8 × 6 | companies × policy domains (governance) |
| `ca_dmv_disengagement` | 16 × 7 | manufacturers × location types |
| `nhtsa_sgo` | 27 × 17 | manufacturers × vehicle types |

### BENCHMARKS_PENDING (35)

Questions-only datasets, AI governance catalogs, conversation logs, or
benchmarks whose per-item model predictions aren't publicly released.
Most have a `build.py` that downloads the raw items and produces
`task_metadata.csv` + `item_content.csv`, but no response matrix yet.

| Category | Benchmarks |
|----------|------------|
| No public per-item predictions | `ceval`, `cmmlu`, `fineval` (OpenCompass data is gated) |
| Preference data without model IDs | `hh_rlhf` |
| Medical QA (questions only) | `cmb`, `cmexam`, `frenchmedmcqa`, `medarabiq`, `medexpqa`, `medqa_chinese`, `mmedbench`, `permedcqa` |
| Safety / red teaming (no multi-model eval) | `apollo_deception`, `cot_safety_behaviors`, `cot_unfaithfulness`, `gandalf`, `lmsys_toxicchat`, `reward_hacks`, `safeagentbench`, `sycophancy_subterfuge`, `tensortrust`, `atbench`, `bells`, `odcv_bench`, `scale_mrt`, `trail` |
| AI governance / incident catalogs | `aiid`, `mit_airisk`, `oecd_aim`, `responsible_ai_measures`, `alignment_faking` |
| Conversation logs | `wildchat` |
| Multilingual (questions only) | `agreval`, `asiaeval`, `iberbench` |

## Access Notes

Most benchmarks have fully public data. Exceptions:

- **GAIA**: HuggingFace dataset is gated (requires manual approval)
- **OpenCompass**: `compass_academic_predictions` is gated — unlocking it
  would let `ceval`, `cmmlu`, etc. become full BENCHMARKS
- **Terminal-Bench**: Queries a live Supabase database (requires network)
- **WebArena**: Downloads execution traces from Google Drive via `gdown`
- **MLE-bench**: Uses Git LFS for its `runs/` directory; the download
  function auto-fixes a known upstream merge conflict in one LFS pointer

## Adding a New Benchmark

1. Create `data/<name>/build.py` following the self-contained pattern:
   - Use `_BENCHMARK_DIR = Path(__file__).resolve().parent`
   - Put raw data under `_BENCHMARK_DIR / "raw"`
   - Put outputs under `_BENCHMARK_DIR / "processed"`
   - At the end of `main()`, append the shared upload block (see any
     existing `build.py` for the boilerplate — it calls
     `data/scripts/visualize_response_matrix.py` and `upload_to_hf.py`)
2. Add the name to `BENCHMARKS` (or `BENCHMARKS_AGGREGATE` /
   `BENCHMARKS_PENDING`) in `data/reproduce.py`
3. Test: `NO_UPLOAD=1 python data/<name>/build.py`
4. Verify the matrix shape and that `processed/response_matrix.png` is
   produced.
