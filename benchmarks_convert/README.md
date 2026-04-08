# Benchmark Data Pipeline

Converts wide-format response matrices into a unified long-format dataset
and a SQLite database used at runtime for hidden evaluation.

## Folder Structure

```
benchmarks/
├── _source/            # Raw data: response matrices + metadata CSVs
│   └── {name}/         #   response_matrix.csv, item_content.csv, etc.
├── _compiled/          # Processed: items.csv, ground_truth.csv, embeddings.npy
│   └── {name}/
├── _original_tests/    # Archived original development + train data
├── pt/                 # Long-format .pt files (per-benchmark + merged)
├── benchmarks.sqlite   # Runtime DB (items, pairs, embeddings) — deployed to GCS
├── benchmark_responses.sqlite  # Analytics DB (flat responses table)
├── benchmark_registry.yaml     # Benchmark configuration
└── *.py                # Pipeline scripts
```

## Active Benchmarks (8)

| Benchmark | Items | Models | Domain |
|---|---:|---:|---|
| bbq | 58,492 | 7 | Bias in QA |
| mmlupro | 12,257 | 48 | General knowledge (14 categories) |
| lawbench | 9,000 | 51 | Chinese legal tasks |
| swebench_full | 2,294 | 24 | Software engineering |
| bigcodebench | 1,140 | 153 | Code generation |
| mathvista_mini | 1,000 | 101 | Math visual QA |
| swebench | 500 | 134 | Software engineering (verified) |
| jailbreakbench | 100 | 18 | Jailbreak attacks |

**Total: 84,783 items, 519 models, 1,832,070 pairs.**

## Pipeline

### Step 1: Build .pt files from response matrices

```bash
python benchmarks/01_build_pt.py                 # all
python benchmarks/01_build_pt.py swebench bbq    # specific
```

Reads `benchmark_registry.yaml`, melts wide CSV to long format, joins metadata
for item_text / category / test_condition. Saves to `benchmarks/pt/{name}.pt`.

### Step 2: Merge .pt files (optional, for analytics)

```bash
python benchmarks/02_merge_pt.py
```

### Step 3: Generate CSV folders + embeddings

```bash
python benchmarks/03_compile_folders.py                    # all
python benchmarks/03_compile_folders.py swebench           # single (only embeds new data)
python benchmarks/03_compile_folders.py --no-embeddings    # skip embeddings
```

Outputs to `benchmarks/_compiled/{name}/`: `items.csv`, `ground_truth.csv`, `embeddings.npy`.
Embeddings are per-benchmark — adding a new benchmark only embeds its items.

### Step 4: Compile benchmarks.sqlite

```bash
python scripts/compile_benchmark_db.py
```

Reads all folders in `_compiled/`, merges into `benchmarks/benchmarks.sqlite` with:

- `items` table: item_idx, source_item_id, benchmark_name, item_text, category, embedding (BLOB)
- `pairs` table: source_item_id, benchmark_name, model_id, label
- `metadata` table: manifest JSON

### Step 5: Upload + deploy

```bash
python scripts/upload_snapshot.py              # upload to GCS
bash deploy/03-build-push-images.sh            # rebuild Docker images
bash deploy/05-deploy-orchestrator.sh          # restart orchestrator
```

## Adding a New Benchmark

1. Add `response_matrix.csv` + metadata to `benchmarks/_source/{name}/`
2. Add entry in `benchmark_registry.yaml`
3. Run steps 1, 3, 4 above (step 3 only embeds the new benchmark)
4. Upload + deploy

## Known Issues: Model Naming

Model names are not normalized across benchmarks. The same model may appear as
`GPT-4o`, `GPT4o`, `gpt-4o-2024-11-20`, etc. Some benchmarks encode test
conditions in the model name (e.g., jailbreakbench: `gpt-3.5-turbo-1106_GCG`).

## TODO: Training Set

The codabench training set is 100% mmlupro (2,000 items, 48 models). Participants
have no training signal for the other 7 benchmarks. A multi-benchmark training
set needs to be generated.
