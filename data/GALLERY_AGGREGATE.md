# Benchmark Gallery — Aggregate-Only Benchmarks

Heatmaps for benchmarks in `BENCHMARKS_AGGREGATE`. These have multi-model data but the cells are aggregate rates across trials, conditions, or sub-benchmarks — **not** per-item responses. Useful for model-level comparisons but not IRT-ready. See [`GALLERY.md`](GALLERY.md) for per-item benchmarks.

**13 benchmarks**, **16 heatmaps**.

Each image shows the full response matrix: rows are subjects (usually models, sorted by mean score), columns are items (sorted by difficulty), colored by score (red=low, green=high). Matrices larger than 1000 × 2000 are downsampled for render performance.

To regenerate the heatmaps, run the relevant `build.py` or `python data/scripts/visualize_response_matrix.py`. To regenerate this file, run `python data/scripts/build_gallery.py`.

---

### agent_safetybench

<img src="agent_safetybench/processed/response_matrix.png" alt="agent_safetybench" width="360">

### agentbench

<img src="agentbench/processed/response_matrix.png" alt="agentbench" width="360">

<img src="agentbench/processed/response_matrix_with_overall.png" alt="agentbench / with_overall" width="360">
_agentbench / with_overall_

### agentharm

<img src="agentharm/processed/response_matrix.png" alt="agentharm" width="360">

### agentic_misalignment

<img src="agentic_misalignment/processed/response_matrix.png" alt="agentic_misalignment" width="360">

### ai_safety_index

<img src="ai_safety_index/processed/response_matrix.png" alt="ai_safety_index" width="360">

<img src="ai_safety_index/processed/response_matrix_normalized.png" alt="ai_safety_index / normalized" width="360">
_ai_safety_index / normalized_

### aider

<img src="aider/processed/response_matrix.png" alt="aider" width="360">

### browsergym

<img src="browsergym/processed/response_matrix.png" alt="browsergym" width="360">

<img src="browsergym/processed/response_matrix_with_stderr.png" alt="browsergym / with_stderr" width="360">
_browsergym / with_stderr_

### ca_dmv_disengagement

<img src="ca_dmv_disengagement/processed/response_matrix_binary.png" alt="ca_dmv_disengagement / binary" width="360">
_ca_dmv_disengagement / binary_

### ko_leaderboard

<img src="ko_leaderboard/processed/response_matrix.png" alt="ko_leaderboard" width="360">

### la_leaderboard

<img src="la_leaderboard/processed/response_matrix.png" alt="la_leaderboard" width="360">

### nhtsa_sgo

<img src="nhtsa_sgo/processed/response_matrix_binary.png" alt="nhtsa_sgo / binary" width="360">
_nhtsa_sgo / binary_

### pt_leaderboard

<img src="pt_leaderboard/processed/response_matrix.png" alt="pt_leaderboard" width="360">

### thai_leaderboard

<img src="thai_leaderboard/processed/response_matrix.png" alt="thai_leaderboard" width="360">
