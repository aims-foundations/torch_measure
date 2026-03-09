#!/usr/bin/env python3
"""
Build a PaperBench response matrix by extracting Tables 10-18 from the paper PDF.

Data source: "PaperBench: Evaluating AI's Ability to Replicate AI Research"
  - arXiv: https://arxiv.org/abs/2504.01848v3
  - GitHub: https://github.com/openai/preparedness/tree/main/project/paperbench
  - Tables 10-18 contain per-paper per-run scores for 9 model-agent configurations.
  - Table 7 contains rubric node counts per paper.
  - Table 9 contains per-requirement-type scores.

Tables 10-18 are extracted programmatically from the PDF using pdfplumber.
Tables 7 and 9 contain aggregate metadata that is hardcoded (not per-task response data).

The benchmark evaluates 20 ICML 2024 papers with 8,316 individually gradable subtask
nodes across hierarchical rubrics. Scores are weighted averages of leaf-node binary
pass/fail grades, reported as fractions in [0, 1].

Output:
  - processed/response_matrix.csv         -- papers x models (mean scores)
  - processed/response_matrix_runs.csv    -- papers x models x runs (individual run scores)
  - processed/task_metadata.csv           -- per-paper rubric metadata
  - processed/overall_scores.csv          -- per-model overall and per-type scores

Requirements:
  pip install pdfplumber
"""

import os
import csv
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber is required. Install with: pip install pdfplumber")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARXIV_ID = "2504.01848"
PDF_URL = f"https://arxiv.org/pdf/{ARXIV_ID}"
PDF_PATH = os.path.join(RAW_DIR, "paperbench_paper.pdf")

# =============================================================================
# Paper and model configuration constants
# =============================================================================

PAPERS = [
    "adaptive-pruning",
    "all-in-one",
    "bam",
    "bbox",
    "bridging-data-gaps",
    "fre",
    "ftrl",
    "lbcs",
    "lca-on-the-line",
    "mechanistic-understanding",
    "pinn",
    "rice",
    "robust-clip",
    "sample-specific-masks",
    "sapg",
    "sequential-neural-score-estimation",
    "stay-on-topic-with-classifier-free-guidance",
    "stochastic-interpolants",
    "test-time-model-adaptation",
    "what-will-my-model-forget",
]

# Model configurations: (table_number, model_name, scaffold, notes)
MODEL_CONFIGS = [
    ("Table10", "gpt-4o", "BasicAgent", "gpt-4o-2024-08-06"),
    ("Table11", "o1-high", "BasicAgent", "o1-2024-12-17, reasoning=high"),
    ("Table12", "o1-high", "IterativeAgent", "o1-2024-12-17, reasoning=high"),
    ("Table13", "o3-mini-high", "BasicAgent", "o3-mini-2025-01-31, reasoning=high"),
    ("Table14", "o3-mini-high", "IterativeAgent", "o3-mini-2025-01-31, reasoning=high"),
    ("Table15", "claude-3-5-sonnet", "BasicAgent", "claude-3-5-sonnet-20241022"),
    ("Table16", "claude-3-5-sonnet", "IterativeAgent", "claude-3-5-sonnet-20241022"),
    ("Table17", "gemini-2.0-flash", "BasicAgent", "gemini-2.0-flash"),
    ("Table18", "deepseek-r1", "BasicAgent", "DeepSeek-R1 via OpenRouter"),
]

def model_key(cfg):
    return f"{cfg[1]} ({cfg[2]})"

MODEL_KEYS = [model_key(c) for c in MODEL_CONFIGS]

NaN = float("nan")

# Mapping from abbreviated PDF model names to canonical model IDs.
# Tables 4/5 use full names (e.g., "O1-HIGH"), Table 9 uses short names (e.g., "O1").
_PDF_MODEL_MAP = {
    "GPT-4O": "gpt-4o",
    "4O": "gpt-4o",
    "O1-HIGH": "o1-high",
    "O1": "o1-high",
    "O3-MINI-HIGH": "o3-mini-high",
    "O3-MINI": "o3-mini-high",
    "DEEPSEEK-R1": "deepseek-r1",
    "R1": "deepseek-r1",
    "CLAUDE-3.5-SONNET": "claude-3-5-sonnet",
    "CLAUDE-3-5-SONNET": "claude-3-5-sonnet",
    "GEMINI-2.0-FLASH": "gemini-2.0-flash",
}
_PDF_SCAFFOLD_MAP = {
    "BASICAGENT": "BasicAgent",
    "ITERATIVEAGENT": "IterativeAgent",
}


# =============================================================================
# PDF downloading and table extraction
# =============================================================================

def download_pdf():
    """Download paper PDF from arXiv if not already present."""
    if os.path.exists(PDF_PATH):
        print(f"  PDF already exists: {PDF_PATH}")
        return
    print(f"  Downloading {PDF_URL} ...")
    req = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    with open(PDF_PATH, "wb") as f:
        f.write(data)
    print(f"  Saved {len(data)} bytes to {PDF_PATH}")


def extract_tables_from_pdf(pdf_path):
    """Extract all data tables from the PaperBench paper PDF.

    Extracts:
      - Tables 10-18: Per-paper per-run scores for 9 model-agent configs
      - Table 7: Rubric node counts per paper
      - Tables 4/5: Overall replication scores (BasicAgent / IterativeAgent)
      - Table 9: Per-requirement-type scores

    Returns:
        (run_tables, rubric_metadata, overall_scores) where:
        - run_tables: dict {table_num: {paper_name: [run1, run2, run3]}}
        - rubric_metadata: dict {paper_name: {total_nodes, leaf_nodes, ...}}
        - overall_scores: dict {canonical_key: {overall, overall_se, code_dev, ...}}
    """
    pdf = pdfplumber.open(pdf_path)

    # --- Tables 10-18: per-paper per-run scores ---
    run_tables = {}
    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text()
        if not text:
            continue
        lines = text.split('\n')
        current_table = None
        for line in lines:
            line_nospace = line.replace(' ', '')
            table_match = re.match(r'^Table(\d+)[\.\s]', line_nospace)
            if table_match:
                tnum = int(table_match.group(1))
                if 10 <= tnum <= 18:
                    current_table = tnum
                    run_tables[current_table] = {}
                continue
            if current_table and 'PAPER' in line and 'RUN1' in line:
                continue
            if not current_table:
                continue
            m = re.match(
                r'([A-Z][A-Z0-9\-]+(?:-[A-Z0-9\-]+)*)\s+(.*)', line
            )
            if m:
                paper_name = m.group(1).lower()
                tokens = m.group(2).strip().split()
                if len(tokens) >= 3:
                    runs = []
                    for t in tokens[:3]:
                        t_clean = t.rstrip('*')
                        try:
                            runs.append(float(t_clean))
                        except ValueError:
                            runs.append(NaN)
                    run_tables[current_table][paper_name] = runs

    # Handle infrastructure failures (Table 17, what-will-my-model-forget Run 3)
    if 17 in run_tables and "what-will-my-model-forget" in run_tables[17]:
        runs_17 = run_tables[17]["what-will-my-model-forget"]
        while len(runs_17) < 3:
            runs_17.append(NaN)

    # Validate Tables 10-18
    for tnum in range(10, 19):
        if tnum not in run_tables:
            raise RuntimeError(f"Table {tnum} not found in PDF")
        missing = [p for p in PAPERS if p not in run_tables[tnum]]
        if missing:
            raise RuntimeError(
                f"Table {tnum}: Missing papers: {missing}. "
                "PDF extraction may have failed."
            )
        for paper in PAPERS:
            if len(run_tables[tnum][paper]) != 3:
                raise RuntimeError(
                    f"Table {tnum}, {paper}: Expected 3 runs, got "
                    f"{len(run_tables[tnum][paper])}"
                )

    # --- Table 7: Rubric node counts ---
    rubric_metadata = {}
    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text() or ''
        if 'Table7' not in text.replace(' ', ''):
            continue
        for line in text.split('\n'):
            m = re.match(
                r'([a-z][a-z0-9\-]+(?:-[a-z0-9\-]+)*)\s+'
                r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
                line
            )
            if m:
                rubric_metadata[m.group(1)] = {
                    "total_nodes": int(m.group(2)),
                    "leaf_nodes": int(m.group(3)),
                    "code_dev": int(m.group(4)),
                    "execution": int(m.group(5)),
                    "result_match": int(m.group(6)),
                }
        if len(rubric_metadata) == 20:
            break

    missing_rubric = [p for p in PAPERS if p not in rubric_metadata]
    if missing_rubric:
        raise RuntimeError(
            f"Table 7: Missing rubric data for: {missing_rubric}"
        )

    # --- Tables 4/5: Overall replication scores ---
    # These are on a two-column page; find boundaries via table headers.
    overall_basic = {}  # BasicAgent (Table 4)
    overall_iter = {}   # IterativeAgent (Table 5)
    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text() or ''
        nospace = text.replace(' ', '')
        if 'Table4.' not in nospace and 'Table5.' not in nospace:
            continue
        lines = text.split('\n')
        # Find line indices for table boundaries
        starts = {}
        for i, line in enumerate(lines):
            ln = line.replace(' ', '')
            for tnum in (4, 5, 6):
                if re.search(rf'Table{tnum}[\.\s]', ln) and tnum not in starts:
                    starts[tnum] = i
        t4_start = starts.get(4, 0)
        t5_start = starts.get(5, len(lines))
        t6_start = starts.get(6, len(lines))

        def _extract_scores(lines_slice):
            scores = {}
            for line in lines_slice:
                for model_raw, score, se in re.findall(
                    r'([A-Z][A-Z0-9\.\-]+)\s+([\d\.]+)\xb1([\d\.]+)', line
                ):
                    if model_raw in _PDF_MODEL_MAP and model_raw not in scores:
                        scores[model_raw] = (float(score), float(se))
            return scores

        overall_basic = _extract_scores(lines[t4_start:t5_start])
        overall_iter = _extract_scores(lines[t5_start:t6_start])
        break

    # --- Table 9: Per-requirement-type scores ---
    per_type = {}
    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text() or ''
        if 'Table9' not in text.replace(' ', ''):
            continue
        for line in text.split('\n'):
            if '\xb1' not in line or '(' not in line:
                continue
            m = re.search(
                r'([A-Z0-9\.\-]+)\s*\(([A-Z]+AGENT)\)\s+'
                r'([\d\.]+)\s*\xb1\s*([\d\.]+)\s+'
                r'([\d\.]+)\s*\xb1\s*([\d\.]+)\s+'
                r'([\d\.]+)\s*\xb1\s*([\d\.]+)',
                line
            )
            if m:
                model_raw = m.group(1)
                scaffold_raw = m.group(2)
                per_type[f"{model_raw}({scaffold_raw})"] = {
                    "code_dev": float(m.group(3)),
                    "code_dev_se": float(m.group(4)),
                    "execution": float(m.group(5)),
                    "execution_se": float(m.group(6)),
                    "result_analysis": float(m.group(7)),
                    "result_analysis_se": float(m.group(8)),
                }
        break

    pdf.close()

    # --- Assemble OVERALL_SCORES from Tables 4/5/9 ---
    overall_scores = {}
    for cfg in MODEL_CONFIGS:
        canonical = model_key(cfg)
        model_id = cfg[1]
        scaffold = cfg[2]

        entry = {}
        # Overall score from Table 4 or 5
        pdf_name_upper = None
        for pdf_k, canon_v in _PDF_MODEL_MAP.items():
            if canon_v == model_id:
                src = overall_basic if scaffold == "BasicAgent" else overall_iter
                if pdf_k in src:
                    pdf_name_upper = pdf_k
                    entry["overall"], entry["overall_se"] = src[pdf_k]
                    break

        # Per-type scores from Table 9
        for pdf_k, canon_v in _PDF_MODEL_MAP.items():
            if canon_v == model_id:
                scaffold_key = "BASICAGENT" if scaffold == "BasicAgent" else "ITERATIVEAGENT"
                t9_key = f"{pdf_k}({scaffold_key})"
                if t9_key in per_type:
                    entry.update(per_type[t9_key])
                    break

        if entry:
            overall_scores[canonical] = entry

    return run_tables, rubric_metadata, overall_scores


# =============================================================================
# Utility functions
# =============================================================================

def nanmean(values):
    """Compute mean ignoring NaN values."""
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return NaN
    return sum(valid) / len(valid)


def nanstderr(values):
    """Compute standard error of the mean ignoring NaN values."""
    valid = [v for v in values if not math.isnan(v)]
    n = len(valid)
    if n <= 1:
        return NaN
    mean = sum(valid) / n
    variance = sum((v - mean) ** 2 for v in valid) / (n - 1)
    return math.sqrt(variance / n)


# =============================================================================
# Build outputs
# =============================================================================

def build_response_matrix(output_dir, all_tables):
    """Build and save the response matrix (papers x models, mean scores)."""
    out_path = os.path.join(output_dir, "response_matrix.csv")

    header = ["paper"] + MODEL_KEYS
    rows = []
    for paper in PAPERS:
        row = [paper]
        for tnum in range(10, 19):
            runs = all_tables[tnum][paper]
            mean = nanmean(runs)
            row.append(f"{mean:.4f}" if not math.isnan(mean) else "")
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote response_matrix.csv: {len(rows)} papers x {len(MODEL_KEYS)} models")
    return rows


def build_response_matrix_runs(output_dir, all_tables):
    """Build per-run response matrix (papers x models x runs)."""
    out_path = os.path.join(output_dir, "response_matrix_runs.csv")

    header = ["paper", "model", "scaffold", "run_1", "run_2", "run_3", "mean", "std_error"]
    rows = []
    for paper in PAPERS:
        for cfg_idx, cfg in enumerate(MODEL_CONFIGS):
            tnum = 10 + cfg_idx
            runs = all_tables[tnum][paper]
            mean = nanmean(runs)
            se = nanstderr(runs)
            row = [paper, cfg[1], cfg[2]]
            for r in runs:
                row.append(f"{r:.4f}" if not math.isnan(r) else "")
            row.append(f"{mean:.4f}" if not math.isnan(mean) else "")
            row.append(f"{se:.4f}" if not math.isnan(se) else "")
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote response_matrix_runs.csv: {len(rows)} rows "
          f"({len(PAPERS)} papers x {len(MODEL_CONFIGS)} models)")
    return rows


def build_task_metadata(output_dir, rubric_metadata):
    """Build task metadata CSV from rubric structure."""
    out_path = os.path.join(output_dir, "task_metadata.csv")

    header = [
        "paper",
        "total_nodes",
        "leaf_nodes",
        "code_dev_leaves",
        "execution_leaves",
        "result_match_leaves",
    ]
    rows = []
    total_nodes_sum = 0
    total_leaf_sum = 0
    for paper in PAPERS:
        meta = rubric_metadata[paper]
        total_nodes_sum += meta["total_nodes"]
        total_leaf_sum += meta["leaf_nodes"]
        rows.append([
            paper,
            meta["total_nodes"],
            meta["leaf_nodes"],
            meta["code_dev"],
            meta["execution"],
            meta["result_match"],
        ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote task_metadata.csv: {len(rows)} papers")
    print(f"  Total rubric nodes across all papers: {total_nodes_sum}")
    print(f"  Total leaf nodes across all papers: {total_leaf_sum}")
    return total_nodes_sum, total_leaf_sum


def build_overall_scores(output_dir, overall_scores):
    """Build overall scores CSV with per-requirement-type breakdowns."""
    out_path = os.path.join(output_dir, "overall_scores.csv")

    header = [
        "model",
        "overall_pct", "overall_se",
        "code_dev_pct", "code_dev_se",
        "execution_pct", "execution_se",
        "result_analysis_pct", "result_analysis_se",
    ]
    rows = []
    for model_key_name in MODEL_KEYS:
        scores = overall_scores.get(model_key_name, {})
        if not scores:
            continue
        rows.append([
            model_key_name,
            scores.get("overall", ""), scores.get("overall_se", ""),
            scores.get("code_dev", ""), scores.get("code_dev_se", ""),
            scores.get("execution", ""), scores.get("execution_se", ""),
            scores.get("result_analysis", ""), scores.get("result_analysis_se", ""),
        ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote overall_scores.csv: {len(rows)} models")


def print_summary_stats(all_tables, rubric_metadata, overall_scores):
    """Print summary statistics for the response matrix."""
    print("\n" + "=" * 80)
    print("PAPERBENCH RESPONSE MATRIX - SUMMARY REPORT")
    print("=" * 80)

    print(f"\nBenchmark: PaperBench (arXiv:{ARXIV_ID})")
    print(f"Source: OpenAI, April 2025")
    print(f"Data extraction: Programmatic (pdfplumber from arXiv PDF)")
    print(f"Paper count: {len(PAPERS)}")
    print(f"Model-scaffold configurations: {len(MODEL_CONFIGS)}")

    total_nodes = sum(m["total_nodes"] for m in rubric_metadata.values())
    total_leaves = sum(m["leaf_nodes"] for m in rubric_metadata.values())

    print(f"\nRubric structure:")
    print(f"  Total nodes (all papers):       {total_nodes}")
    print(f"  Total leaf nodes (gradable):    {total_leaves}")

    n_papers = len(PAPERS)
    n_models = len(MODEL_CONFIGS)
    print(f"\nResponse matrix dimensions:")
    print(f"  Papers (rows):                  {n_papers}")
    print(f"  Models (columns):               {n_models}")
    print(f"  Matrix shape:                   {n_papers} x {n_models}")

    total_cells_runs = 0
    filled_cells_runs = 0
    for tnum in range(10, 19):
        for paper in PAPERS:
            for val in all_tables[tnum][paper]:
                total_cells_runs += 1
                if not math.isnan(val):
                    filled_cells_runs += 1

    total_cells_mean = n_papers * n_models
    filled_cells_mean = 0
    for paper in PAPERS:
        for tnum in range(10, 19):
            mean = nanmean(all_tables[tnum][paper])
            if not math.isnan(mean):
                filled_cells_mean += 1

    print(f"\nFill rate:")
    print(f"  Mean matrix: {filled_cells_mean}/{total_cells_mean} "
          f"= {100.0 * filled_cells_mean / total_cells_mean:.1f}%")
    print(f"  Run-level:   {filled_cells_runs}/{total_cells_runs} "
          f"= {100.0 * filled_cells_runs / total_cells_runs:.1f}%")

    print(f"\nScore type: Continuous [0, 1]")
    print(f"  Weighted average of binary leaf-node pass/fail grades")

    print(f"\nModels evaluated:")
    for cfg in MODEL_CONFIGS:
        print(f"  {model_key(cfg):45s}  ({cfg[3]})")

    print(f"\nPer-model overall scores (from paper Tables 4 & 5):")
    for mk in MODEL_KEYS:
        scores = overall_scores.get(mk, {})
        if "overall" in scores:
            print(f"  {mk:45s}  {scores['overall']:5.1f}% "
                  f"+/- {scores['overall_se']:.1f}%")

    print(f"\nPer-paper difficulty (mean score across all 9 model configs):")
    paper_means = []
    for paper in PAPERS:
        all_means = []
        for tnum in range(10, 19):
            m = nanmean(all_tables[tnum][paper])
            if not math.isnan(m):
                all_means.append(m)
        overall = nanmean(all_means) if all_means else NaN
        paper_means.append((paper, overall))
    paper_means.sort(key=lambda x: x[1] if not math.isnan(x[1]) else -1,
                     reverse=True)
    for paper, pm in paper_means:
        meta = rubric_metadata[paper]
        print(f"  {paper:50s}  mean={pm:.3f}  "
              f"(nodes={meta['total_nodes']}, leaves={meta['leaf_nodes']})")


def main():
    print("Building PaperBench response matrix...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Step 1: Download PDF
    print("Step 1: Download paper PDF")
    download_pdf()

    # Step 2: Extract tables from PDF
    print("\nStep 2: Extract all tables from PDF")
    all_tables, rubric_metadata, overall_scores = extract_tables_from_pdf(PDF_PATH)
    for tnum in range(10, 19):
        cfg = MODEL_CONFIGS[tnum - 10]
        print(f"  Table {tnum} ({model_key(cfg)}): "
              f"{len(all_tables[tnum])} papers extracted")
    print(f"  Table 7 (rubric metadata): {len(rubric_metadata)} papers")
    print(f"  Tables 4/5/9 (overall scores): {len(overall_scores)} models")

    # Step 3: Build outputs
    print("\nStep 3: Build response matrices")
    build_response_matrix(OUTPUT_DIR, all_tables)
    build_response_matrix_runs(OUTPUT_DIR, all_tables)
    build_task_metadata(OUTPUT_DIR, rubric_metadata)
    build_overall_scores(OUTPUT_DIR, overall_scores)
    print_summary_stats(all_tables, rubric_metadata, overall_scores)

    # Also save raw data as JSON for programmatic access
    raw_data = {
        "papers": PAPERS,
        "model_configs": [
            {"table": c[0], "model": c[1], "scaffold": c[2], "notes": c[3]}
            for c in MODEL_CONFIGS
        ],
        "per_paper_runs": {},
        "rubric_metadata": rubric_metadata,
        "overall_scores": overall_scores,
    }
    for paper in PAPERS:
        raw_data["per_paper_runs"][paper] = {}
        for cfg_idx, cfg in enumerate(MODEL_CONFIGS):
            tnum = 10 + cfg_idx
            runs = all_tables[tnum][paper]
            raw_data["per_paper_runs"][paper][model_key(cfg)] = {
                "runs": [r if not math.isnan(r) else None for r in runs],
                "mean": nanmean(runs) if not math.isnan(nanmean(runs)) else None,
                "std_error": nanstderr(runs) if not math.isnan(nanstderr(runs)) else None,
            }

    json_path = os.path.join(OUTPUT_DIR, "paperbench_raw.json")
    with open(json_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nWrote paperbench_raw.json")


if __name__ == "__main__":
    main()
