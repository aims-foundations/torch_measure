#!/usr/bin/env python3
"""
Build response matrix for Cybench (cybersecurity CTF agent benchmark).

Data sources:
  - Paper: "Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks
    of Language Models" (ICLR 2025). arXiv:2408.08926
    Tables 10-12: Per-task results for 8 models (single attempt, structured bash agent).
    Extracted programmatically using pdfplumber.
  - Leaderboard CSV from https://cybench.github.io/data/leaderboard.csv
    Aggregate results for newer models (no per-task breakdown available).
  - GitHub repo: https://github.com/andyzorigin/cybench
    task_list.txt and challenge metadata.

Output files:
  - processed/response_matrix.csv
      Rows = 40 CTF tasks, Columns = models
      Values: 1 (solved), 0 (failed) for unguided mode (single attempt)
  - processed/response_matrix_subtask_guided.csv
      Same format but for subtask-guided mode (single attempt)
  - processed/response_matrix_subtask_scores.csv
      Fractional subtask scores (e.g. "2/5") for single attempt
  - processed/task_metadata.csv
      Task metadata: name, path, FST, category, competition, difficulty
  - processed/leaderboard_aggregate.csv
      Aggregate scores for all models (including newer ones without per-task data)

Requirements:
  pip install pdfplumber
"""

import csv
import os
import re
import sys
import urllib.request

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

ARXIV_ID = "2408.08926"
PDF_URL = f"https://arxiv.org/pdf/{ARXIV_ID}"
PDF_PATH = os.path.join(RAW_DIR, "cybench_paper.pdf")

# ===========================================================================
# Task list in exact order from the paper (Tables 10-12)
# Format: (task_name, fst_str, category, competition, task_path)
# The task names here are the canonical names with spaces.
# ===========================================================================
TASKS = [
    ("Loot Stash",                    "00:02", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Very Easy] LootStash"),
    ("Urgent",                        "00:04", "Forensics",  "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Very Easy] Urgent"),
    ("Flag Command",                  "00:06", "Web",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/web/[Very Easy] Flag Command"),
    ("Packed Away",                   "00:06", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Very Easy] PackedAway"),
    ("It Has Begun",                  "00:07", "Forensics",  "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Very Easy] It Has Begun"),
    ("Dynastic",                      "00:09", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic"),
    ("Primary Knowledge",             "00:11", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Primary Knowledge"),
    ("Crushing",                      "00:20", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Easy] Crushing"),
    ("Permuted",                      "00:42", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Hard] Permuted"),
    ("Labyrinth Linguist",            "00:43", "Web",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/web/[Easy] Labyrinth Linguist"),
    ("Partial Tenacity",              "01:05", "Crypto",     "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Medium] Partial Tenacity"),
    ("Delulu",                        "01:15", "Pwn",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/pwn/[Very Easy] Delulu"),
    ("Data Siege",                    "01:18", "Forensics",  "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Medium] Data Siege"),
    ("Were Pickle Phreaks Revenge",   "02:00", "Misc",       "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/misc/[Medium] Were Pickle Phreaks Revenge"),
    ("Unbreakable",                   "02:03", "Misc",       "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/misc/[Easy] Unbreakable"),
    ("Lock Talk",                     "02:12", "Web",        "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/web/[Medium] LockTalk"),
    ("Flecks of Gold",                "06:08", "Reverse",    "HTB", "benchmark/hackthebox/cyber-apocalypse-2024/reversing/[Hard] FlecksOfGold"),
    ("Skilift",                       "00:05", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/intro/skilift"),
    ("Glacier Exchange",              "00:09", "Web",        "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/web/GlacierExchange"),
    ("Missing Bits",                  "00:09", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/MissingBits"),
    ("SLCG",                          "00:29", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/SLCG"),
    ("RPGO",                          "00:45", "Reverse",    "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/rev/rpgo"),
    ("SOP",                           "00:51", "Reverse",    "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/rev/sop"),
    ("Avatar",                        "01:24", "Misc",       "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/misc/avatar"),
    ("Walking to the Seaside",        "02:13", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/WalkingToTheSeaSide"),
    ("Shuffled AES",                  "02:39", "Crypto",     "GLA", "benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/shuffled-aes"),
    ("Eval Me",                       "00:11", "Forensics",  "S23", "benchmark/project-sekai-ctf/sekaictf-2023/forensics/eval-me"),
    ("Noisy CRC",                     "00:35", "Crypto",     "S23", "benchmark/project-sekai-ctf/sekaictf-2023/crypto/noisy-crc"),
    ("Network Tools",                 "00:42", "Pwn",        "S23", "benchmark/project-sekai-ctf/sekaictf-2023/pwn/network-tools"),
    ("Noisier CRC",                   "01:25", "Crypto",     "S23", "benchmark/project-sekai-ctf/sekaictf-2023/crypto/noisier-crc"),
    ("Chunky",                        "01:45", "Web",        "S23", "benchmark/project-sekai-ctf/sekaictf-2023/web/chunky"),
    ("Just Another Pickle Jail",      "04:04", "Misc",       "S23", "benchmark/project-sekai-ctf/sekaictf-2023/misc/just-another-pickle-jail"),
    ("Frog WAF",                      "05:30", "Web",        "S23", "benchmark/project-sekai-ctf/sekaictf-2023/web/frog-waf"),
    ("Randsubware",                   "05:56", "Crypto",     "S23", "benchmark/project-sekai-ctf/sekaictf-2023/crypto/randsubware"),
    ("Failproof",                     "01:00", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/failproof"),
    ("Ezmaze",                        "03:24", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/ezmaze"),
    ("Diffecient",                    "07:34", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/diffecient"),
    ("Robust CBC",                    "24:54", "Crypto",     "S22", "benchmark/project-sekai-ctf/sekaictf-2022/crypto/robust-cbc"),
    ("Back to the Past",              "00:32", "Web",        "HKC", "benchmark/hkcert-ctf/CTF-Challenges/CTF-2022/22-back-to-the-past"),
    ("MOTP",                          "00:52", "Web",        "HKC", "benchmark/hkcert-ctf/CTF-Challenges/CTF-2022/05-motp"),
]

TASK_NAMES = [t[0] for t in TASKS]

# Model names (8 models from the paper, structured bash, single attempt)
MODELS_PAPER = [
    "GPT-4o",
    "OpenAI o1-preview",
    "Claude 3 Opus",
    "Claude 3.5 Sonnet",
    "Mixtral 8x22b Instruct",
    "Gemini 1.5 Pro",
    "Llama 3 70b Chat",
    "Llama 3.1 405B Instruct",
]

# Leaderboard aggregate data (from cybench.github.io/data/leaderboard.csv)
LEADERBOARD_DATA = [
    ("Claude Opus 4.6", 37, 93, None, None, None, None, None, "5-star, partial eval (37/40 tasks)"),
    ("Claude 4.5 Opus", 39, 82, None, None, None, None, None, "3-star, partial eval (39/40 tasks)"),
    ("Claude 4.5 Sonnet", 39, 60, None, None, None, None, None, "3-star, partial eval (39/40 tasks)"),
    ("Grok 4", 40, 43, None, None, None, None, None, "4-star"),
    ("Claude 4.1 Opus", 39, 42, None, None, None, None, None, "3-star, partial eval (39/40 tasks)"),
    ("Grok 4.1 Thinking", 40, 39, None, None, None, None, None, "4-star"),
    ("Claude 4 Opus", 37, 38, None, None, None, None, None, "2-star, partial eval (37/40 tasks)"),
    ("Claude 4 Sonnet", 37, 35, None, None, None, None, None, "2-star, partial eval (37/40 tasks)"),
    ("Grok 4 Fast", 40, 30, None, None, None, None, None, "4-star"),
    ("OpenAI o3-mini", 40, 22.5, 10, None, None, None, None, "1-star, dagger"),
    ("GPT-4.5-preview", 40, 17.5, 7, None, None, None, None, "1-star"),
    ("Claude 3.7 Sonnet", 40, 20, 8, None, None, None, None, "1-star"),
    ("OpenAI o1-mini", 40, 10, 5, None, None, None, None, "1-star, dagger"),
    ("GPT-4o", 40, 12.5, 5, 17.5, 28.7, "0:11", "0:52", "Original paper"),
    ("OpenAI o1-preview", 40, 10, 4, 10, 46.8, "0:11", "0:11", "Original paper"),
    ("Claude 3 Opus", 40, 10, 4, 12.5, 36.8, "0:11", "0:11", "Original paper"),
    ("Claude 3.5 Sonnet", 40, 17.5, 7, 15, 43.9, "0:11", "0:11", "Original paper"),
    ("Mixtral 8x22b Instruct", 40, 7.5, 3, 5, 15.2, "0:09", "0:07", "Original paper"),
    ("Gemini 1.5 Pro", 40, 7.5, 3, 5, 11.7, "0:09", "0:06", "Original paper"),
    ("Llama 3 70b Chat", 40, 5, 2, 7.5, 8.2, "0:09", "0:11", "Original paper"),
    ("Llama 3.1 405B Instruct", 40, 7.5, 3, 15, 20.5, "0:09", "0:11", "Original paper"),
]


# ===========================================================================
# PDF downloading and table extraction
# ===========================================================================

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


def _normalize_task_name(name):
    """Normalize task name by removing spaces, dashes, apostrophes for matching."""
    return re.sub(r'[\s\-\'"]', '', name).lower()


def _build_name_map():
    """Build mapping from normalized (concatenated) PDF names to canonical task names."""
    return {_normalize_task_name(t): t for t in TASK_NAMES}


def _find_table_pages(pdf):
    """Find which PDF pages contain Tables 10, 11, 12 as headings.

    Looks for the pattern "Table N:" or "Table N." at the start of a line,
    distinguishing actual table headings from cross-references in text.
    """
    table_pages = {}
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        for line in text.split('\n'):
            line_nospace = line.replace(' ', '')
            for tnum in (10, 11, 12):
                # Match heading pattern: "TableN:" at start of line (with optional spaces)
                if tnum not in table_pages and re.match(
                    rf'^Table\s*{tnum}\s*[:.]', line_nospace
                ):
                    table_pages[tnum] = i
    return table_pages


def extract_table_10_11(pdf, page_idx):
    """Extract binary (✓/X) table from a page (Table 10 or 11).

    Returns dict mapping normalized task name -> list of 8 binary values.
    """
    page = pdf.pages[page_idx]
    text = page.extract_text()
    results = {}
    for line in text.split('\n'):
        # Match: TaskName HH:MM Cat Comp [8 ✓/X values]
        m = re.match(
            r'(.+?)\s+(\d{2}:\d{2})\s+([WRCFPM])\s+(HTB|GLA|S23|S22|HKC)\s+(.*)',
            line
        )
        if m and 'SuccessCount' not in line:
            task_raw = m.group(1)
            vals = [
                1 if ch == '\u2713' else 0
                for ch in m.group(5).strip().split()
                if ch in ('\u2713', 'X')
            ]
            if len(vals) == 8:
                results[_normalize_task_name(task_raw)] = vals
    return results


def extract_table_12(pdf, page_idx):
    """Extract fractional subtask scores from Table 12.

    Returns dict mapping normalized task name -> list of 8 fraction strings.
    """
    page = pdf.pages[page_idx]
    text = page.extract_text()
    results = {}
    for line in text.split('\n'):
        # Table 12 uses HH:MM:SS format for FST
        m = re.match(
            r'(.+?)\s+(\d{2}:\d{2}:\d{2})\s+([WRCFPM])\s+(HTB|GLA|S23|S22|HKC)\s+(.*)',
            line
        )
        if m and 'SumofScores' not in line:
            task_raw = m.group(1)
            vals = [
                t for t in m.group(5).strip().split()
                if t == 'X' or '/' in t
            ]
            if len(vals) == 8:
                results[_normalize_task_name(task_raw)] = vals
    return results


def extract_all_tables(pdf_path):
    """Extract Tables 10, 11, 12 from the Cybench paper PDF.

    Returns:
        unguided: dict {task_name: [8 binary values]}
        guided: dict {task_name: [8 binary values]}
        subtask: dict {task_name: [8 fraction strings]}
    """
    pdf = pdfplumber.open(pdf_path)
    table_pages = _find_table_pages(pdf)
    name_map = _build_name_map()

    if 10 not in table_pages or 11 not in table_pages or 12 not in table_pages:
        pdf.close()
        raise RuntimeError(
            f"Could not find all tables in PDF. Found pages: {table_pages}. "
            "The PDF format may have changed."
        )

    print(f"  Table 10 (unguided) on page {table_pages[10]}")
    print(f"  Table 11 (subtask-guided) on page {table_pages[11]}")
    print(f"  Table 12 (subtask scores) on page {table_pages[12]}")

    raw_t10 = extract_table_10_11(pdf, table_pages[10])
    raw_t11 = extract_table_10_11(pdf, table_pages[11])
    raw_t12 = extract_table_12(pdf, table_pages[12])
    pdf.close()

    # Map normalized names back to canonical task names
    def map_names(raw_dict, table_label):
        mapped = {}
        for norm_name, vals in raw_dict.items():
            if norm_name in name_map:
                mapped[name_map[norm_name]] = vals
            else:
                print(f"  WARNING: {table_label}: unmapped task '{norm_name}'")
        return mapped

    unguided = map_names(raw_t10, "Table 10")
    guided = map_names(raw_t11, "Table 11")
    subtask = map_names(raw_t12, "Table 12")

    # Validate completeness
    for label, data in [("Table 10", unguided), ("Table 11", guided), ("Table 12", subtask)]:
        missing = [t for t in TASK_NAMES if t not in data]
        if missing:
            raise RuntimeError(
                f"{label}: Missing {len(missing)} tasks: {missing}. "
                "PDF extraction may have failed."
            )
        if len(data) != 40:
            raise RuntimeError(f"{label}: Expected 40 tasks, got {len(data)}")

    return unguided, guided, subtask


# ===========================================================================
# Utility functions
# ===========================================================================

def fst_to_minutes(fst_str):
    """Convert FST string HH:MM or HH:MM:SS to total minutes."""
    parts = fst_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    return 0


def fraction_to_float(frac_str):
    """Convert '2/5' to 0.4, 'X' to 0.0."""
    if frac_str == "X":
        return 0.0
    parts = frac_str.split("/")
    return int(parts[0]) / int(parts[1])


def write_csv(filepath, header, rows):
    """Write a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {filepath} ({len(rows)} rows x {len(header)} cols)")


# ===========================================================================
# Build outputs
# ===========================================================================

def build_response_matrix(unguided_data):
    """Build the primary unguided response matrix."""
    header = ["task_name"] + MODELS_PAPER
    rows = []
    for task_name in TASK_NAMES:
        row = [task_name] + unguided_data[task_name]
        rows.append(row)
    filepath = os.path.join(OUTPUT_DIR, "response_matrix.csv")
    write_csv(filepath, header, rows)
    return rows


def build_subtask_guided_matrix(guided_data):
    """Build the subtask-guided response matrix."""
    header = ["task_name"] + MODELS_PAPER
    rows = []
    for task_name in TASK_NAMES:
        row = [task_name] + guided_data[task_name]
        rows.append(row)
    filepath = os.path.join(OUTPUT_DIR, "response_matrix_subtask_guided.csv")
    write_csv(filepath, header, rows)
    return rows


def build_subtask_scores_matrix(subtask_data):
    """Build the subtask fractional scores matrix."""
    header = ["task_name"] + MODELS_PAPER
    rows = []
    for task_name in TASK_NAMES:
        row = [task_name] + subtask_data[task_name]
        rows.append(row)
    filepath = os.path.join(OUTPUT_DIR, "response_matrix_subtask_scores.csv")
    write_csv(filepath, header, rows)
    return rows


def build_task_metadata():
    """Build task metadata CSV."""
    header = [
        "task_name", "task_path", "first_solve_time", "fst_minutes",
        "category", "competition", "competition_full",
    ]
    comp_full = {
        "HTB": "HackTheBox Cyber Apocalypse 2024",
        "GLA": "GlacierCTF 2023",
        "S23": "SekaiCTF 2023",
        "S22": "SekaiCTF 2022",
        "HKC": "HKCert CTF 2022",
    }
    rows = []
    for task_name, fst, cat, comp, path in TASKS:
        rows.append([
            task_name, path, fst, round(fst_to_minutes(fst), 1),
            cat, comp, comp_full[comp],
        ])
    filepath = os.path.join(OUTPUT_DIR, "task_metadata.csv")
    write_csv(filepath, header, rows)
    return rows


def build_leaderboard_aggregate():
    """Build leaderboard aggregate CSV."""
    header = [
        "model", "tasks_evaluated", "unguided_pct_solved", "flag_success_count",
        "subtask_guided_pct_solved", "subtask_pct_solved",
        "fst_unguided", "fst_subtask", "notes",
    ]
    rows = []
    for entry in LEADERBOARD_DATA:
        rows.append(list(entry))
    filepath = os.path.join(OUTPUT_DIR, "leaderboard_aggregate.csv")
    write_csv(filepath, header, rows)
    return rows


def print_summary(unguided_rows, subtask_guided_rows, subtask_scores_rows):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("CYBENCH RESPONSE MATRIX -- SUMMARY")
    print("=" * 70)

    n_tasks = len(TASK_NAMES)
    n_models = len(MODELS_PAPER)

    print(f"\nData source: arXiv:{ARXIV_ID} (Tables 10-12)")
    print(f"  Extracted programmatically using pdfplumber")

    print(f"\nDimensions:")
    print(f"  Tasks (rows):  {n_tasks}")
    print(f"  Models (cols): {n_models}")
    print(f"  Total cells:   {n_tasks * n_models}")

    # Unguided stats
    total_ones = sum(sum(r[1:]) for r in unguided_rows)
    total_cells = n_tasks * n_models
    fill_nonzero = total_ones / total_cells * 100
    print(f"\nUnguided (binary solve/fail, single attempt):")
    print(f"  Score type:    binary (1=solved, 0=failed)")
    print(f"  Total solves:  {total_ones} / {total_cells}")
    print(f"  Fill rate (non-zero): {fill_nonzero:.1f}%")
    print(f"  Per-model solve counts:")
    for i, model in enumerate(MODELS_PAPER):
        solves = sum(r[i + 1] for r in unguided_rows)
        print(f"    {model:30s}  {solves:2d}/40  ({solves/40*100:5.1f}%)")

    # Subtask-guided stats
    total_ones_sg = sum(sum(r[1:]) for r in subtask_guided_rows)
    fill_nonzero_sg = total_ones_sg / total_cells * 100
    print(f"\nSubtask-guided (binary, single attempt):")
    print(f"  Score type:    binary (1=solved, 0=failed)")
    print(f"  Total solves:  {total_ones_sg} / {total_cells}")
    print(f"  Fill rate (non-zero): {fill_nonzero_sg:.1f}%")
    print(f"  Per-model solve counts:")
    for i, model in enumerate(MODELS_PAPER):
        solves = sum(r[i + 1] for r in subtask_guided_rows)
        print(f"    {model:30s}  {solves:2d}/40  ({solves/40*100:5.1f}%)")

    # Subtask fractional stats
    total_nonzero_st = 0
    total_score_st = 0.0
    for row in subtask_scores_rows:
        for cell in row[1:]:
            val = fraction_to_float(cell)
            if val > 0:
                total_nonzero_st += 1
            total_score_st += val
    fill_nonzero_st = total_nonzero_st / total_cells * 100
    print(f"\nSubtask scores (fractional, single attempt):")
    print(f"  Score type:    fractional (e.g. 2/5)")
    print(f"  Non-zero cells: {total_nonzero_st} / {total_cells} ({fill_nonzero_st:.1f}%)")
    print(f"  Mean score (across all cells): {total_score_st / total_cells:.3f}")

    # Task difficulty distribution
    print(f"\nTask metadata:")
    cats = {}
    comps = {}
    for _, fst, cat, comp, _ in TASKS:
        cats[cat] = cats.get(cat, 0) + 1
        comps[comp] = comps.get(comp, 0) + 1
    print(f"  Categories: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")
    print(f"  Competitions: {dict(sorted(comps.items(), key=lambda x: -x[1]))}")
    fst_mins = [fst_to_minutes(t[1]) for t in TASKS]
    print(f"  FST range: {min(fst_mins):.0f} min - {max(fst_mins):.0f} min")
    print(f"  FST median: {sorted(fst_mins)[len(fst_mins)//2]:.0f} min")

    # Leaderboard summary
    print(f"\nLeaderboard (all models, aggregate only):")
    print(f"  Total models: {len(LEADERBOARD_DATA)}")
    print(f"  Models with per-task data (from paper): {n_models}")
    print(f"  Models with aggregate-only data: {len(LEADERBOARD_DATA) - n_models}")
    print(f"  Unguided solve rate range: "
          f"{min(e[2] for e in LEADERBOARD_DATA):.1f}% - "
          f"{max(e[2] for e in LEADERBOARD_DATA):.1f}%")


def main():
    print("Building Cybench response matrices...")
    print()

    # Step 1: Download PDF
    print("Step 1: Download paper PDF")
    download_pdf()

    # Step 2: Extract tables from PDF
    print("\nStep 2: Extract Tables 10-12 from PDF")
    unguided, guided, subtask = extract_all_tables(PDF_PATH)
    print(f"  Extracted: {len(unguided)} unguided, {len(guided)} guided, "
          f"{len(subtask)} subtask entries")

    # Step 3: Build response matrices
    print("\nStep 3: Build response matrices")
    unguided_rows = build_response_matrix(unguided)
    subtask_guided_rows = build_subtask_guided_matrix(guided)
    subtask_scores_rows = build_subtask_scores_matrix(subtask)
    build_task_metadata()
    build_leaderboard_aggregate()

    print_summary(unguided_rows, subtask_guided_rows, subtask_scores_rows)


if __name__ == "__main__":
    main()
