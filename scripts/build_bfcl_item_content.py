#!/usr/bin/env python3
"""Build item_content.csv for BFCL (Berkeley Function Calling Leaderboard).

Downloads BFCL v3 JSON files from HuggingFace via hf_hub_download,
extracts item content (question + function signatures), and maps to
response matrix item IDs.

Usage:
    python scripts/build_bfcl_item_content.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "bfcl_data" / "processed" / "item_content.csv"
RM_PATH = DATA_DIR / "bfcl_data" / "processed" / "response_matrix.csv"

# Map from RM category prefix to HuggingFace JSON filename.
# "chatable" is excluded because it has no id field and doesn't appear in the RM.
# "multi_turn_composite" and "sql" are not in the RM either.
CATEGORY_TO_FILE: dict[str, str] = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "exec_simple": "BFCL_v3_exec_simple.json",
    "exec_multiple": "BFCL_v3_exec_multiple.json",
    "exec_parallel": "BFCL_v3_exec_parallel.json",
    "exec_parallel_multiple": "BFCL_v3_exec_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
    "rest": "BFCL_v3_rest.json",
    "java": "BFCL_v3_java.json",
    "javascript": "BFCL_v3_javascript.json",
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    "live_parallel": "BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
    "live_irrelevance": "BFCL_v3_live_irrelevance.json",
    "live_relevance": "BFCL_v3_live_relevance.json",
    "multi_turn_base": "BFCL_v3_multi_turn_base.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param.json",
}


def _download(filename: str) -> Path:
    """Download a file from the BFCL HuggingFace repo."""
    return Path(
        hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    )


def _extract_question_text(question) -> str:
    """Extract user-facing question text from BFCL question field.

    The question field can be:
    - A list of message dicts: [[{"role": "user", "content": "..."}]]
    - A list of lists of message dicts (multi-turn): [list_of_msgs, ...]
    - A string (rare)
    """
    if isinstance(question, str):
        return question.strip()

    if not isinstance(question, list) or len(question) == 0:
        return ""

    parts = []
    for turn in question:
        # Each turn is either a list of message dicts or a single message dict
        if isinstance(turn, list):
            for msg in turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    parts.append(msg.get("content", ""))
        elif isinstance(turn, dict) and turn.get("role") == "user":
            parts.append(turn.get("content", ""))

    return "\n---\n".join(parts).strip()


def _summarize_functions(functions) -> str:
    """Create a compact summary of function signatures.

    Each function is a dict with 'name', 'description', and 'parameters'.
    We include name + description + parameter names/types for context.
    """
    if not isinstance(functions, list):
        if isinstance(functions, dict):
            functions = [functions]
        else:
            return ""

    summaries = []
    for func in functions:
        if not isinstance(func, dict):
            continue
        name = func.get("name", "?")
        desc = func.get("description", "")
        # Extract parameter names and types
        params = func.get("parameters", {})
        param_parts = []
        if isinstance(params, dict):
            props = params.get("properties", {})
            if isinstance(props, dict):
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "?") if isinstance(pinfo, dict) else "?"
                    param_parts.append(f"{pname}: {ptype}")
        params_str = ", ".join(param_parts) if param_parts else ""
        summary = f"{name}({params_str})"
        if desc:
            summary += f" - {desc}"
        summaries.append(summary)

    return "\n".join(summaries)


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_rm_item_ids() -> set[str]:
    """Read item IDs from the response matrix header."""
    with open(RM_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
    # First column is 'model', rest are item IDs in format "category::item_id"
    return set(header[1:])


def build_content_map() -> dict[str, str]:
    """Download all BFCL files and build a mapping from RM item_id to content."""
    content_map: dict[str, str] = {}

    for category, filename in CATEGORY_TO_FILE.items():
        print(f"  Downloading {filename}...")
        try:
            path = _download(filename)
        except Exception as e:
            print(f"    WARN: failed to download {filename}: {e}")
            continue

        data = _load_jsonl(path)
        print(f"    Loaded {len(data)} items")

        for item in data:
            item_id = item.get("id", "")
            if not item_id:
                continue

            # RM format is "category::item_id"
            rm_key = f"{category}::{item_id}"

            # Extract question text
            question_text = _extract_question_text(item.get("question", ""))

            # Extract function signatures (not all categories have this)
            func_summary = _summarize_functions(item.get("function", []))

            # Combine question + function context
            parts = []
            if question_text:
                parts.append(f"Question: {question_text}")
            if func_summary:
                parts.append(f"Functions:\n{func_summary}")

            content = "\n\n".join(parts) if parts else item_id
            content_map[rm_key] = content

    return content_map


def main():
    print("Building BFCL item_content.csv...")
    print("=" * 60)

    # Step 1: Get RM item IDs for verification
    rm_ids = _get_rm_item_ids()
    print(f"Response matrix has {len(rm_ids)} items")

    # Step 2: Build content map from HuggingFace downloads
    content_map = build_content_map()
    print(f"\nExtracted content for {len(content_map)} items")

    # Step 3: Check match rate
    matched = rm_ids & set(content_map.keys())
    missing_from_content = rm_ids - set(content_map.keys())
    extra_in_content = set(content_map.keys()) - rm_ids

    print(f"\nMatch statistics:")
    print(f"  RM items matched:      {len(matched)} / {len(rm_ids)} ({100*len(matched)/len(rm_ids):.1f}%)")
    if missing_from_content:
        print(f"  RM items NOT matched:  {len(missing_from_content)}")
        # Show a few examples
        for item in sorted(missing_from_content)[:10]:
            print(f"    {item}")
        if len(missing_from_content) > 10:
            print(f"    ... and {len(missing_from_content) - 10} more")
    if extra_in_content:
        print(f"  Extra (not in RM):     {len(extra_in_content)}")

    # Step 4: Write CSV — include all RM items, use ID as fallback content
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "content"])
        # Write in RM order
        with open(RM_PATH) as rm_f:
            rm_reader = csv.reader(rm_f)
            rm_header = next(rm_reader)
        for rm_id in rm_header[1:]:
            content = content_map.get(rm_id, rm_id)  # fallback to ID
            writer.writerow([rm_id, content])
            rows_written += 1

    print(f"\nWrote {rows_written} items to {OUT_PATH}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
