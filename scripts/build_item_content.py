#!/usr/bin/env python3
"""Extract item content (task descriptions, questions, instructions) for all benchmarks.

Creates ``processed/item_content.csv`` (columns: item_id, content) for each
benchmark, to be consumed by ``migrate_bench_data.py`` during .pt conversion.

Usage:
    # Build all benchmarks (local sources + HuggingFace downloads)
    python scripts/build_item_content.py

    # Build a single benchmark
    python scripts/build_item_content.py --benchmark swebench

    # Local-only (skip HuggingFace downloads)
    python scripts/build_item_content.py --local-only

    # Custom data directory
    python scripts/build_item_content.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _write_content(benchmark: str, rows: list[tuple[str, str]]) -> Path:
    """Write item_content.csv for a benchmark. Returns the output path."""
    out_dir = DATA_DIR / f"{benchmark}_data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "item_content.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "content"])
        for item_id, content in rows:
            writer.writerow([str(item_id), content])
    print(f"  {benchmark}: wrote {len(rows)} items to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Tier 1: Local extraction (no network required)
# ---------------------------------------------------------------------------


def build_cruxeval():
    """CRUXEval — code reasoning tasks from JSONL."""
    jsonl_path = DATA_DIR / "cruxeval_data" / "raw" / "cruxeval" / "data" / "cruxeval.jsonl"
    if not jsonl_path.exists():
        print(f"  cruxeval: SKIP — {jsonl_path} not found")
        return
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            # Content: the code snippet (primary item content for IRT)
            content = obj.get("code", "")
            item_id = obj.get("id", "")
            rows.append((item_id, content))
    _write_content("cruxeval", rows)


def build_mmlupro():
    """MMLU-Pro — multiple-choice questions from raw CSV."""
    csv_path = DATA_DIR / "mmlupro_data" / "raw" / "mmlu_pro_test_questions.csv"
    if not csv_path.exists():
        print(f"  mmlupro: SKIP — {csv_path} not found")
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        question = str(row.get("question", ""))
        options = str(row.get("options", ""))
        category = str(row.get("category", ""))
        content = f"[{category}] {question}\nOptions: {options}"
        rows.append((qid, content))
    _write_content("mmlupro", rows)


def build_clinebench():
    """ClineBench — task instructions from markdown files."""
    tasks_dir = DATA_DIR / "clinebench_data" / "raw" / "cline-bench" / "tasks"
    if not tasks_dir.exists():
        print(f"  clinebench: SKIP — {tasks_dir} not found")
        return
    rows = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        instruction_file = task_dir / "instruction.md"
        if instruction_file.exists():
            content = instruction_file.read_text(encoding="utf-8").strip()
        else:
            content = task_dir.name
        # RM uses just the hash prefix (before the first hyphen-word slug)
        # e.g., "01k6kr5hbv8za80v8vnze3at8h" from "01k6kr5hbv8za80v8vnze3at8h-every-plugin-api-migration"
        task_id = task_dir.name.split("-")[0] if "-" in task_dir.name else task_dir.name
        rows.append((task_id, content))
    _write_content("clinebench", rows)


def build_arcagi():
    """ARC-AGI v1 — abstract reasoning grids from JSON files."""
    eval_dir = DATA_DIR / "arcagi_data" / "raw" / "arc-agi-1" / "data" / "evaluation"
    if not eval_dir.exists():
        print(f"  arcagi: SKIP — {eval_dir} not found")
        return
    rows = []
    for json_file in sorted(eval_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file) as f:
            task_data = json.load(f)
        # Serialize train examples + test input as content
        train_examples = task_data.get("train", [])
        test_examples = task_data.get("test", [])
        parts = []
        for i, ex in enumerate(train_examples):
            inp = ex.get("input", [])
            out = ex.get("output", [])
            parts.append(f"Train {i}: {_grid_to_str(inp)} -> {_grid_to_str(out)}")
        for i, ex in enumerate(test_examples):
            inp = ex.get("input", [])
            parts.append(f"Test {i}: {_grid_to_str(inp)}")
        content = "\n".join(parts)
        rows.append((task_id, content))
    _write_content("arcagi", rows)


def _grid_to_str(grid: list[list[int]]) -> str:
    """Compact string representation of an ARC grid."""
    return "[" + "|".join("".join(str(c) for c in row) for row in grid) + "]"


def build_swepolybench():
    """SWE-PolyBench — GitHub issue descriptions from instance JSON."""
    for split_name, filename in [("full", "full_instances.json"), ("verified", "verified_instances.json")]:
        json_path = DATA_DIR / "swepolybench_data" / "raw" / filename
        if not json_path.exists():
            continue
        with open(json_path) as f:
            instances = json.load(f)
        rows = []
        for inst in instances:
            iid = inst.get("instance_id", "")
            # Try different fields for content
            content = inst.get("problem_statement", "")
            if not content:
                content = inst.get("issue_body", "")
            if not content:
                content = inst.get("title", "")
            if not content:
                content = iid
            rows.append((str(iid), str(content)))
        if rows:
            _write_content("swepolybench", rows)
            return  # Use the first available split
    print("  swepolybench: SKIP — no instance JSON found")


def build_workarena():
    """WorkArena — task descriptions from per-task JSON files."""
    arb_dir = DATA_DIR / "workarena_data" / "raw" / "agentrewardbench"
    if not arb_dir.exists():
        print(f"  workarena: SKIP — {arb_dir} not found")
        return
    # Collect unique task descriptions from any agent's JSON files
    content_map: dict[str, str] = {}
    for agent_dir in arb_dir.iterdir():
        if not agent_dir.is_dir():
            continue
        for json_file in agent_dir.glob("workarena.servicenow.*.json"):
            # Extract task name from filename
            fname = json_file.stem
            # Keep full name to match RM column format: workarena.servicenow.all-menu
            task_name = fname
            if task_name in content_map:
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                # Look for task description in various fields
                content = ""
                if isinstance(data, dict):
                    content = data.get("goal", "") or data.get("task", "") or data.get("intent", "")
                    if not content and "chat_messages" in data:
                        msgs = data["chat_messages"]
                        if msgs and isinstance(msgs, list) and len(msgs) > 0:
                            first = msgs[0]
                            if isinstance(first, dict):
                                content = first.get("content", "")[:500]
                if not content:
                    content = task_name.replace("-", " ").replace("_", " ")
                content_map[task_name] = content
            except (json.JSONDecodeError, KeyError):
                content_map[task_name] = task_name.replace("-", " ").replace("_", " ")
    rows = [(k, v) for k, v in sorted(content_map.items())]
    if rows:
        _write_content("workarena", rows)
    else:
        print("  workarena: SKIP — no task JSON files found")


def build_paperbench():
    """PaperBench — paper names as content."""
    paper_names = [
        "adaptive-pruning", "all-in-one", "bam", "bbox", "bridging-data-gaps",
        "fre", "ftrl", "lbcs", "lca-on-the-line", "mechanistic-understanding",
        "pinn", "rice", "robust-clip", "sample-specific-masks", "sapg",
        "sequential-neural-score-estimation", "stay-on-topic-with-classifier-free-guidance",
        "stochastic-interpolants", "test-time-model-adaptation", "what-will-my-model-forget",
    ]
    rows = [(name, name.replace("-", " ").title()) for name in paper_names]
    _write_content("paperbench", rows)


def build_agentbench():
    """AgentBench — environment descriptions (8 items)."""
    env_map = {
        "OS": "Operating System — command-line interaction and system administration tasks",
        "DB": "Database — SQL query generation and database manipulation tasks",
        "KG": "Knowledge Graph — SPARQL query generation and knowledge reasoning tasks",
        "DCG": "Digital Card Game — strategic decision-making in card game environments",
        "LTP": "Lateral Thinking Puzzles — creative reasoning and puzzle-solving tasks",
        "HH": "House Holding — embodied agent household task planning and execution",
        "WS": "Web Shopping — product search and purchase decision tasks on e-commerce sites",
        "WB": "Web Browsing — information retrieval and navigation tasks on the web",
    }
    rows = [(k, v) for k, v in env_map.items()]
    _write_content("agentbench", rows)


def build_browsergym():
    """BrowserGym — benchmark descriptions (8 items)."""
    bench_map = {
        "MiniWoB": "MiniWoB++ — simple web interaction tasks (click, type, navigate)",
        "WebArena": "WebArena — complex web navigation and task completion on realistic websites",
        "VisualWebArena": "VisualWebArena — multimodal web tasks requiring visual understanding",
        "WorkArena-L1": "WorkArena Level 1 — basic ServiceNow enterprise web tasks",
        "WorkArena-L2": "WorkArena Level 2 — intermediate ServiceNow enterprise web tasks",
        "WorkArena-L3": "WorkArena Level 3 — advanced ServiceNow enterprise web tasks",
        "AssistantBench": "AssistantBench — open-ended web assistant tasks",
        "WebLINX": "WebLINX — web navigation following natural language instructions",
    }
    rows = [(k, v) for k, v in bench_map.items()]
    _write_content("browsergym", rows)


def build_androidworld():
    """AndroidWorld — task names from metadata."""
    meta_path = DATA_DIR / "androidworld_data" / "processed" / "task_metadata.csv"
    if not meta_path.exists():
        print(f"  androidworld: SKIP — {meta_path} not found")
        return
    import pandas as pd
    df = pd.read_csv(meta_path)
    rows = []
    for _, row in df.iterrows():
        task_id = str(row["task_id"])
        # Build description from available metadata
        parts = [task_id]
        if "primary_app" in df.columns:
            parts.append(f"App: {row['primary_app']}")
        if "task_type" in df.columns:
            parts.append(f"Type: {row['task_type']}")
        content = " | ".join(parts)
        rows.append((task_id, content))
    _write_content("androidworld", rows)


# ---------------------------------------------------------------------------
# Tier 2: HuggingFace downloads (network required)
# ---------------------------------------------------------------------------


def _load_hf_dataset(repo_id: str, name: str | None = None, split: str = "test", columns: list[str] | None = None):
    """Load a HuggingFace dataset, return as Dataset object."""
    from datasets import load_dataset
    kwargs = {}
    if columns:
        kwargs["columns"] = columns
    ds = load_dataset(repo_id, name=name, split=split, **kwargs)
    return ds


def build_swebench():
    """SWE-bench — GitHub issue problem statements from HF."""
    try:
        ds = _load_hf_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = []
        for item in ds:
            iid = item["instance_id"]
            content = item.get("problem_statement", "")
            rows.append((str(iid), str(content)))
        _write_content("swebench", rows)
    except Exception as e:
        print(f"  swebench: SKIP — {e}")


def build_swebench_full():
    """SWE-bench Full — GitHub issue problem statements from HF."""
    try:
        ds = _load_hf_dataset("princeton-nlp/SWE-bench", split="test")
        rows = []
        for item in ds:
            iid = item["instance_id"]
            content = item.get("problem_statement", "")
            rows.append((str(iid), str(content)))
        _write_content("swebench_full", rows)
    except Exception as e:
        print(f"  swebench_full: SKIP — {e}")


def build_swebench_java():
    """SWE-bench Java — problem statements from Multi-SWE-bench HF + full SWE-bench."""
    content_map: dict[str, str] = {}
    # Multi-SWE-bench java_verified (91 instances)
    try:
        ds = _load_hf_dataset("Daoguang/Multi-SWE-bench", split="java_verified")
        for item in ds:
            iid = str(item.get("instance_id", ""))
            content = str(item.get("problem_statement", ""))
            content_map[iid] = content
    except Exception as e:
        print(f"  swebench_java: Multi-SWE-bench failed — {e}")

    # Also check SWE-bench Multilingual for additional Java instances
    try:
        ds = _load_hf_dataset("SWE-bench/SWE-bench_Multilingual", split="test")
        for item in ds:
            iid = str(item.get("instance_id", ""))
            if iid not in content_map:
                content = str(item.get("problem_statement", ""))
                content_map[iid] = content
    except Exception:
        pass

    if content_map:
        rows = [(k, v) for k, v in sorted(content_map.items())]
        _write_content("swebench_java", rows)
    else:
        print("  swebench_java: SKIP — no data found")


def build_swebench_multilingual():
    """SWE-bench Multilingual — problem statements from HF datasets."""
    all_rows = {}
    # Official SWE-bench Multilingual (300 instances)
    try:
        ds = _load_hf_dataset("SWE-bench/SWE-bench_Multilingual", split="test")
        for item in ds:
            iid = str(item.get("instance_id", ""))
            content = str(item.get("problem_statement", ""))
            all_rows[iid] = content
    except Exception as e:
        print(f"  swebench_multilingual: SWE-bench/SWE-bench_Multilingual failed — {e}")

    # Multi-SWE-bench Java verified (91 instances)
    try:
        ds = _load_hf_dataset("Daoguang/Multi-SWE-bench", split="java_verified")
        for item in ds:
            iid = str(item.get("instance_id", ""))
            if iid not in all_rows:
                content = str(item.get("problem_statement", ""))
                all_rows[iid] = content
    except Exception:
        pass

    # Also pull from SWE-bench full for any Python instances that overlap
    try:
        ds = _load_hf_dataset("princeton-nlp/SWE-bench", split="test")
        for item in ds:
            iid = str(item.get("instance_id", ""))
            if iid not in all_rows:
                content = str(item.get("problem_statement", ""))
                all_rows[iid] = content
    except Exception:
        pass

    if all_rows:
        rows = [(k, v) for k, v in sorted(all_rows.items())]
        _write_content("swebench_multilingual", rows)
    else:
        print("  swebench_multilingual: SKIP — no data found")


def build_evalplus():
    """EvalPlus — HumanEval and MBPP prompts from HF."""
    rows = []
    # HumanEval+
    try:
        ds = _load_hf_dataset("evalplus/humanevalplus", split="test")
        for item in ds:
            task_id = item.get("task_id", "")
            content = item.get("prompt", "")
            rows.append((str(task_id), str(content)))
    except Exception as e:
        print(f"  evalplus (humaneval): WARN — {e}")

    # MBPP+
    try:
        ds = _load_hf_dataset("evalplus/mbppplus", split="test")
        for item in ds:
            task_id = item.get("task_id", "")
            # Ensure Mbpp/ prefix to match RM column names
            if task_id and not str(task_id).startswith("Mbpp/"):
                task_id = f"Mbpp/{task_id}"
            content = item.get("prompt", "")
            if not content:
                content = item.get("text", "")
            rows.append((str(task_id), str(content)))
    except Exception as e:
        print(f"  evalplus (mbpp): WARN — {e}")

    if rows:
        _write_content("evalplus", rows)
    else:
        print("  evalplus: SKIP — no data found")


def build_bigcodebench():
    """BigCodeBench — task descriptions from HF."""
    try:
        ds = _load_hf_dataset("bigcode/bigcodebench", split="v0.1.2")
        rows = []
        for item in ds:
            task_id = item.get("task_id", "")
            content = item.get("complete_prompt", "")
            if not content:
                content = item.get("instruct_prompt", "")
            rows.append((str(task_id), str(content)))
        _write_content("bigcodebench", rows)
    except Exception as e:
        print(f"  bigcodebench: SKIP — {e}")


def build_hle():
    """Humanity's Last Exam — questions from HF or local JSON."""
    # Try HF first — select only text columns to avoid image decoding issues
    try:
        ds = _load_hf_dataset("cais/hle", split="test",
                              columns=["id", "question", "category", "answer_type"])
        rows = []
        for item in ds:
            qid = item.get("id", "")
            question = item.get("question", "")
            category = item.get("category", "")
            content = f"[{category}] {question}" if category else question
            rows.append((str(qid), str(content)))
        if rows:
            _write_content("hle", rows)
            return
    except Exception as e:
        print(f"  hle: HF attempt failed — {e}")

    # Fallback: local JSON
    json_path = DATA_DIR / "hle_data" / "raw" / "judged_hle_pro.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        rows = []
        for qid, info in data.items():
            content = ""
            if isinstance(info, dict):
                content = info.get("question", "") or info.get("body", "")
            rows.append((str(qid), str(content)))
        if any(c for _, c in rows):
            _write_content("hle", rows)
            return
    print("  hle: SKIP — no question text found (dataset may be gated)")


def build_livebench():
    """LiveBench — question text from HF per-category datasets."""
    rows = []
    categories = ["coding", "data_analysis", "instruction_following", "language", "math", "reasoning"]
    for cat in categories:
        try:
            ds = _load_hf_dataset(f"livebench/{cat}", split="test")
            for item in ds:
                qid = item.get("question_id", "")
                # Content from turns or question field
                content = ""
                turns = item.get("turns", [])
                if turns and isinstance(turns, list):
                    content = str(turns[0]) if turns else ""
                if not content:
                    content = item.get("question", "")
                rows.append((str(qid), str(content)))
        except Exception:
            pass
    if rows:
        _write_content("livebench", rows)
    else:
        print("  livebench: SKIP — no data found")


def build_arcagi_v2():
    """ARC-AGI v2 — grid data from GitHub fchollet/ARC-AGI-2 repo."""
    eval_dir = DATA_DIR / "arcagi_data" / "raw" / "arc-agi-2" / "data" / "evaluation"
    if not eval_dir.exists():
        # Try to download the evaluation data from GitHub
        try:
            import urllib.request
            import zipfile
            import io
            print("  arcagi_v2: downloading ARC-AGI-2 evaluation data from GitHub...")
            url = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"
            resp = urllib.request.urlopen(url)
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            # Extract just the evaluation JSONs
            eval_dir.mkdir(parents=True, exist_ok=True)
            for name in z.namelist():
                if "/data/evaluation/" in name and name.endswith(".json"):
                    fname = name.split("/")[-1]
                    with z.open(name) as src:
                        (eval_dir / fname).write_bytes(src.read())
        except Exception as e:
            print(f"  arcagi_v2: SKIP — could not download: {e}")
            return

    if not eval_dir.exists():
        print("  arcagi_v2: SKIP — evaluation dir not found")
        return

    rows = []
    for json_file in sorted(eval_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file) as f:
            task_data = json.load(f)
        train_examples = task_data.get("train", [])
        test_examples = task_data.get("test", [])
        parts = []
        for i, ex in enumerate(train_examples):
            inp = ex.get("input", [])
            out = ex.get("output", [])
            parts.append(f"Train {i}: {_grid_to_str(inp)} -> {_grid_to_str(out)}")
        for i, ex in enumerate(test_examples):
            inp = ex.get("input", [])
            parts.append(f"Test {i}: {_grid_to_str(inp)}")
        content = "\n".join(parts)
        rows.append((task_id, content))
    if rows:
        # Write as separate file, don't overwrite v1
        out_dir = DATA_DIR / "arcagi_data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "item_content_v2.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "content"])
            for item_id, content in rows:
                writer.writerow([str(item_id), content])
        print(f"  arcagi_v2: wrote {len(rows)} items to {out_path}")


def build_osworld():
    """OSWorld — task instructions. Try HF dataset or local files."""
    # Try the HF evaluation config
    try:
        ds = _load_hf_dataset("xlangai/osworld_taskinfo", split="train")
        rows = []
        for item in ds:
            task_id = item.get("id", item.get("task_id", ""))
            content = item.get("instruction", item.get("intent", ""))
            rows.append((str(task_id), str(content)))
        if rows:
            _write_content("osworld", rows)
            return
    except Exception:
        pass

    # Fallback: try local test_all.json — RM uses "domain/uuid" format
    json_path = DATA_DIR / "osworld_data" / "raw" / "test_all.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        rows = []
        for domain, task_ids in data.items():
            for tid in task_ids:
                # RM column format is "domain/uuid"
                item_id = f"{domain}/{tid}"
                content = f"Domain: {domain} | Task: {tid}"
                rows.append((item_id, content))
        _write_content("osworld", rows)
    else:
        print("  osworld: SKIP — no task data found")


def build_theagentcompany():
    """TheAgentCompany — task descriptions from evaluation results."""
    eval_dir = DATA_DIR / "theagentcompany_data" / "raw" / "experiments" / "evaluation" / "1.0.0"
    if not eval_dir.exists():
        print(f"  theagentcompany: SKIP — {eval_dir} not found")
        return
    task_ids = set()
    for model_dir in eval_dir.iterdir():
        if not model_dir.is_dir():
            continue
        results_dir = model_dir / "results"
        if not results_dir.exists():
            continue
        for json_file in results_dir.glob("eval_*.json"):
            task_id = json_file.stem.replace("eval_", "")
            task_ids.add(task_id)

    # Task names are descriptive kebab-case slugs
    rows = [(tid, tid.replace("-", " ").replace("_", " ")) for tid in sorted(task_ids)]
    if rows:
        _write_content("theagentcompany", rows)
    else:
        print("  theagentcompany: SKIP — no task files found")


def build_toolbench():
    """ToolBench — query text from baselines data."""
    baselines_dir = DATA_DIR / "toolbench_data" / "raw" / "data_baselines"
    if not baselines_dir.exists():
        print(f"  toolbench: SKIP — {baselines_dir} not found")
        return
    content_map: dict[str, str] = {}
    # Structure: data_baselines/<model>/<scenario>/<task_id>_*.json
    # RM column names are like "G1_category_1198", "G2_instruction_87064"
    for model_dir in baselines_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for scenario_dir in model_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            scenario = scenario_dir.name  # e.g., "G2_instruction"
            for json_file in scenario_dir.glob("*.json"):
                # Task ID from filename: e.g., "87064_CoT@1.json" -> "87064"
                raw_id = json_file.stem.split("_")[0]
                # Use composite key matching RM format: G1_category_1198
                tid = f"{scenario}_{raw_id}"
                if tid in content_map:
                    continue
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    # Query is nested inside answer_generation
                    query = ""
                    if isinstance(data, dict):
                        ag = data.get("answer_generation", {})
                        if isinstance(ag, dict):
                            query = ag.get("query", "")
                    if query:
                        content_map[tid] = query
                except (json.JSONDecodeError, KeyError):
                    pass
    rows = [(k, v) for k, v in sorted(content_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])]
    if rows:
        _write_content("toolbench", rows)
    else:
        print("  toolbench: SKIP — no query text found in baselines")


def build_matharena():
    """MathArena — problem descriptions from parquet files."""
    raw_dir = DATA_DIR / "matharena_data" / "raw"
    if not raw_dir.exists():
        print(f"  matharena: SKIP — {raw_dir} not found")
        return
    import pandas as pd
    content_map: dict[str, str] = {}
    for parquet_file in sorted(raw_dir.glob("*.parquet")):
        competition = parquet_file.stem  # e.g., "aime_2025"
        try:
            df = pd.read_parquet(parquet_file)
            for _, row in df.iterrows():
                pid = str(row.get("problem_idx", ""))
                # RM format: aime_2025_p1 (with "p" prefix before problem number)
                item_id = f"{competition}_p{pid}"
                # Content: competition name + problem index + gold answer if available
                gold = str(row.get("gold_answer", ""))
                content = f"{competition.replace('_', ' ').title()} Problem {pid}"
                if gold:
                    content += f" (Answer: {gold})"
                if item_id not in content_map:
                    content_map[item_id] = content
        except Exception:
            pass
    rows = [(k, v) for k, v in sorted(content_map.items())]
    if rows:
        _write_content("matharena", rows)
    else:
        print("  matharena: SKIP — no parquet files found")


def build_webarena():
    """WebArena — task intents. Try local config, then download from GitHub."""
    config_path = DATA_DIR / "webarena_data" / "raw" / "test.raw.json"
    if not config_path.exists():
        # Download from WebArena GitHub repo
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
            print("  webarena: downloading test config from GitHub...")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, config_path)
        except Exception as e:
            print(f"  webarena: SKIP — could not download config: {e}")
            return

    with open(config_path) as f:
        tasks = json.load(f)
    rows = []
    for task in tasks:
        tid = str(task.get("task_id", ""))
        content = task.get("intent", "")
        rows.append((tid, str(content)))
    _write_content("webarena", rows)


def build_bfcl():
    """BFCL — function calling task specs from HF via hf_hub_download.

    The BFCL dataset has multiple JSON files with different schemas, so
    load_dataset fails. Instead we download individual JSONL files and parse
    them manually. Item IDs in the response matrix use the format
    ``category::item_id`` (e.g. ``simple::simple_0``).
    """
    from huggingface_hub import hf_hub_download

    bfcl_repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

    # Map RM category prefix -> HuggingFace JSON filename
    category_files = {
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

    def _extract_question(question) -> str:
        if isinstance(question, str):
            return question.strip()
        if not isinstance(question, list) or len(question) == 0:
            return ""
        parts = []
        for turn in question:
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        parts.append(msg.get("content", ""))
            elif isinstance(turn, dict) and turn.get("role") == "user":
                parts.append(turn.get("content", ""))
        return "\n---\n".join(parts).strip()

    def _summarize_funcs(functions) -> str:
        if not isinstance(functions, list):
            functions = [functions] if isinstance(functions, dict) else []
        summaries = []
        for func in functions:
            if not isinstance(func, dict):
                continue
            name = func.get("name", "?")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            param_parts = []
            if isinstance(params, dict):
                for pname, pinfo in params.get("properties", {}).items():
                    ptype = pinfo.get("type", "?") if isinstance(pinfo, dict) else "?"
                    param_parts.append(f"{pname}: {ptype}")
            sig = f"{name}({', '.join(param_parts)})"
            if desc:
                sig += f" - {desc}"
            summaries.append(sig)
        return "\n".join(summaries)

    rows = []
    for category, filename in category_files.items():
        try:
            path = hf_hub_download(
                repo_id=bfcl_repo, filename=filename, repo_type="dataset"
            )
        except Exception as e:
            print(f"    WARN: failed to download {filename}: {e}")
            continue
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
        for item in data:
            item_id = item.get("id", "")
            if not item_id:
                continue
            rm_key = f"{category}::{item_id}"
            q_text = _extract_question(item.get("question", ""))
            f_text = _summarize_funcs(item.get("function", []))
            parts = []
            if q_text:
                parts.append(f"Question: {q_text}")
            if f_text:
                parts.append(f"Functions:\n{f_text}")
            content = "\n\n".join(parts) if parts else item_id
            rows.append((rm_key, content))

    if rows:
        _write_content("bfcl", rows)
    else:
        print("  bfcl: SKIP — no content source found")


def build_appworld():
    """AppWorld — task metadata from HF parquet."""
    parquet_path = DATA_DIR / "appworld_data" / "raw" / "appworld_tasks.parquet"
    if parquet_path.exists():
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        rows = []
        for _, row in df.iterrows():
            tid = str(row.get("task_id", ""))
            instruction = str(row.get("instruction", ""))
            if not instruction:
                instruction = str(row.get("description", ""))
            rows.append((tid, instruction))
        if rows:
            _write_content("appworld", rows)
            return
    print("  appworld: SKIP — no task content found")


def build_gaia():
    """GAIA — general AI assistant questions from HF (may be gated)."""
    try:
        ds = _load_hf_dataset("gaia-benchmark/GAIA", name="2023_all", split="test")
        rows = []
        for item in ds:
            tid = item.get("task_id", "")
            content = item.get("Question", item.get("question", ""))
            rows.append((str(tid), str(content)))
        if rows:
            _write_content("gaia", rows)
            return
    except Exception as e:
        print(f"  gaia: SKIP — {e} (dataset may be gated)")


def build_agentdojo():
    """AgentDojo — task descriptions from local clone or metadata."""
    meta_path = DATA_DIR / "agentdojo_data" / "processed" / "task_metadata.csv"
    if meta_path.exists():
        import pandas as pd
        df = pd.read_csv(meta_path)
        rows = []
        for _, row in df.iterrows():
            tid = str(row.get("task_key", ""))
            # Build content from available fields
            suite = str(row.get("suite", ""))
            task_type = str(row.get("task_type", ""))
            content = f"{suite} — {task_type} task: {tid}"
            rows.append((tid, content))
        if rows:
            _write_content("agentdojo", rows)
            return
    print("  agentdojo: SKIP — no task metadata found")


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

# Map benchmark name -> builder function
BUILDERS: dict[str, callable] = {
    # Tier 1: Local extraction
    "cruxeval": build_cruxeval,
    "mmlupro": build_mmlupro,
    "clinebench": build_clinebench,
    "arcagi": build_arcagi,
    "swepolybench": build_swepolybench,
    "workarena": build_workarena,
    "paperbench": build_paperbench,
    "agentbench": build_agentbench,
    "browsergym": build_browsergym,
    "androidworld": build_androidworld,
    "agentdojo": build_agentdojo,
    "appworld": build_appworld,
    # Tier 2: HuggingFace downloads
    "swebench": build_swebench,
    "swebench_full": build_swebench_full,
    "swebench_java": build_swebench_java,
    "swebench_multilingual": build_swebench_multilingual,
    "evalplus": build_evalplus,
    "bigcodebench": build_bigcodebench,
    "hle": build_hle,
    "livebench": build_livebench,
    "arcagi_v2": build_arcagi_v2,
    "osworld": build_osworld,
    "theagentcompany": build_theagentcompany,
    "toolbench": build_toolbench,
    "matharena": build_matharena,
    "webarena": build_webarena,
    "bfcl": build_bfcl,
    "gaia": build_gaia,
}

LOCAL_ONLY_BENCHMARKS = {
    "cruxeval", "mmlupro", "clinebench", "arcagi", "swepolybench",
    "workarena", "paperbench", "agentbench", "browsergym", "androidworld",
    "agentdojo", "appworld",
}


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Build item_content.csv for benchmarks")
    parser.add_argument("--benchmark", type=str, default=None, help="Process only this benchmark")
    parser.add_argument("--local-only", action="store_true", help="Skip HuggingFace downloads")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = args.data_dir

    if args.benchmark:
        benchmarks = [args.benchmark]
    elif args.local_only:
        benchmarks = sorted(LOCAL_ONLY_BENCHMARKS)
    else:
        benchmarks = list(BUILDERS.keys())

    print(f"Building item content for {len(benchmarks)} benchmarks...")
    print("=" * 60)

    for name in benchmarks:
        if name not in BUILDERS:
            print(f"  {name}: unknown benchmark, skipping")
            continue
        if args.local_only and name not in LOCAL_ONLY_BENCHMARKS:
            continue
        try:
            BUILDERS[name]()
        except Exception as e:
            print(f"  {name}: ERROR — {e}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
