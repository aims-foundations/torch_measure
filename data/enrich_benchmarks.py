#!/usr/bin/env python3
"""
Enrich benchmark papers with:
1. Online availability (GitHub / HuggingFace URLs)
2. Number of models evaluated
3. Whether response/output data is available

Reads from benchmark_papers/benchmarks.json (full crawl) and
benchmark_papers/benchmark_list.json (filtered benchmarks).

Usage:
    python enrich_benchmarks.py
    python enrich_benchmarks.py --search_apis       # Also search GitHub/HF APIs
    python enrich_benchmarks.py --input benchmarks.json --output enriched.json
"""

import argparse
import json
import logging
import os
import re
import time

import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Extract URLs from abstracts
# ---------------------------------------------------------------------------

def extract_urls_from_abstract(abstract: str) -> dict:
    """Extract GitHub and HuggingFace URLs from abstract text."""
    github_urls = []
    hf_urls = []

    if not abstract:
        return {"github_urls": [], "hf_urls": []}

    # GitHub URLs
    for m in re.finditer(
        r'https?://github\.com/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+', abstract
    ):
        url = m.group(0).rstrip('.),;:')
        if url not in github_urls:
            github_urls.append(url)

    # HuggingFace URLs (huggingface.co or hf.co)
    for m in re.finditer(
        r'https?://(?:huggingface\.co|hf\.co)/[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-\.]+)?',
        abstract,
    ):
        url = m.group(0).rstrip('.),;:')
        if url not in hf_urls:
            hf_urls.append(url)

    return {"github_urls": github_urls, "hf_urls": hf_urls}


# ---------------------------------------------------------------------------
# 2. Extract number of models evaluated
# ---------------------------------------------------------------------------

# Patterns that indicate number of models evaluated
MODEL_COUNT_PATTERNS = [
    # "we evaluate/test/benchmark N models/LLMs"
    r'(?:evaluat|test|benchmark|assess|compar)\w*\s+(\d+)\s+(?:model|llm|system|method|approach)',
    # "N models/LLMs are evaluated/tested"
    r'(\d+)\s+(?:model|llm|system|method|approach)\w*\s+(?:are|were|is)\s+(?:evaluat|test|benchmark|compar)',
    # "across/on/over N models/LLMs"
    r'(?:across|on|over|with)\s+(\d+)\s+(?:model|llm|system|method|approach|ai\s+system)',
    # "N state-of-the-art/leading/popular models"
    r'(\d+)\s+(?:state-of-the-art|leading|popular|prominent|mainstream|representative|open-source|closed-source|commercial|frontier)\s+(?:model|llm)',
    # "N LLMs" (standalone, common pattern)
    r'(\d+)\s+(?:large\s+language\s+)?(?:llm|model|system)s?\b',
    # "results on N models"
    r'(?:results?\s+(?:on|from|across|over))\s+(\d+)\s+(?:model|llm)',
    # "including GPT-4, Claude, ... (N models)"
    r'\((\d+)\s+(?:model|llm)s?\s*(?:in\s+total)?\)',
    # "comprehensive evaluation of N"
    r'(?:comprehensive|extensive|systematic|thorough)\s+(?:evaluation|comparison|benchmark\w*)\s+(?:of|on|across|over)\s+(\d+)',
]

# Model name patterns to count individually mentioned models
KNOWN_MODEL_PATTERNS = [
    r'\bGPT-?[34]\w*\b', r'\bGPT-?4[o]?\b', r'\bGPT-?5\b',
    r'\bClaude[\s\-]?\d?\b', r'\bClaude[\s\-]?3\.?5?\b',
    r'\bGemini[\s\-]?\w*\b', r'\bGemma[\s\-]?\w*\b',
    r'\bLlama[\s\-]?\d?\b', r'\bLLaMA[\s\-]?\d?\b',
    r'\bMistral\b', r'\bMixtral\b',
    r'\bQwen\d?\b', r'\bDeepSeek\w*\b',
    r'\bPaLM\w*\b', r'\bCommand[\s\-]?R\b',
    r'\bPhi[\s\-]?\d\b', r'\bStarCoder\w*\b',
    r'\bCodeLlama\b', r'\bWizardCoder\b',
    r'\bVicuna\b', r'\bAlpaca\b', r'\bZephyr\b',
    r'\bYi[\s\-]?\d\b', r'\bInternLM\b', r'\bBaichuan\b',
    r'\bFalcon\b', r'\bo[13][\s\-](?:mini|preview)?\b',
]


def extract_model_count(abstract: str) -> dict:
    """Extract the number of models evaluated from abstract text."""
    if not abstract:
        return {"model_count": None, "model_count_source": "none"}

    text = abstract.lower()

    # Try explicit count patterns first
    for pattern in MODEL_COUNT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            counts = [int(m) for m in matches if m.isdigit() and 2 <= int(m) <= 200]
            if counts:
                return {
                    "model_count": max(counts),
                    "model_count_source": "explicit_mention",
                }

    # Fall back to counting individually named models
    found_models = set()
    for pattern in KNOWN_MODEL_PATTERNS:
        for m in re.finditer(pattern, abstract, re.IGNORECASE):
            found_models.add(m.group(0).lower().strip())

    if len(found_models) >= 2:
        return {
            "model_count": len(found_models),
            "model_count_source": "named_models",
            "named_models": sorted(found_models),
        }

    return {"model_count": None, "model_count_source": "not_found"}


# ---------------------------------------------------------------------------
# 3. Check if response/output data is available
# ---------------------------------------------------------------------------

RESPONSE_DATA_POSITIVE = [
    r'(?:model\s+)?(?:response|output|prediction|generation|answer|result)s?\s+'
    r'(?:are|is|will\s+be|have\s+been)\s+(?:available|released|provided|shared|open)',
    r'(?:releas|provid|shar|publish|open.sourc)\w*\s+(?:the\s+)?'
    r'(?:model\s+)?(?:response|output|prediction|generation|answer|result)s?',
    r'(?:response|output|prediction)s?\s+(?:data|dataset|dump)',
    r'all\s+(?:model\s+)?(?:outputs?|responses?|predictions?|generations?)\s+'
    r'(?:are|is|will\s+be)\s+(?:available|public)',
    r'(?:we|our)\s+(?:also\s+)?(?:release|provide|share|publish|open.source)\s+'
    r'(?:all\s+)?(?:the\s+)?(?:model\s+)?(?:outputs?|responses?|predictions?|results?|generations?)',
    r'including\s+(?:model\s+)?(?:outputs?|responses?|predictions?)\s+'
    r'(?:at|on|from)\s+(?:https?://|github|huggingface)',
]

RESPONSE_DATA_SIGNALS = [
    r'(?:response|output|prediction|generation)s?\s+(?:are|is)\s+(?:publicly\s+)?available',
    r'leaderboard',
    r'open.source.*(?:data|code|model\s+output)',
    r'(?:data|code|benchmark).*available\s+at',
    r'reproducib',
]


def check_response_data(abstract: str) -> dict:
    """Check if paper mentions availability of model response/output data."""
    if not abstract:
        return {"response_data": "unknown", "response_data_evidence": ""}

    text_lower = abstract.lower()

    # Strong positive signals
    for pattern in RESPONSE_DATA_POSITIVE:
        m = re.search(pattern, text_lower)
        if m:
            return {
                "response_data": "yes",
                "response_data_evidence": m.group(0).strip(),
            }

    # Weaker signals
    signals_found = []
    for pattern in RESPONSE_DATA_SIGNALS:
        m = re.search(pattern, text_lower)
        if m:
            signals_found.append(m.group(0).strip())

    if len(signals_found) >= 2:
        return {
            "response_data": "likely",
            "response_data_evidence": "; ".join(signals_found),
        }
    if signals_found:
        return {
            "response_data": "unclear",
            "response_data_evidence": "; ".join(signals_found),
        }

    return {"response_data": "unknown", "response_data_evidence": ""}


# ---------------------------------------------------------------------------
# 4. (Optional) Search GitHub/HuggingFace APIs
# ---------------------------------------------------------------------------

def search_github(query: str, max_retries=2) -> list[dict]:
    """Search GitHub for repositories matching query."""
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "per_page": 3}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 403:
                log.warning("GitHub API rate limited. Stopping API search.")
                return []
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "name": item["full_name"],
                        "url": item["html_url"],
                        "stars": item["stargazers_count"],
                        "description": (item.get("description") or "")[:200],
                    }
                    for item in data.get("items", [])[:3]
                ]
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return []


def search_huggingface(query: str, max_retries=2) -> list[dict]:
    """Search HuggingFace for datasets matching query."""
    url = "https://huggingface.co/api/datasets"
    params = {"search": query, "limit": 3, "sort": "downloads", "direction": -1}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "id": item.get("id", ""),
                        "url": f"https://huggingface.co/datasets/{item.get('id', '')}",
                        "downloads": item.get("downloads", 0),
                    }
                    for item in data[:3]
                ]
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return []


def extract_benchmark_name(title: str) -> str | None:
    """Extract a short benchmark name from the title for API search."""
    skip = {"the", "how", "can", "are", "why", "do", "is", "on", "an", "a", "we",
            "from", "for", "and", "towards", "beyond", "when", "what", "all"}

    # Pattern: "BenchName: description" or "BenchName -- description"
    m = re.match(
        r'^[\$\\]?([A-Z][A-Za-z0-9_\-\.]+(?:\s*[A-Z][A-Za-z0-9_\-]*)?)\s*[:–\-—]\s+',
        title,
    )
    if m:
        name = m.group(1).strip()
        if len(name) >= 3 and name.lower() not in skip:
            return name

    # Pattern: acronym/name with benchmark-like suffix
    m = re.search(
        r'\b([A-Z][A-Za-z0-9]*(?:[-_]?(?:Bench|Eval|Mark|Test|Suite|QA|Dataset|Arena|Gym|Lab|Hub))\w*)\b',
        title,
    )
    if m:
        return m.group(1)

    # All-caps or CamelCase name at start: "MMLU", "BigBench", etc.
    m = re.match(r'^([A-Z]{2,}[A-Za-z0-9\-]*)\b', title)
    if m:
        name = m.group(1)
        if len(name) >= 3 and name not in ("THE", "HOW", "CAN", "ARE"):
            return name

    # Quoted name: "BenchName" in title
    m = re.search(r'"([A-Z][A-Za-z0-9_\-]+)"', title)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_paper(paper: dict) -> dict:
    """Enrich a single paper with availability, model count, and response data."""
    abstract = paper.get("abstract", "")
    title = paper.get("title", "")

    # 1. URL extraction from abstract
    url_info = extract_urls_from_abstract(abstract)
    paper["github_urls"] = url_info["github_urls"]
    paper["hf_urls"] = url_info["hf_urls"]
    paper["has_github"] = len(url_info["github_urls"]) > 0
    paper["has_huggingface"] = len(url_info["hf_urls"]) > 0
    paper["benchmark_available_online"] = (
        paper["has_github"] or paper["has_huggingface"]
    )

    # 2. Model count
    model_info = extract_model_count(abstract)
    paper["model_count"] = model_info["model_count"]
    paper["model_count_source"] = model_info["model_count_source"]
    if "named_models" in model_info:
        paper["named_models"] = model_info["named_models"]

    # 3. Response data availability
    resp_info = check_response_data(abstract)
    paper["response_data_available"] = resp_info["response_data"]
    paper["response_data_evidence"] = resp_info["response_data_evidence"]

    return paper


def enrich_with_api_search(papers: list[dict]) -> list[dict]:
    """Search GitHub/HuggingFace APIs for papers without URLs in abstract."""
    papers_without_urls = [
        p for p in papers if not p.get("benchmark_available_online")
    ]
    log.info(
        "Searching APIs for %d papers without URLs in abstract...",
        len(papers_without_urls),
    )

    github_rate_limited = False
    searched = 0

    for paper in tqdm(papers_without_urls, desc="API search"):
        name = extract_benchmark_name(
            paper.get("full_title", paper.get("title", ""))
        )
        if not name:
            continue

        # Search GitHub
        if not github_rate_limited:
            gh_results = search_github(f"{name} benchmark")
            if not gh_results and not github_rate_limited:
                # Might be rate limited, or no results
                pass
            else:
                # Check if any result is a plausible match
                for r in gh_results:
                    desc = r.get("description", "").lower()
                    repo_name = r.get("name", "").lower()
                    name_lower = name.lower()
                    if name_lower in repo_name or name_lower in desc:
                        paper["github_urls"] = [r["url"]]
                        paper["has_github"] = True
                        paper["benchmark_available_online"] = True
                        paper["github_stars"] = r["stars"]
                        break
            time.sleep(2.5)  # GitHub: 10 req/min unauthenticated
            searched += 1
            if searched % 10 == 0:
                # Check if we're being rate-limited
                test = requests.get(
                    "https://api.github.com/rate_limit", timeout=10
                )
                if test.status_code == 200:
                    remaining = test.json().get("resources", {}).get(
                        "search", {}
                    ).get("remaining", 0)
                    if remaining < 3:
                        log.warning("GitHub API rate limit nearly exhausted. Stopping.")
                        github_rate_limited = True

        # Search HuggingFace
        hf_results = search_huggingface(name)
        for r in hf_results:
            hf_id = r.get("id", "").lower()
            name_lower = name.lower()
            if name_lower in hf_id:
                paper["hf_urls"] = [r["url"]]
                paper["has_huggingface"] = True
                paper["benchmark_available_online"] = True
                paper["hf_downloads"] = r.get("downloads", 0)
                break
        time.sleep(0.5)

    return papers


def main():
    parser = argparse.ArgumentParser(
        description="Enrich benchmark papers with availability and model count info."
    )
    parser.add_argument(
        "--input",
        default="./benchmark_papers/benchmark_list.json",
        help="Input JSON file (default: benchmark_list.json)",
    )
    parser.add_argument(
        "--full_data",
        default="./benchmark_papers/benchmarks.json",
        help="Full crawl data for abstract lookup (default: benchmarks.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="./benchmark_papers",
        help="Output directory",
    )
    parser.add_argument(
        "--search_apis",
        action="store_true",
        help="Also search GitHub/HuggingFace APIs for papers without URLs",
    )
    args = parser.parse_args()

    # Load benchmark list
    with open(args.input) as f:
        benchmarks = json.load(f)
    log.info("Loaded %d benchmarks from %s", len(benchmarks), args.input)

    # Load full data to get complete abstracts
    full_abstracts = {}
    if os.path.exists(args.full_data):
        with open(args.full_data) as f:
            full_data = json.load(f)
        for p in full_data:
            key = re.sub(r"[^a-z0-9]", "", p.get("title", "").lower())
            full_abstracts[key] = p.get("abstract", "")
        log.info("Loaded %d full abstracts", len(full_abstracts))

    # Merge full abstracts into benchmark list
    for b in benchmarks:
        key = re.sub(r"[^a-z0-9]", "", b.get("full_title", b.get("title", "")).lower())
        if key in full_abstracts and len(full_abstracts[key]) > len(
            b.get("abstract", b.get("abstract_snippet", ""))
        ):
            b["abstract"] = full_abstracts[key]
        elif "abstract_snippet" in b and "abstract" not in b:
            b["abstract"] = b["abstract_snippet"]

    # Enrich each paper
    log.info("Enriching %d benchmarks...", len(benchmarks))
    for b in tqdm(benchmarks, desc="Enriching"):
        enrich_paper(b)

    # Optional: API search
    if args.search_apis:
        benchmarks = enrich_with_api_search(benchmarks)

    # Stats
    has_github = sum(1 for b in benchmarks if b.get("has_github"))
    has_hf = sum(1 for b in benchmarks if b.get("has_huggingface"))
    has_any = sum(1 for b in benchmarks if b.get("benchmark_available_online"))
    has_model_count = sum(1 for b in benchmarks if b.get("model_count"))
    resp_yes = sum(1 for b in benchmarks if b.get("response_data_available") == "yes")
    resp_likely = sum(
        1 for b in benchmarks if b.get("response_data_available") == "likely"
    )

    log.info("\n=== Enrichment Summary ===")
    log.info("Total benchmarks: %d", len(benchmarks))
    log.info("With GitHub URL: %d (%.0f%%)", has_github, 100 * has_github / len(benchmarks))
    log.info("With HuggingFace URL: %d (%.0f%%)", has_hf, 100 * has_hf / len(benchmarks))
    log.info(
        "Available online (either): %d (%.0f%%)",
        has_any, 100 * has_any / len(benchmarks),
    )
    log.info(
        "Model count extracted: %d (%.0f%%)",
        has_model_count, 100 * has_model_count / len(benchmarks),
    )
    log.info("Response data available: %d yes, %d likely", resp_yes, resp_likely)

    # Save enriched JSON
    output_json = os.path.join(args.output_dir, "benchmarks_enriched.json")
    with open(output_json, "w") as f:
        json.dump(benchmarks, f, indent=2, default=str)
    log.info("Saved: %s", output_json)

    # Save enriched CSV
    output_csv = os.path.join(args.output_dir, "benchmarks_enriched.csv")
    rows = []
    for b in benchmarks:
        rows.append({
            "name": b.get("name", ""),
            "full_title": b.get("full_title", b.get("title", "")),
            "year": b.get("year", ""),
            "url": b.get("url", ""),
            "github_urls": "; ".join(b.get("github_urls", [])),
            "hf_urls": "; ".join(b.get("hf_urls", [])),
            "benchmark_available_online": b.get("benchmark_available_online", False),
            "model_count": b.get("model_count", ""),
            "model_count_source": b.get("model_count_source", ""),
            "named_models": "; ".join(b.get("named_models", [])),
            "response_data_available": b.get("response_data_available", ""),
            "response_data_evidence": b.get("response_data_evidence", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    log.info("Saved: %s", output_csv)

    # Generate readable summary
    _generate_summary(benchmarks, args.output_dir)


def _categorize(title, abstract):
    """Categorize a benchmark by domain."""
    text = f"{title} {abstract}".lower()
    rules = [
        ("Reasoning & Math", [r"\breason", r"\bmath", r"\barithmetic", r"\blogic"]),
        ("Code & Programming", [r"\bcode", r"\bprogram", r"\bsoftware", r"\bcoding"]),
        ("Multimodal & Vision-Language", [r"\bmultimodal", r"\bvision.language", r"\bvisual", r"\bimage", r"\bvideo", r"\bchart"]),
        ("Safety & Alignment", [r"\bsafety", r"\balignment", r"\bjailbreak", r"\btoxic", r"\bbias", r"\bharmful"]),
        ("Multilingual", [r"\bmultilingual", r"\blow.resource", r"\bcross.lingual", r"\btranslation"]),
        ("Agents & Tool Use", [r"\bagent", r"\btool.use", r"\bweb.brows", r"\bgui"]),
        ("Medical & Bio", [r"\bmedic", r"\bclinic", r"\bhealth", r"\bbiomedic", r"\bdrug"]),
        ("Knowledge & QA", [r"\bknowledge", r"\bquestion.answer", r"\bqa\b", r"\bfactual"]),
        ("Hallucination", [r"\bhallucin", r"\bfaithful", r"\bfact.check"]),
        ("RAG & Retrieval", [r"\bretrieval.augmented", r"\brag\b", r"\blong.context"]),
        ("Science & Domain", [r"\bscien", r"\bchemist", r"\bphysic", r"\bfinanc", r"\blegal"]),
        ("Audio & Speech", [r"\baudio", r"\bspeech", r"\bvoice", r"\bmusic"]),
        ("Evaluation & Meta", [r"\bjudge", r"\bevaluator", r"\bllm.as.judge"]),
    ]
    for cat_name, patterns in rules:
        if any(re.search(p, text) for p in patterns):
            return cat_name
    return "Other"


def _generate_summary(benchmarks, output_dir):
    """Generate a readable summary file."""
    from collections import defaultdict

    cat_benchmarks = defaultdict(list)
    for b in benchmarks:
        title = b.get("full_title", b.get("title", ""))
        abstract = b.get("abstract", "")
        cat = _categorize(title, abstract)
        cat_benchmarks[cat].append(b)

    lines = []
    lines.append("# Recently Proposed LM Benchmarks (2024-2026) — Enriched")
    lines.append(f"# Total: {len(benchmarks)} unique benchmarks")
    lines.append("#")
    has_any = sum(1 for b in benchmarks if b.get("benchmark_available_online"))
    has_mc = sum(1 for b in benchmarks if b.get("model_count"))
    resp_yes = sum(1 for b in benchmarks if b.get("response_data_available") == "yes")
    lines.append(f"# Available online (GitHub/HF): {has_any} / {len(benchmarks)}")
    lines.append(f"# Model count extracted: {has_mc} / {len(benchmarks)}")
    lines.append(f"# Response data available: {resp_yes} confirmed")
    lines.append("")
    lines.append("# Legend:")
    lines.append("#   [GH] = GitHub    [HF] = HuggingFace    [--] = not found in abstract")
    lines.append("#   Models: N = number of models evaluated  (? = not found)")
    lines.append("#   RespData: Y = yes  L = likely  ? = unclear  - = unknown")
    lines.append("")

    sorted_cats = sorted(cat_benchmarks.items(), key=lambda x: -len(x[1]))

    for cat, papers in sorted_cats:
        lines.append(f"\n{'='*100}")
        lines.append(f"## {cat} ({len(papers)} benchmarks)")
        lines.append(f"{'='*100}")

        papers_sorted = sorted(papers, key=lambda x: (-x.get("year", 0), x.get("name", "")))
        for p in papers_sorted:
            name = p.get("name", "")[:40]
            year = p.get("year", "?")
            title = p.get("full_title", p.get("title", ""))[:65]

            # Availability tag
            avail_tags = []
            if p.get("has_github"):
                avail_tags.append("GH")
            if p.get("has_huggingface"):
                avail_tags.append("HF")
            avail_str = ",".join(avail_tags) if avail_tags else "--"

            # Model count
            mc = p.get("model_count")
            mc_str = str(mc) if mc else "?"

            # Response data
            rd = p.get("response_data_available", "unknown")
            rd_map = {"yes": "Y", "likely": "L", "unclear": "?", "unknown": "-"}
            rd_str = rd_map.get(rd, "-")

            lines.append(
                f"  [{year}] [{avail_str:5s}] Models:{mc_str:>3s}  Resp:{rd_str}"
                f"  {name:40s} | {title}"
            )

            # Print URLs if available
            gh = p.get("github_urls", [])
            hf = p.get("hf_urls", [])
            if gh or hf:
                urls = gh + hf
                lines.append(f"        -> {urls[0]}")

    out_path = os.path.join(output_dir, "benchmark_summary_enriched.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    log.info("Saved summary: %s", out_path)


if __name__ == "__main__":
    main()
