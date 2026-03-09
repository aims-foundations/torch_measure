#!/usr/bin/env python3
"""
Crawl papers from ArXiv, Semantic Scholar, and OpenReview to discover
recently proposed benchmarks for language models.

Usage:
    python crawl_benchmarks.py                          # Full crawl
    python crawl_benchmarks.py --dry_run                # Show queries without fetching
    python crawl_benchmarks.py --max_results_per_source 50  # Quick test
    python crawl_benchmarks.py --skip_openreview        # Skip OpenReview source

Output: benchmark_papers/ directory with benchmarks.csv and benchmarks.json
"""

import argparse
import json
import os
import re
import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlencode

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# A paper must match at least one primary keyword (benchmark-related)
PRIMARY_KEYWORDS = [
    "benchmark",
    "benchmarking",
    "evaluation benchmark",
    "evaluation suite",
    "test suite",
    "test bed",
    "testbed",
    "evaluation framework",
    "evaluation harness",
    "leaderboard",
    "dataset for evaluating",
    "datasets for evaluating",
    "evaluation dataset",
    "evaluation datasets",
    "new dataset",
    "challenge dataset",
    "shared task",
]

# And at least one LM-relevance keyword
LM_KEYWORDS = [
    "language model",
    "llm",
    "large language model",
    "gpt",
    "foundation model",
    "transformer",
    "nlp",
    "natural language",
    "text generation",
    "chatbot",
    "instruction following",
    "reasoning",
    "code generation",
    "math reasoning",
    "multilingual",
    "question answering",
    "summarization",
    "machine translation",
    "dialogue",
    "commonsense",
    "knowledge",
    "hallucination",
    "safety",
    "alignment",
    "multimodal",
    "vision-language",
    "retrieval-augmented",
    "rag",
    "agent",
    "tool use",
]

# Conferences to search on Semantic Scholar
CONFERENCES = [
    "NeurIPS",
    "ICML",
    "ICLR",
    "ACL",
    "EMNLP",
    "NAACL",
    "AAAI",
    "EACL",
    "COLING",
    "CoNLL",
    "Findings of ACL",
    "Findings of EMNLP",
    "COLM",
]

# OpenReview venue IDs (venue_id format used in the API)
OPENREVIEW_VENUES = {
    "ICLR 2025": "ICLR.cc/2025/Conference",
    "ICLR 2024": "ICLR.cc/2024/Conference",
    "NeurIPS 2024": "NeurIPS.cc/2024/Conference",
    "NeurIPS 2025": "NeurIPS.cc/2025/Conference",
    "COLM 2024": "COLM.cc/2024/Conference",
    "COLM 2025": "COLM.cc/2025/Conference",
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _word_match(keyword: str, text: str) -> bool:
    """Check if keyword appears as a whole word/phrase in text (not as substring)."""
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def matches_keywords(text: str) -> tuple[bool, list[str]]:
    """Check if text matches both a primary benchmark keyword and an LM keyword.
    Uses word-boundary matching to avoid substring false positives.
    Returns (is_match, list_of_matched_keywords).
    """
    if not text:
        return False, []
    text_lower = text.lower()
    matched_primary = [k for k in PRIMARY_KEYWORDS if _word_match(k, text_lower)]
    matched_lm = [k for k in LM_KEYWORDS if _word_match(k, text_lower)]
    if matched_primary and matched_lm:
        return True, matched_primary + matched_lm
    return False, []


def normalize_title(title: str) -> str:
    """Normalize title for deduplication."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def safe_request(url, params=None, headers=None, max_retries=3, delay=3.0):
    """Make an HTTP GET request with retries and rate limiting."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                log.warning("Rate limited. Waiting %.1fs...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = delay * (attempt + 1)
                log.warning("Request failed (%s). Retrying in %.1fs...", e, wait)
                time.sleep(wait)
            else:
                log.error("Request failed after %d attempts: %s", max_retries, e)
                return None
    return None


# ---------------------------------------------------------------------------
# ArXiv Crawler
# ---------------------------------------------------------------------------

class ArxivCrawler:
    """Search ArXiv API for benchmark papers in ML/NLP categories."""

    BASE_URL = "http://export.arxiv.org/api/query"
    NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, start_year=2024, end_year=2026, max_results=2000):
        self.start_year = start_year
        self.end_year = end_year
        self.max_results = max_results

    def _build_queries(self) -> list[str]:
        """Build arxiv search queries combining benchmark + LM keywords."""
        categories = ["cs.CL", "cs.AI", "cs.LG"]
        cat_query = " OR ".join(f"cat:{c}" for c in categories)

        benchmark_terms = [
            "benchmark", "evaluation suite", "evaluation benchmark",
            "evaluation dataset", "evaluation framework", "testbed",
            "leaderboard", "test suite", "evaluation harness",
            "shared task", "challenge dataset",
        ]
        lm_terms = [
            "language model", "LLM", "large language model",
            "foundation model", "NLP", "natural language processing",
            "transformer", "reasoning benchmark", "code generation",
            "question answering", "multimodal",
        ]

        queries = []
        # Create combination queries to stay within API limits
        for bt in benchmark_terms:
            for lt in lm_terms[:4]:  # top LM terms
                q = f'({cat_query}) AND (ti:"{bt}" OR abs:"{bt}") AND (ti:"{lt}" OR abs:"{lt}")'
                queries.append(q)

        return queries

    def get_query_info(self) -> list[str]:
        """Return queries for dry run display."""
        return self._build_queries()

    def crawl(self) -> list[dict]:
        """Execute the crawl and return list of paper dicts."""
        queries = self._build_queries()
        all_papers = {}
        batch_size = 100
        max_per_query = min(500, self.max_results)

        log.info("ArXiv: Running %d queries...", len(queries))
        for query in tqdm(queries, desc="ArXiv queries"):
            start = 0
            while start < max_per_query:
                params = {
                    "search_query": query,
                    "start": start,
                    "max_results": min(batch_size, max_per_query - start),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
                resp = safe_request(self.BASE_URL, params=params)
                if resp is None:
                    break

                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", self.NS)
                if not entries:
                    break

                new_count = 0
                for entry in entries:
                    paper = self._parse_entry(entry)
                    if paper and paper["year"] >= self.start_year and paper["year"] <= self.end_year:
                        key = normalize_title(paper["title"])
                        if key not in all_papers:
                            all_papers[key] = paper
                            new_count += 1

                if new_count == 0 or len(entries) < batch_size:
                    break
                start += batch_size
                time.sleep(3)  # ArXiv rate limit

        log.info("ArXiv: Found %d unique papers.", len(all_papers))
        return list(all_papers.values())

    def _parse_entry(self, entry) -> dict | None:
        """Parse a single ArXiv Atom entry."""
        ns = self.NS
        title_el = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        if title_el is None or abstract_el is None:
            return None

        title = " ".join(title_el.text.strip().split())
        abstract = " ".join(abstract_el.text.strip().split())
        published = published_el.text.strip() if published_el is not None else ""

        try:
            year = int(published[:4])
        except (ValueError, IndexError):
            year = 0

        # Extract arxiv ID from the id URL
        id_el = entry.find("atom:id", ns)
        arxiv_url = id_el.text.strip() if id_el is not None else ""
        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None:
                authors.append(name_el.text.strip())

        # Categories
        categories = []
        for cat_el in entry.findall("arxiv:primary_category", ns):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        for cat_el in entry.findall("atom:category", ns):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)

        return {
            "title": title,
            "authors": "; ".join(authors),
            "venue": "ArXiv",
            "year": year,
            "abstract": abstract,
            "url": arxiv_url,
            "arxiv_id": arxiv_id,
            "source": "arxiv",
            "categories": ", ".join(categories),
            "citation_count": None,
            "published_date": published,
        }


# ---------------------------------------------------------------------------
# Semantic Scholar Crawler
# ---------------------------------------------------------------------------

class SemanticScholarCrawler:
    """Search Semantic Scholar API for benchmark papers from conferences."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, start_year=2024, end_year=2026, max_results=2000):
        self.start_year = start_year
        self.end_year = end_year
        self.max_results = max_results
        self.fields = "title,abstract,authors,venue,year,citationCount,externalIds,url,publicationDate"

    def get_query_info(self) -> list[dict]:
        """Return query info for dry run."""
        queries = []
        benchmark_search_terms = [
            "benchmark language model",
            "evaluation benchmark LLM",
            "benchmark NLP",
            "evaluation suite language model",
            "benchmark reasoning",
            "benchmark code generation",
            "benchmark multimodal",
            "leaderboard language model",
            "evaluation dataset NLP",
            "benchmark foundation model",
            "testbed language model",
            "benchmark alignment",
            "benchmark safety LLM",
            "benchmark hallucination",
            "benchmark agent",
        ]
        for term in benchmark_search_terms:
            queries.append({
                "endpoint": f"{self.BASE_URL}/paper/search",
                "query": term,
                "year": f"{self.start_year}-{self.end_year}",
            })
        return queries

    def crawl(self) -> list[dict]:
        """Execute crawl via Semantic Scholar paper search."""
        all_papers = {}
        search_terms = [q["query"] for q in self.get_query_info()]

        log.info("Semantic Scholar: Running %d search queries...", len(search_terms))

        for term in tqdm(search_terms, desc="S2 queries"):
            offset = 0
            per_query_limit = min(200, self.max_results)

            while offset < per_query_limit:
                batch = min(100, per_query_limit - offset)
                params = {
                    "query": term,
                    "year": f"{self.start_year}-{self.end_year}",
                    "fields": self.fields,
                    "offset": offset,
                    "limit": batch,
                }
                resp = safe_request(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                if resp is None:
                    break

                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    log.warning("S2: Invalid JSON response for query '%s'", term)
                    break

                papers = data.get("data", [])
                if not papers:
                    break

                for p in papers:
                    paper = self._parse_paper(p)
                    if paper:
                        key = normalize_title(paper["title"])
                        if key not in all_papers:
                            all_papers[key] = paper

                total = data.get("total", 0)
                offset += batch
                if offset >= total:
                    break

                time.sleep(1.0)  # Rate limit: ~100 req / 5 min

            if len(all_papers) >= self.max_results:
                break

        log.info("Semantic Scholar: Found %d unique papers.", len(all_papers))
        return list(all_papers.values())

    def _parse_paper(self, p: dict) -> dict | None:
        """Parse a single Semantic Scholar paper result."""
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        if not title:
            return None

        authors_list = p.get("authors", [])
        authors = "; ".join(a.get("name", "") for a in authors_list if a.get("name"))

        ext_ids = p.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv", "")

        venue = p.get("venue", "") or ""
        year = p.get("year") or 0

        url = p.get("url", "")
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        return {
            "title": title,
            "authors": authors,
            "venue": venue,
            "year": year,
            "abstract": abstract or "",
            "url": url,
            "arxiv_id": arxiv_id,
            "source": "semantic_scholar",
            "categories": "",
            "citation_count": p.get("citationCount"),
            "published_date": p.get("publicationDate", ""),
        }


# ---------------------------------------------------------------------------
# OpenReview Crawler
# ---------------------------------------------------------------------------

class OpenReviewCrawler:
    """Search OpenReview API for accepted benchmark papers at ICLR/NeurIPS."""

    BASE_URL = "https://api2.openreview.net"

    def __init__(self, start_year=2024, end_year=2026, max_results=2000):
        self.start_year = start_year
        self.end_year = end_year
        self.max_results = max_results

    def get_query_info(self) -> list[dict]:
        """Return venue info for dry run."""
        venues = []
        for name, venue_id in OPENREVIEW_VENUES.items():
            try:
                year = int(name.split()[-1])
            except ValueError:
                year = 0
            if self.start_year <= year <= self.end_year:
                venues.append({"venue_name": name, "venue_id": venue_id})
        return venues

    def crawl(self) -> list[dict]:
        """Crawl accepted papers from OpenReview venues."""
        all_papers = {}
        venues = self.get_query_info()

        log.info("OpenReview: Searching %d venues...", len(venues))

        for venue_info in tqdm(venues, desc="OpenReview venues"):
            venue_id = venue_info["venue_id"]
            venue_name = venue_info["venue_name"]
            offset = 0
            batch_size = 100

            while True:
                params = {
                    "content.venue": venue_name,
                    "invitation": f"{venue_id}/-/Submission",
                    "offset": offset,
                    "limit": batch_size,
                }
                resp = safe_request(
                    f"{self.BASE_URL}/notes",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                if resp is None:
                    # Try alternative invitation format
                    params["invitation"] = f"{venue_id}/-/Blind_Submission"
                    resp = safe_request(
                        f"{self.BASE_URL}/notes",
                        params=params,
                        headers={"Accept": "application/json"},
                    )
                    if resp is None:
                        break

                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    log.warning("OpenReview: Invalid JSON for %s", venue_name)
                    break

                notes = data.get("notes", [])
                if not notes:
                    break

                for note in notes:
                    paper = self._parse_note(note, venue_name)
                    if paper:
                        key = normalize_title(paper["title"])
                        if key not in all_papers:
                            all_papers[key] = paper

                offset += batch_size
                if len(notes) < batch_size:
                    break

                time.sleep(1.0)

            if len(all_papers) >= self.max_results:
                break

        log.info("OpenReview: Found %d unique papers.", len(all_papers))
        return list(all_papers.values())

    def _parse_note(self, note: dict, venue_name: str) -> dict | None:
        """Parse an OpenReview note into a paper dict."""
        content = note.get("content", {})

        # Handle both old and new OpenReview formats
        title = content.get("title", "")
        if isinstance(title, dict):
            title = title.get("value", "")
        abstract = content.get("abstract", "")
        if isinstance(abstract, dict):
            abstract = abstract.get("value", "")

        if not title:
            return None

        # Authors
        authors_field = content.get("authors", [])
        if isinstance(authors_field, dict):
            authors_field = authors_field.get("value", [])
        if isinstance(authors_field, list):
            authors = "; ".join(authors_field)
        else:
            authors = str(authors_field)

        # Year from venue name
        try:
            year = int(venue_name.split()[-1])
        except ValueError:
            year = 0

        forum_id = note.get("forum", note.get("id", ""))
        url = f"https://openreview.net/forum?id={forum_id}" if forum_id else ""

        return {
            "title": title.strip(),
            "authors": authors,
            "venue": venue_name,
            "year": year,
            "abstract": abstract.strip() if abstract else "",
            "url": url,
            "arxiv_id": "",
            "source": "openreview",
            "categories": "",
            "citation_count": None,
            "published_date": "",
        }


# ---------------------------------------------------------------------------
# Filter & Deduplication
# ---------------------------------------------------------------------------

class BenchmarkFilter:
    """Filter papers for benchmark relevance and deduplicate."""

    @staticmethod
    def filter_and_deduplicate(papers: list[dict]) -> list[dict]:
        """Apply keyword matching and deduplicate by normalized title."""
        seen = {}
        filtered = []

        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            is_match, matched_kw = matches_keywords(text)
            if not is_match:
                continue

            key = normalize_title(paper["title"])
            if key in seen:
                # Keep the version with more info (longer abstract, has citation count)
                existing = seen[key]
                if (len(paper.get("abstract", "")) > len(existing.get("abstract", ""))
                        or (paper.get("citation_count") is not None
                            and existing.get("citation_count") is None)):
                    paper["matched_keywords"] = ", ".join(matched_kw)
                    # Merge sources
                    existing_source = existing.get("source", "")
                    if paper["source"] not in existing_source:
                        paper["source"] = f"{existing_source}+{paper['source']}"
                    seen[key] = paper
                    # Update in filtered list
                    for i, f in enumerate(filtered):
                        if normalize_title(f["title"]) == key:
                            filtered[i] = paper
                            break
            else:
                paper["matched_keywords"] = ", ".join(matched_kw)
                seen[key] = paper
                filtered.append(paper)

        # Sort by year (newest first), then by citation count
        filtered.sort(
            key=lambda p: (
                -p.get("year", 0),
                -(p.get("citation_count") or 0),
            )
        )
        return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crawl ML/AI/NLP papers to find recently proposed LM benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output_dir", default="./benchmark_papers",
                        help="Directory for output files (default: ./benchmark_papers)")
    parser.add_argument("--start_year", type=int, default=2024,
                        help="Start year for papers (default: 2024)")
    parser.add_argument("--end_year", type=int, default=2026,
                        help="End year for papers (default: 2026)")
    parser.add_argument("--max_results_per_source", type=int, default=2000,
                        help="Max results per source (default: 2000)")
    parser.add_argument("--skip_arxiv", action="store_true",
                        help="Skip ArXiv crawling")
    parser.add_argument("--skip_s2", action="store_true",
                        help="Skip Semantic Scholar crawling")
    parser.add_argument("--skip_openreview", action="store_true",
                        help="Skip OpenReview crawling")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show queries without making requests")
    args = parser.parse_args()

    log.info("Benchmark Paper Crawler")
    log.info("Year range: %d-%d", args.start_year, args.end_year)
    log.info("Max results per source: %d", args.max_results_per_source)

    # Initialize crawlers
    arxiv = ArxivCrawler(args.start_year, args.end_year, args.max_results_per_source)
    s2 = SemanticScholarCrawler(args.start_year, args.end_year, args.max_results_per_source)
    openreview = OpenReviewCrawler(args.start_year, args.end_year, args.max_results_per_source)

    # Dry run mode
    if args.dry_run:
        log.info("=== DRY RUN MODE ===")
        if not args.skip_arxiv:
            queries = arxiv.get_query_info()
            log.info("\nArXiv: %d queries", len(queries))
            for i, q in enumerate(queries[:5]):
                log.info("  [%d] %s", i + 1, q[:120] + "..." if len(q) > 120 else q)
            if len(queries) > 5:
                log.info("  ... and %d more", len(queries) - 5)

        if not args.skip_s2:
            queries = s2.get_query_info()
            log.info("\nSemantic Scholar: %d queries", len(queries))
            for q in queries:
                log.info("  query='%s' year=%s", q["query"], q["year"])

        if not args.skip_openreview:
            venues = openreview.get_query_info()
            log.info("\nOpenReview: %d venues", len(venues))
            for v in venues:
                log.info("  %s (%s)", v["venue_name"], v["venue_id"])

        log.info("\nPrimary keywords: %s", PRIMARY_KEYWORDS)
        log.info("LM-relevance keywords: %s", LM_KEYWORDS)
        return

    # Crawl from each source
    all_papers = []

    if not args.skip_arxiv:
        log.info("\n--- ArXiv ---")
        all_papers.extend(arxiv.crawl())

    if not args.skip_s2:
        log.info("\n--- Semantic Scholar ---")
        all_papers.extend(s2.crawl())

    if not args.skip_openreview:
        log.info("\n--- OpenReview ---")
        all_papers.extend(openreview.crawl())

    log.info("\nTotal papers before filtering: %d", len(all_papers))

    # Filter and deduplicate
    filtered = BenchmarkFilter.filter_and_deduplicate(all_papers)
    log.info("Papers after keyword filtering + dedup: %d", len(filtered))

    if not filtered:
        log.warning("No benchmark papers found. Try broadening keywords or year range.")
        return

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(args.output_dir, "benchmarks.csv")
    df = pd.DataFrame(filtered)
    column_order = [
        "title", "authors", "venue", "year", "abstract", "url",
        "arxiv_id", "source", "matched_keywords", "categories",
        "citation_count", "published_date",
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV: %s (%d rows)", csv_path, len(df))

    # JSON
    json_path = os.path.join(args.output_dir, "benchmarks.json")
    with open(json_path, "w") as f:
        json.dump(filtered, f, indent=2, default=str)
    log.info("Saved JSON: %s", json_path)

    # Print summary
    log.info("\n=== Summary ===")
    log.info("Total benchmark papers found: %d", len(filtered))
    if "venue" in df.columns:
        venue_counts = df["venue"].value_counts().head(15)
        log.info("\nTop venues:")
        for venue, count in venue_counts.items():
            log.info("  %-30s %d", venue, count)
    if "year" in df.columns:
        year_counts = df["year"].value_counts().sort_index()
        log.info("\nBy year:")
        for year, count in year_counts.items():
            log.info("  %s: %d", year, count)
    if "citation_count" in df.columns:
        cited = df.dropna(subset=["citation_count"]).copy()
        cited["citation_count"] = pd.to_numeric(cited["citation_count"], errors="coerce")
        cited = cited.dropna(subset=["citation_count"])
        if not cited.empty:
            top_cited = cited.nlargest(10, "citation_count")
            log.info("\nTop 10 most cited:")
            for _, row in top_cited.iterrows():
                log.info("  [%s citations] %s",
                         int(row["citation_count"]), row["title"][:80])


if __name__ == "__main__":
    main()
