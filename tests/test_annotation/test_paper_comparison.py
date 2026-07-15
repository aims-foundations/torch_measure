# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Compare annotator output against the paper's annotations from ADeLe battery.

Fetches 15 diverse items from the HuggingFace ADeLe battery dataset, runs the
annotator on each, and reports score agreement + statistical analysis against
the paper's reference annotations across all 19 dimensions.

Items are sampled one per benchmark/task-type for diversity:
    molecule_captioning, name_prediction, reaction_prediction, retrosynthesis,
    LogiQA-en, Date Arithmetic, LSAT-AR, MCTACO, MMLU-Pro x3,
    MedCalcBench, OmniMath, SciBench, TimeQA

Requires:
    HF_TOKEN            — HuggingFace token (dataset is gated)
    ANNOTATOR_CLIENT    — 'gemini' (default), 'claude', or 'openai'
    GEMINI_API_KEY      — Gemini API key      (when ANNOTATOR_CLIENT=gemini)
    GEMINI_MODEL        — model string        (default: gemini-3.1-flash-lite)
    ANTHROPIC_API_KEY   — Anthropic API key   (when ANNOTATOR_CLIENT=claude)
    CLAUDE_MODEL        — model string        (default: claude-opus-4-8)
    OPENAI_API_KEY      — OpenAI API key      (when ANNOTATOR_CLIENT=openai)
    OPENAI_MODEL        — model string        (default: gpt-4o)

Usage (Gemini — writes to paper_comparison_cache.jsonl):
    $env:HF_TOKEN = "<token>"
    $env:GEMINI_API_KEY = "<key>"
    python -m pytest tests/test_annotation/test_paper_comparison.py -v -s -m "network and slow"

Usage (Claude — writes to claude_opus_4_8_comparison_cache.jsonl):
    $env:HF_TOKEN = "<token>"
    $env:ANNOTATOR_CLIENT = "claude"
    $env:ANTHROPIC_API_KEY = "<key>"
    python -m pytest tests/test_annotation/test_paper_comparison.py -v -s -m "network and slow"

Usage (OpenAI — writes to gpt_4o_comparison_cache.jsonl):
    $env:HF_TOKEN = "<token>"
    $env:ANNOTATOR_CLIENT = "openai"
    $env:OPENAI_API_KEY = "<key>"
    python -m pytest tests/test_annotation/test_paper_comparison.py -v -s -m "network and slow"

Cost: 15 items x 19 calls = 285 API calls.
"""

import json
import math
import os
import time
import urllib.request
from statistics import mean, stdev

import pytest

from torch_measure.annotation import (
    AnnotationCache,
    AnnotationJob,
    ClaudeClient,
    DemandAnnotator,
    GeminiClient,
    OpenAIClient,
    RubricsCatalog,
)
from torch_measure.annotation._types import DEMAND_DIMENSIONS, DIMENSION_ORDER

pytestmark = [pytest.mark.network, pytest.mark.slow]

_HF_BASE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=CFI-Kinds-of-Intelligence%2FADeLe_battery_v1dot0"
    "&config=default&split=train&length=1&offset={offset}"
)

_OFFSETS = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 7000, 9000, 10000, 11000, 13000, 15000]

_N_ITEMS = len(_OFFSETS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        pytest.skip("HF_TOKEN environment variable not set")
    return token


@pytest.fixture(scope="module")
def client_type():
    return os.environ.get("ANNOTATOR_CLIENT", "gemini").strip().lower()


@pytest.fixture(scope="module")
def api_key(client_type):
    if client_type == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    elif client_type == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
    else:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            pytest.skip("GEMINI_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="module")
def model_id(client_type):
    if client_type == "claude":
        return os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")
    if client_type == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


@pytest.fixture(scope="module")
def paper_rows(hf_token):
    """Fetch one row per offset from the ADeLe battery dataset."""
    rows = []
    for i, offset in enumerate(_OFFSETS):
        url = _HF_BASE.format(offset=offset)
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + hf_token})
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        rows.append(data["rows"][0]["row"])
        if len(rows) < len(_OFFSETS):
            time.sleep(0.5)
    print(f"\nFetched {len(rows)} items from ADeLe battery")
    return rows


def _cache_name(client_type: str, model_id: str) -> str:
    if client_type == "gemini":
        return "paper_comparison_cache.jsonl"
    safe = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"{safe}_comparison_cache.jsonl"


@pytest.fixture(scope="module")
def annotator(api_key, model_id, client_type):
    import pathlib
    if client_type == "claude":
        client = ClaudeClient(api_key=api_key, model=model_id)
    elif client_type == "openai":
        client = OpenAIClient(api_key=api_key, model=model_id)
    else:
        client = GeminiClient(api_key=api_key, model=model_id)
    rubrics = RubricsCatalog()
    cache_path = pathlib.Path(__file__).parent / _cache_name(client_type, model_id)
    cache = AnnotationCache(cache_path)
    print(f"\nClient: {client_type}  Model: {model_id}")
    print(f"Cache:  {cache_path}")
    return DemandAnnotator(client=client, rubrics=rubrics, cache=cache)


@pytest.fixture(scope="module")
def comparison_results(paper_rows, annotator):
    """Annotate all 15 items and pair with paper scores.

    Failures on individual items are caught and skipped — partial runs
    still produce results for completed items. The persistent cache means
    re-runs resume from where the previous run left off at zero extra cost.
    """
    results = []
    skipped = []
    print(f"\n{'─'*60}")
    print(f"Annotating {_N_ITEMS} items — 19 API calls each")
    print(f"{'─'*60}")
    for i, row in enumerate(paper_rows):
        job = AnnotationJob(
            item_id=str(row["instance_id"]),
            content=row["question"],
            reference_answer=row["groundtruth"],
        )
        print(f"  [{i+1:2d}/{_N_ITEMS}] {row['benchmark']} / {row['task']} ...", end="", flush=True)
        try:
            annotation = annotator.annotate(job)
            vector = annotation.to_feature_vector()
            n_nan = sum(math.isnan(v) for v in vector)
            status = f" done  (NaN: {n_nan})" if n_nan > 0 else " done ✓"
            results.append({
                "item_id": job.item_id,
                "benchmark": row["benchmark"],
                "task": row["task"],
                "paper": {dim: float(row[dim]) for dim in DIMENSION_ORDER},
                "ours": vector,
            })
        except Exception as exc:
            status = f" FAILED — {type(exc).__name__}: {exc}"
            skipped.append(job.item_id)
        print(status)
    print(f"Completed: {len(results)}  |  Skipped: {len(skipped)}")
    if skipped:
        print(f"Skipped items (re-run to retry from cache): {skipped}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_pair(p: float, o: float) -> bool:
    return not math.isnan(p) and not math.isnan(o)


def _diff(p: float, o: float) -> float:
    return o - p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchAndAnnotate:

    def test_all_items_fetched(self, paper_rows):
        assert len(paper_rows) == _N_ITEMS

    def test_all_items_annotated(self, comparison_results):
        assert len(comparison_results) > 0, "No items were annotated — check API key and quota"
        r = len(comparison_results)
        if r < _N_ITEMS:
            pytest.xfail(
                f"Only {r} items completed. Re-run to finish — completed items are cached at zero cost."
            )

    def test_no_parse_failures(self, comparison_results):
        failures = []
        for r in comparison_results:
            nan_dims = [DIMENSION_ORDER[i] for i, v in enumerate(r["ours"]) if math.isnan(v)]
            if nan_dims:
                failures.append(f"  {r['item_id']}: {nan_dims}")
        assert not failures, "Parse failures (NaN scores):\n" + "\n".join(failures)


class TestItemComparison:

    def test_print_per_item_scores(self, comparison_results):
        """Print each item's paper vs our scores side-by-side. 0 extra API calls."""
        for r in comparison_results:
            print(f"\n{'─'*72}")
            print(f"Item : {r['item_id']}")
            print(f"Bench: {r['benchmark']} / {r['task']}")
            print(f"{'Dim':<8} {'Paper':>6} {'Ours':>6} {'Diff':>6}  Verdict")
            print(f"{'─'*50}")
            for i, dim in enumerate(DIMENSION_ORDER):
                p = r["paper"][dim]
                o = r["ours"][i]
                if math.isnan(p):
                    print(f"{dim:<8} {'N/A':>6} {o:>6.1f}        (no paper score)")
                    continue
                if math.isnan(o):
                    print(f"{dim:<8} {p:>6.1f} {'N/A':>6}        (parse failure)")
                    continue
                d = _diff(p, o)
                if abs(d) == 0:
                    verdict = "exact"
                elif abs(d) <= 1:
                    verdict = "~±1"
                else:
                    verdict = f"OFF {d:+.0f}"
                print(f"{dim:<8} {p:>6.1f} {o:>6.1f} {d:>+6.1f}  {verdict}")


class TestStatistics:

    def test_print_statistical_summary(self, comparison_results):
        """Full statistical breakdown: per-dimension and overall. 0 extra API calls."""

        def pearson(pairs: list) -> float:
            if len(pairs) < 2:
                return float("nan")
            xs = [float(p) for p, _ in pairs]
            ys = [float(o) for _, o in pairs]
            mx, my = mean(xs), mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
            return float("nan") if den == 0 else num / den

        def spearman(pairs: list) -> float:
            if len(pairs) < 2:
                return float("nan")

            def ranks(lst):
                sorted_idx = sorted(range(len(lst)), key=lambda i: lst[i])
                r = [0.0] * len(lst)
                for rank, idx in enumerate(sorted_idx):
                    r[idx] = float(rank)
                return r

            xs = [float(p) for p, _ in pairs]
            ys = [float(o) for _, o in pairs]
            rx = ranks(xs)
            ry = ranks(ys)
            return pearson(list(zip(rx, ry)))

        dim_pairs: dict[str, list] = {d: [] for d in DIMENSION_ORDER}
        for r in comparison_results:
            for i, dim in enumerate(DIMENSION_ORDER):
                p = r["paper"][dim]
                o = r["ours"][i]
                if _valid_pair(p, o):
                    dim_pairs[dim].append((p, o))

        all_pairs = [po for pairs in dim_pairs.values() for po in pairs]
        n_total = len(all_pairs)
        all_diffs = [_diff(p, o) for p, o in all_pairs]
        abs_diffs = [abs(d) for d in all_diffs]
        exact = sum(d == 0 for d in all_diffs)
        within_1 = sum(abs(d) <= 1 for d in all_diffs)

        r_pearson = pearson(all_pairs)
        r_spearman = spearman(all_pairs)
        mae = mean(abs_diffs)
        bias = mean(all_diffs)
        sd = stdev(all_diffs) if len(all_diffs) > 1 else float("nan")

        print(f"\n{'='*72}")
        print(f"OVERALL STATISTICS  ({_N_ITEMS} items × 19 dims = {n_total} scored pairs)")
        print(f"  MAE (mean |ours − paper|)  : {mae:.3f}")
        print(
            f"  Bias (mean ours − paper)   : {bias:+.3f}  "
            f"({'our scores higher' if bias > 0 else 'our scores lower' if bias < 0 else 'no bias'})"
        )
        print(f"  Std dev of differences     : {sd:.3f}")
        print(f"  Pearson r                  : {r_pearson:.3f}")
        print(f"  Spearman ρ                 : {r_spearman:.3f}")
        print(f"  Exact match                : {exact}/{n_total}  ({100*exact/n_total:.1f}%)")
        print(f"  Within ±1                  : {within_1}/{n_total}  ({100*within_1/n_total:.1f}%)")
        print(f"  Off by >1                  : {n_total - within_1}/{n_total}  ({100*(n_total - within_1)/n_total:.1f}%)")

        print(f"\n{'─'*72}")
        print(f"PER-DIMENSION BREAKDOWN")
        print(f"{'Dim':<8} {'MAE':>6} {'Bias':>7} {'Exact%':>7} {'±1%':>6}  Agreement")
        print(f"{'─'*60}")

        dim_stats = []
        for dim in DIMENSION_ORDER:
            pairs = dim_pairs[dim]
            if not pairs:
                continue
            diffs = [_diff(p, o) for p, o in pairs]
            adiffs = [abs(d) for d in diffs]
            d_mae = mean(adiffs)
            d_bias = mean(diffs)
            d_exact = sum(d == 0 for d in diffs)
            d_w1 = sum(abs(d) <= 1 for d in diffs)
            n = len(pairs)
            bar = "█" * int(10 * d_w1 / n)
            print(
                f"{dim:<8} {d_mae:>6.2f} {d_bias:>+7.2f} {100*d_exact/n:>7.1f}% "
                f"{100*d_w1/n:>6.1f}%  {bar}"
            )
            dim_stats.append((dim, d_mae))

        dim_stats.sort(key=lambda x: x[1])
        best = ", ".join(d for d, _ in dim_stats[:3])
        worst = ", ".join(d for d, _ in reversed(dim_stats[-3:]))
        print(f"\nMost agreed dimensions  (lowest MAE): {best}")
        print(f"Least agreed dimensions (highest MAE): {worst}")
