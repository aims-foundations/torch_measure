#!/usr/bin/env python3
"""
Regenerate the benchmark gallery files from tracked response_matrix*.png files.

Produces two markdown files under data/:
  - GALLERY.md           — per-item BENCHMARKS (ready for IRT)
  - GALLERY_AGGREGATE.md — aggregate-only BENCHMARKS_AGGREGATE

(BENCHMARKS_PENDING has no heatmaps, so no file for it.)

Usage:
    python data/scripts/build_gallery.py
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GALLERY_PATH = BASE_DIR / "GALLERY.md"
GALLERY_AGGREGATE_PATH = BASE_DIR / "GALLERY_AGGREGATE.md"


def _parse_list(source: str, list_name: str) -> list[str]:
    """Extract a benchmark name list from reproduce.py."""
    pattern = rf"{list_name}\s*=\s*\[(.*?)\]\s*\n\s*(?:#|\Z)"
    m = re.search(pattern, source, re.DOTALL)
    if not m:
        return []
    return re.findall(r'"(\w+)"', m.group(1))


def collect_pngs(benchmark_names: list[str]) -> dict[str, list[Path]]:
    """Return {benchmark: [png_paths]} for benchmarks with heatmaps."""
    result = {}
    for bench in sorted(benchmark_names):
        pngs = sorted((BASE_DIR / bench / "processed").glob("response_matrix*.png"))
        if pngs:
            result[bench] = pngs
    return result


def _variant_label(png_path: Path) -> str:
    """Return the variant suffix (or empty string for primary)."""
    stem = png_path.stem  # e.g. "response_matrix_instruct"
    return stem.removeprefix("response_matrix").lstrip("_")


IMG_WIDTH = 360  # pixels — tune for readability vs density


def render_gallery_body(pngs_by_bench: dict[str, list[Path]]) -> list[str]:
    """Render the benchmark sections as markdown with embedded thumbnail images."""
    lines = []
    for bench in sorted(pngs_by_bench):
        lines.append(f"### {bench}")
        lines.append("")
        for png in pngs_by_bench[bench]:
            variant = _variant_label(png)
            caption = f"{bench} / {variant}" if variant else bench
            rel = png.relative_to(BASE_DIR)
            lines.append(
                f'<img src="{rel}" alt="{caption}" width="{IMG_WIDTH}">'
            )
            if variant:
                lines.append(f"_{caption}_")
            lines.append("")
    return lines


def write_gallery(
    path: Path,
    title: str,
    description: str,
    pngs_by_bench: dict[str, list[Path]],
) -> None:
    """Write a self-contained gallery markdown file."""
    n_benchmarks = len(pngs_by_bench)
    n_heatmaps = sum(len(v) for v in pngs_by_bench.values())

    lines = [
        f"# {title}",
        "",
        f"{description}",
        "",
        f"**{n_benchmarks} benchmarks**, **{n_heatmaps} heatmaps**.",
        "",
        "Each image shows the full response matrix: rows are subjects (usually "
        "models, sorted by mean score), columns are items (sorted by difficulty), "
        "colored by score (red=low, green=high). Matrices larger than "
        "1000 × 2000 are downsampled for render performance.",
        "",
        "To regenerate the heatmaps, run the relevant `build.py` or "
        "`python data/scripts/visualize_response_matrix.py`. "
        "To regenerate this file, run `python data/scripts/build_gallery.py`.",
        "",
        "---",
        "",
    ]
    lines.extend(render_gallery_body(pngs_by_bench))
    path.write_text("\n".join(lines))
    print(f"Wrote {path}  ({n_benchmarks} benchmarks, {n_heatmaps} heatmaps)")


def main():
    source = (BASE_DIR / "reproduce.py").read_text()
    benchmarks = _parse_list(source, "BENCHMARKS")
    aggregate = _parse_list(source, "BENCHMARKS_AGGREGATE")

    ready_pngs = collect_pngs(benchmarks)
    aggregate_pngs = collect_pngs(aggregate)

    write_gallery(
        GALLERY_PATH,
        title="Benchmark Gallery — Per-Item Response Matrices",
        description=(
            "Heatmaps for all benchmarks in `BENCHMARKS` (ready for IRT / "
            "psychometric analysis). Each cell is a single subject's response "
            "to a single item."
        ),
        pngs_by_bench=ready_pngs,
    )

    write_gallery(
        GALLERY_AGGREGATE_PATH,
        title="Benchmark Gallery — Aggregate-Only Benchmarks",
        description=(
            "Heatmaps for benchmarks in `BENCHMARKS_AGGREGATE`. These have "
            "multi-model data but the cells are aggregate rates across "
            "trials, conditions, or sub-benchmarks — **not** per-item "
            "responses. Useful for model-level comparisons but not IRT-ready. "
            "See [`GALLERY.md`](GALLERY.md) for per-item benchmarks."
        ),
        pngs_by_bench=aggregate_pngs,
    )


if __name__ == "__main__":
    main()
