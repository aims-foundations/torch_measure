#!/usr/bin/env python3
"""
Regenerate data/GALLERY.md from the tracked response_matrix*.png files.

Walks data/<benchmark>/processed/ for each benchmark in BENCHMARKS and
BENCHMARKS_AGGREGATE, collects the PNG heatmaps, and writes a grouped
markdown file with embedded image links (relative to data/).

Usage:
    python data/scripts/build_gallery.py
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GALLERY_PATH = BASE_DIR / "GALLERY.md"


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


def render_section(title: str, pngs_by_bench: dict[str, list[Path]]) -> list[str]:
    """Render a markdown section with a grid of heatmaps, one group per benchmark.

    Uses HTML `<img>` tags with a fixed width to keep the gallery compact.
    """
    lines = [f"## {title}", ""]
    lines.append(
        f"_{len(pngs_by_bench)} benchmarks, "
        f"{sum(len(v) for v in pngs_by_bench.values())} heatmaps_"
    )
    lines.append("")

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


def main():
    source = (BASE_DIR / "reproduce.py").read_text()
    benchmarks = _parse_list(source, "BENCHMARKS")
    aggregate = _parse_list(source, "BENCHMARKS_AGGREGATE")

    ready_pngs = collect_pngs(benchmarks)
    aggregate_pngs = collect_pngs(aggregate)

    n_ready_pngs = sum(len(v) for v in ready_pngs.values())
    n_agg_pngs = sum(len(v) for v in aggregate_pngs.values())

    lines = [
        "# Benchmark Gallery",
        "",
        f"Response matrix heatmaps for all tracked benchmarks — "
        f"**{len(ready_pngs) + len(aggregate_pngs)} benchmarks**, "
        f"**{n_ready_pngs + n_agg_pngs} heatmaps** total.",
        "",
        "Each image shows the full response matrix: rows are subjects (usually "
        "models, sorted by mean score), columns are items (sorted by difficulty), "
        "colored by score (red=low, green=high). Matrices larger than "
        "1000 × 2000 are downsampled for render performance.",
        "",
        "To regenerate the heatmaps themselves, run the relevant `build.py` or "
        "`python data/scripts/visualize_response_matrix.py`. "
        "To regenerate this file, run `python data/scripts/build_gallery.py`.",
        "",
        "---",
        "",
    ]

    if ready_pngs:
        lines.extend(render_section(
            f"BENCHMARKS — per-item response matrices ({len(ready_pngs)})",
            ready_pngs,
        ))
        lines.append("---")
        lines.append("")

    if aggregate_pngs:
        lines.extend(render_section(
            f"BENCHMARKS_AGGREGATE — aggregate-only ({len(aggregate_pngs)})",
            aggregate_pngs,
        ))

    GALLERY_PATH.write_text("\n".join(lines))
    print(f"Wrote {GALLERY_PATH}")
    print(f"  BENCHMARKS: {len(ready_pngs)} benchmarks, {n_ready_pngs} heatmaps")
    print(f"  BENCHMARKS_AGGREGATE: {len(aggregate_pngs)} benchmarks, {n_agg_pngs} heatmaps")


if __name__ == "__main__":
    main()
