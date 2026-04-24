# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Dataset discovery against the measurement-db long-form HF bucket.

The data-curation repo
(`measurement-db <https://github.com/aims-foundations/measurement-db>`_)
publishes a flat layout on HuggingFace Hub:

  - ``<dataset>.parquet``       — long-form responses for one dataset
  - ``<dataset>_traces.parquet`` — optional long-form traces for that dataset
  - ``benchmarks.parquet``      — registry row per dataset
  - ``subjects.parquet``        — registry row per subject across datasets
  - ``items.parquet``           — registry row per item across datasets

This module fetches ``benchmarks.parquet`` to enumerate available datasets
and to populate :class:`DatasetInfo` without having to download each dataset.
``benchmarks.parquet`` is the single source of truth — there is no
hardcoded fallback.
"""
from __future__ import annotations

from typing import Any

from torch_measure.datasets._info import DatasetInfo

MANIFEST_REPO = "aims-foundations/measurement-db"
BENCHMARKS_FILENAME = "benchmarks.parquet"

_manifest_cache: dict[str, dict[str, Any]] | None = None


def _fetch_parquet(filename: str, *, force_download: bool = False):
    """Download a parquet file from the manifest repo; return a DataFrame or None."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return None

    try:
        path = hf_hub_download(
            repo_id=MANIFEST_REPO,
            filename=filename,
            repo_type="dataset",
            force_download=force_download,
        )
    except Exception:
        return None

    try:
        import pandas as pd

        return pd.read_parquet(path)
    except Exception:
        return None


def load_manifest(*, force_download: bool = False) -> dict[str, dict[str, Any]] | None:
    """Return ``{benchmark_id: benchmark_row_dict}`` from HF, or None on failure."""
    global _manifest_cache
    if _manifest_cache is not None and not force_download:
        return _manifest_cache

    df = _fetch_parquet(BENCHMARKS_FILENAME, force_download=force_download)
    if df is None or df.empty or "benchmark_id" not in df.columns:
        return None

    _manifest_cache = {
        str(row["benchmark_id"]): {k: row[k] for k in df.columns}
        for _, row in df.iterrows()
    }
    return _manifest_cache


def _coerce_str(v) -> str:
    if v is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(v):
            return ""
    except (ImportError, ValueError, TypeError):
        pass
    return str(v)


def _coerce_bool(v, default: bool = True) -> bool:
    if v is None:
        return default
    try:
        import pandas as pd

        if isinstance(v, float) and pd.isna(v):
            return default
    except ImportError:
        pass
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", "", "nan"):
        return False
    return default


def _coerce_list(v) -> list[str]:
    if v is None:
        return []
    # numpy arrays / pandas sequences
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return [str(x) for x in v.tolist() if x is not None and str(x) != "nan"]
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None and str(x) != "nan"]
    s = _coerce_str(v)
    if not s:
        return []
    # Common encodings: comma-separated or JSON-ish
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def manifest_to_info(name: str, entry: dict[str, Any]) -> DatasetInfo:
    """Convert a benchmarks.parquet row to a :class:`DatasetInfo`.

    Shape-related fields (``n_subjects``, ``n_items``) are left at 0 here —
    they're filled in by the loader after the actual dataset is downloaded,
    so the manifest fetch stays cheap.
    """
    modality = _coerce_list(entry.get("modality"))
    domain = _coerce_list(entry.get("domain"))
    source_url = _coerce_str(entry.get("source_url"))
    family = domain[0] if domain else "misc"
    tags = sorted(set(modality) | set(domain))

    return DatasetInfo(
        name=name,
        description=_coerce_str(entry.get("description")),
        response_type=_coerce_str(entry.get("response_type")) or "binary",
        response_scale=_coerce_str(entry.get("response_scale")),
        categorical=_coerce_bool(entry.get("categorical"), default=True),
        modality=modality,
        domain=domain,
        release_date=_coerce_str(entry.get("release_date")),
        paper_url=_coerce_str(entry.get("paper_url")),
        license=_coerce_str(entry.get("license")),
        source_url=source_url,
        version=_coerce_str(entry.get("version")),
        repo_id=MANIFEST_REPO,
        filename=f"{name}.parquet",
        citation="",
        tags=tags,
        family=family,
        url=source_url,
    )


def manifest_info(name: str) -> DatasetInfo | None:
    manifest = load_manifest()
    if manifest is None:
        return None
    entry = manifest.get(name)
    if not isinstance(entry, dict):
        return None
    return manifest_to_info(name, entry)


def manifest_entry(name: str) -> dict[str, Any] | None:
    """Return the raw benchmarks-registry row for ``name``, or ``None``.

    Used by the loader to expose the full row (with all columns preserved
    as-is) on :attr:`LongFormData.info`, without going through the
    :class:`DatasetInfo` normalization.
    """
    manifest = load_manifest()
    if manifest is None:
        return None
    entry = manifest.get(name)
    if not isinstance(entry, dict):
        return None
    return dict(entry)


def manifest_dataset_names() -> list[str]:
    manifest = load_manifest()
    if manifest is None:
        return []
    return sorted(manifest.keys())
