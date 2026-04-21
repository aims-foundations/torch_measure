# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Manifest-based dataset discovery.

Fetches ``manifest.json`` from the HuggingFace Hub bucket managed by the
`measurement-db <https://github.com/aims-foundations/measurement-db>`_
data-curation repository. The manifest lists every available ``.pt``
file along with its shape, response type, and metadata (description,
paper URL, license, full BibTeX citation, etc.).

This replaces the old hardcoded per-family registry so new datasets
uploaded by the data pipeline become loadable via
:func:`torch_measure.datasets.load` immediately, without requiring a
library release.

The hardcoded registry in :mod:`torch_measure.datasets._registry` is
kept as an offline fallback. When the network is available, the manifest
is the source of truth; when it isn't (air-gapped environments, CI
without HF access), the hardcoded registry still works for all legacy
dataset names.
"""
from __future__ import annotations

import json
from typing import Any

from torch_measure.datasets._info import DatasetInfo

MANIFEST_REPO = "aims-foundation/torch-measure-data"
MANIFEST_FILENAME = "manifest.json"

# Process-local cache. Use ``load_manifest(force_download=True)`` to refresh.
_manifest_cache: dict[str, Any] | None = None


def load_manifest(*, force_download: bool = False) -> dict[str, Any] | None:
    """Fetch ``manifest.json`` from HuggingFace Hub (cached).

    Parameters
    ----------
    force_download : bool
        If ``True``, bypass the process-local cache and re-fetch from HF.
        HuggingFace Hub itself uses ETag caching for the underlying file,
        so re-fetching is cheap when the manifest hasn't changed.

    Returns
    -------
    dict[str, Any] | None
        The parsed manifest as a dict, or ``None`` if the fetch failed
        (e.g., no network, ``huggingface_hub`` not installed). Callers
        should fall back to the hardcoded registry when ``None`` is
        returned.
    """
    global _manifest_cache
    if _manifest_cache is not None and not force_download:
        return _manifest_cache

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        path = hf_hub_download(
            repo_id=MANIFEST_REPO,
            filename=MANIFEST_FILENAME,
            repo_type="dataset",
            force_download=force_download,
        )
    except Exception:
        # Network failure, 404 (manifest not yet uploaded), auth issue, ...
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "datasets" not in data:
        return None

    _manifest_cache = data
    return _manifest_cache


def manifest_to_info(name: str, entry: dict[str, Any]) -> DatasetInfo:
    """Convert a manifest entry to a :class:`DatasetInfo`.

    The manifest schema nests most metadata under ``entry["info"]`` —
    the same dict that's baked into each ``.pt`` payload's ``info``
    field by the data pipeline. This function flattens it into the
    library's :class:`DatasetInfo` dataclass.
    """
    info = entry.get("info") or {}
    tags = list(info.get("tags") or [])
    # Use the first tag as the `family` (loose convention matching the
    # old per-family registry). Fallback to "misc" if there are no tags.
    family = tags[0] if tags else "misc"
    return DatasetInfo(
        name=name,
        family=family,
        description=info.get("description", ""),
        response_type=entry.get("response_type", "binary"),
        n_subjects=int(entry.get("n_subjects", 0)),
        n_items=int(entry.get("n_items", 0)),
        subject_entity=info.get("subject_type", "LLM"),
        item_entity=info.get("item_type", "question"),
        repo_id=MANIFEST_REPO,
        filename=entry.get("filename", f"{name}.pt"),
        citation=info.get("citation", ""),
        url=info.get("paper_url") or info.get("data_source_url", ""),
        license=info.get("license", ""),
        n_comparisons=int(entry.get("n_comparisons", 0)),
        tags=tags,
    )


def manifest_info(name: str) -> DatasetInfo | None:
    """Return :class:`DatasetInfo` for ``name`` from the manifest.

    Returns ``None`` if the manifest is unavailable or ``name`` is not
    in it. Callers should fall back to the hardcoded registry on ``None``.
    """
    manifest = load_manifest()
    if manifest is None:
        return None
    datasets = manifest.get("datasets") or {}
    entry = datasets.get(name)
    if not isinstance(entry, dict):
        return None
    return manifest_to_info(name, entry)


def manifest_dataset_names() -> list[str]:
    """Return the sorted list of dataset names in the manifest.

    Returns an empty list if the manifest is unavailable.
    """
    manifest = load_manifest()
    if manifest is None:
        return []
    return sorted(manifest.get("datasets") or {})
