# Copyright (c) 2026 AIMS Foundations. MIT License.

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

MANIFEST_REPO = "aims-foundations/measurement-db"
MANIFEST_FILENAME = "manifest.json"

# Process-local cache. Use ``load_manifest(force_download=True)`` to refresh.
_manifest_cache: dict[str, Any] | None = None


def _canonical_dataset_name(name: str, family: str) -> str:
    """Return the public dataset name in ``family/name`` form."""
    if "/" in name:
        return name
    return f"{family}/{name}"


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
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
    if "/" in name:
        # Slash-qualified manifest keys already encode the public family.
        canonical_name = name
        family = name.split("/", 1)[0]
    else:
        # Use the first tag as the `family` (loose convention matching the
        # old per-family registry). Fallback to "misc" if there are no tags.
        family = info.get("family") or (tags[0] if tags else "misc")
        canonical_name = _canonical_dataset_name(name, family)
    base_name = canonical_name.split("/", 1)[-1]
    return DatasetInfo(
        name=canonical_name,
        family=family,
        description=info.get("description", ""),
        response_type=entry.get("response_type", "binary"),
        n_subjects=int(entry.get("n_subjects", 0)),
        n_items=int(entry.get("n_items", 0)),
        subject_entity=info.get("subject_type", "LLM"),
        item_entity=info.get("item_type", "question"),
        repo_id=MANIFEST_REPO,
        filename=entry.get("filename", f"{base_name}.pt"),
        citation=info.get("citation", ""),
        url=info.get("paper_url") or info.get("data_source_url", ""),
        license=info.get("license", ""),
        n_comparisons=int(entry.get("n_comparisons", 0)),
        tags=tags,
    )


def _manifest_dataset_infos() -> dict[str, DatasetInfo]:
    """Return canonical manifest entries keyed by public dataset name."""
    manifest = load_manifest()
    if manifest is None:
        return {}

    datasets = manifest.get("datasets") or {}
    infos: dict[str, DatasetInfo] = {}
    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            continue
        dataset_info = manifest_to_info(name, entry)
        infos[dataset_info.name] = dataset_info
    return infos


def manifest_info(name: str) -> DatasetInfo | None:
    """Return :class:`DatasetInfo` for ``name`` from the manifest.

    Canonical ``family/name`` lookups are matched directly. Bare names
    only resolve when they are unambiguous after canonicalizing manifest
    entries.
    """
    infos = _manifest_dataset_infos()
    if "/" in name:
        return infos.get(name)

    matches = [
        dataset_info for canonical_name, dataset_info in infos.items() if canonical_name.split("/", 1)[1] == name
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def manifest_dataset_names(family: str | None = None) -> list[str]:
    """Return the sorted list of dataset names in the manifest.

    Returns an empty list if the manifest is unavailable.
    """
    infos = _manifest_dataset_infos()
    if family is not None:
        return sorted(name for name, entry in infos.items() if entry.family == family)
    return sorted(infos)
