# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Dataset discovery backed by the measurement-db manifest.

No per-family Python modules live here anymore; the full dataset catalog is
``benchmarks.parquet`` on the measurement-db HuggingFace bucket (see
:mod:`torch_measure.datasets._manifest`). The helpers below are thin wrappers
around that manifest.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo
from torch_measure.datasets._manifest import (
    load_manifest,
    manifest_dataset_names,
    manifest_info,
    manifest_to_info,
)


def list_datasets(family: str | None = None) -> list[str]:
    """List all available dataset names, sourced from the manifest.

    Parameters
    ----------
    family : str | None
        If provided, filter to datasets whose ``family``, ``domain``, or
        ``modality`` contains this tag. ``family`` on a :class:`DatasetInfo`
        defaults to the first ``domain`` entry, so ``family="software_engineering"``
        picks up everything tagged as that domain. Matching is
        case-sensitive.

    Returns
    -------
    list[str]
        Sorted dataset names.
    """
    names = manifest_dataset_names()
    if family is None:
        return names
    manifest = load_manifest() or {}
    return sorted(n for n in names if _matches_family(n, family, manifest))


def _matches_family(name: str, family: str, manifest: dict) -> bool:
    entry = manifest.get(name)
    if entry is None:
        return False
    info_obj = manifest_to_info(name, entry)
    if info_obj.family == family:
        return True
    if family in info_obj.domain:
        return True
    if family in info_obj.modality:
        return True
    if family in info_obj.tags:
        return True
    return False


def info(name: str) -> DatasetInfo:
    """Get metadata about a dataset without downloading its responses.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"mtbench"``).

    Returns
    -------
    DatasetInfo

    Raises
    ------
    ValueError
        If the dataset name is not found in the manifest.
    """
    result = manifest_info(name)
    if result is not None:
        return result

    available = manifest_dataset_names()
    preview = ", ".join(available[:10])
    suffix = "..." if len(available) > 10 else ""
    raise ValueError(
        f"Unknown dataset: {name!r}. Available datasets: {preview}{suffix}"
    )
