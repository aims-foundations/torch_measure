# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Central dataset registry."""

from __future__ import annotations

from dataclasses import replace

from torch_measure.datasets._info import DatasetInfo
from torch_measure.datasets._manifest import manifest_dataset_names, manifest_info
from torch_measure.datasets.agentic import _register_agentic_datasets
from torch_measure.datasets.arena import _register_arena_datasets
from torch_measure.datasets.arena_hard import _register_arena_hard_datasets
from torch_measure.datasets.beavertails import _register_beavertails_datasets
from torch_measure.datasets.bench import _register_bench_datasets
from torch_measure.datasets.biggen import _register_biggen_datasets
from torch_measure.datasets.flask import _register_flask_datasets
from torch_measure.datasets.helm import _register_helm_datasets
from torch_measure.datasets.helpsteer2 import _register_helpsteer2_datasets
from torch_measure.datasets.hh_rlhf import _register_hh_rlhf_datasets
from torch_measure.datasets.indeterminacy import _register_indeterminacy_datasets
from torch_measure.datasets.intervention import _register_intervention_datasets
from torch_measure.datasets.judgebench import _register_judgebench_datasets
from torch_measure.datasets.metr import _register_metr_datasets
from torch_measure.datasets.mtbench import _register_mtbench_datasets
from torch_measure.datasets.nectar import _register_nectar_datasets
from torch_measure.datasets.oasst import _register_oasst_datasets
from torch_measure.datasets.openllm import _register_openllm_datasets
from torch_measure.datasets.personalllm import _register_personalllm_datasets
from torch_measure.datasets.pickapic import _register_pickapic_datasets
from torch_measure.datasets.pku_saferlhf import _register_pku_saferlhf_datasets
from torch_measure.datasets.preference_dissection import _register_preference_dissection_datasets
from torch_measure.datasets.prism import _register_prism_datasets
from torch_measure.datasets.prm800k import _register_prm800k_datasets
from torch_measure.datasets.prometheus import _register_prometheus_datasets
from torch_measure.datasets.rewardbench import _register_rewardbench_datasets
from torch_measure.datasets.rewardbench2 import _register_rewardbench2_datasets
from torch_measure.datasets.safety import _register_safety_datasets
from torch_measure.datasets.shp2 import _register_shp2_datasets
from torch_measure.datasets.summeval import _register_summeval_datasets
from torch_measure.datasets.ultrafeedback import _register_ultrafeedback_datasets
from torch_measure.datasets.vl_rewardbench import _register_vl_rewardbench_datasets
from torch_measure.datasets.wmt_mqm import _register_wmt_mqm_datasets

# ---------------------------------------------------------------------------
# Global registry: name -> DatasetInfo
# ---------------------------------------------------------------------------


def _canonical_name(entry: DatasetInfo, fallback_name: str) -> str:
    """Return the public dataset name in ``family/name`` form."""
    name = entry.name or fallback_name
    if "/" in name:
        return name
    return f"{entry.family}/{name}"


def _namespace_entries(entries: dict[str, DatasetInfo]) -> dict[str, DatasetInfo]:
    """Normalize a family registry to canonical ``family/name`` keys."""
    normalized: dict[str, DatasetInfo] = {}
    for key, entry in entries.items():
        canonical_name = _canonical_name(entry, key)
        normalized[canonical_name] = replace(entry, name=canonical_name)
    return normalized


_REGISTRY: dict[str, DatasetInfo] = {}
for family_entries in (
    _register_helm_datasets(),
    _register_openllm_datasets(),
    _register_arena_datasets(),
    _register_agentic_datasets(),
    _register_metr_datasets(),
    _register_bench_datasets(),
    _register_biggen_datasets(),
    _register_mtbench_datasets(),
    _register_nectar_datasets(),
    _register_oasst_datasets(),
    _register_arena_hard_datasets(),
    _register_helpsteer2_datasets(),
    _register_indeterminacy_datasets(),
    _register_hh_rlhf_datasets(),
    _register_preference_dissection_datasets(),
    _register_rewardbench_datasets(),
    _register_rewardbench2_datasets(),
    _register_shp2_datasets(),
    _register_ultrafeedback_datasets(),
    _register_beavertails_datasets(),
    _register_flask_datasets(),
    _register_judgebench_datasets(),
    _register_personalllm_datasets(),
    _register_pickapic_datasets(),
    _register_pku_saferlhf_datasets(),
    _register_prism_datasets(),
    _register_prm800k_datasets(),
    _register_prometheus_datasets(),
    _register_summeval_datasets(),
    _register_vl_rewardbench_datasets(),
    _register_wmt_mqm_datasets(),
    _register_intervention_datasets(),
    _register_safety_datasets(),
):
    _REGISTRY.update(_namespace_entries(family_entries))

_alias_candidates: dict[str, list[str]] = {}
for canonical_name in _REGISTRY:
    _, bare_name = canonical_name.split("/", 1)
    _alias_candidates.setdefault(bare_name, []).append(canonical_name)


def _matching_dataset_names(name: str) -> list[str]:
    """Return every canonical dataset name matching bare ``name``."""
    matches = set(_alias_candidates.get(name, ()))
    matches.update(
        canonical_name for canonical_name in manifest_dataset_names() if canonical_name.split("/", 1)[1] == name
    )
    return sorted(matches)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_datasets(family: str | None = None) -> list[str]:
    """List all available dataset names.

    Datasets come from two sources:

    1. The **hardcoded registry** — baked into the library at install
       time. This covers the stable public API names and keeps lookups
       offline-safe.
    2. The **manifest** — ``manifest.json`` fetched from the HuggingFace
       Hub bucket managed by the ``measurement-db`` data-curation
       repository. This augments the built-in registry with datasets
       uploaded after the library was released.

    Both sources are unioned so that new manifest-only datasets become
    discoverable without dropping the stable, baked-in names.

    Parameters
    ----------
    family : str | None
        If provided, filter to only datasets in this family. Family
        matching is case-sensitive and uses the canonical family stored
        in each dataset's metadata.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of dataset names.
    """
    if family is not None:
        names = {name for name, entry in _REGISTRY.items() if entry.family == family}
        names.update(manifest_dataset_names(family=family))
        return sorted(names)
    names = set(_REGISTRY)
    names.update(manifest_dataset_names())
    return sorted(names)


def info(name: str) -> DatasetInfo:
    """Get metadata about a dataset without downloading it.

    Looks up ``name`` in the hardcoded registry first, then falls back to
    the manifest for datasets uploaded after the current library release.

    Parameters
    ----------
    name : str
        Dataset name in canonical ``family/name`` form. Bare names are
        accepted only when they are unambiguous across the registry and
        manifest.

    Returns
    -------
    DatasetInfo

    Raises
    ------
    ValueError
        If the dataset name is not found in either source.
    """
    # 1. Prefer the baked-in registry for stable, offline-safe lookups.
    if name in _REGISTRY:
        return _REGISTRY[name]

    if "/" not in name:
        matches = _matching_dataset_names(name)
        if len(matches) == 1:
            canonical_name = matches[0]
            if canonical_name in _REGISTRY:
                return _REGISTRY[canonical_name]
            entry = manifest_info(canonical_name)
            if entry is not None:
                return entry
        if len(matches) > 1:
            raise ValueError(f"Ambiguous dataset name: {name!r}. Use one of: {', '.join(matches)}")

    # 2. Fall back to the dynamic manifest for new datasets.
    entry = manifest_info(name)
    if entry is not None:
        return entry

    available = sorted(set(_REGISTRY) | set(manifest_dataset_names()))
    raise ValueError(f"Unknown dataset: {name!r}. Available datasets: {', '.join(available)}")
