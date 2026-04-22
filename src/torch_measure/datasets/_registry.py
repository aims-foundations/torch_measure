# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Central dataset registry."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo
from torch_measure.datasets._manifest import manifest_dataset_names, manifest_info
from torch_measure.datasets.agentic import _register_agentic_datasets
from torch_measure.datasets.arena import _register_arena_datasets
from torch_measure.datasets.arena_hard import _register_arena_hard_datasets
from torch_measure.datasets.bench import _register_bench_datasets
from torch_measure.datasets.biggen import _register_biggen_datasets
from torch_measure.datasets.helm import _register_helm_datasets
from torch_measure.datasets.helpsteer2 import _register_helpsteer2_datasets
from torch_measure.datasets.indeterminacy import _register_indeterminacy_datasets
from torch_measure.datasets.hh_rlhf import _register_hh_rlhf_datasets
from torch_measure.datasets.metr import _register_metr_datasets
from torch_measure.datasets.mtbench import _register_mtbench_datasets
from torch_measure.datasets.nectar import _register_nectar_datasets
from torch_measure.datasets.oasst import _register_oasst_datasets
from torch_measure.datasets.openllm import _register_openllm_datasets
from torch_measure.datasets.preference_dissection import _register_preference_dissection_datasets
from torch_measure.datasets.rewardbench import _register_rewardbench_datasets
from torch_measure.datasets.rewardbench2 import _register_rewardbench2_datasets
from torch_measure.datasets.shp2 import _register_shp2_datasets
from torch_measure.datasets.beavertails import _register_beavertails_datasets
from torch_measure.datasets.flask import _register_flask_datasets
from torch_measure.datasets.judgebench import _register_judgebench_datasets
from torch_measure.datasets.personalllm import _register_personalllm_datasets
from torch_measure.datasets.pickapic import _register_pickapic_datasets
from torch_measure.datasets.pku_saferlhf import _register_pku_saferlhf_datasets
from torch_measure.datasets.prism import _register_prism_datasets
from torch_measure.datasets.prm800k import _register_prm800k_datasets
from torch_measure.datasets.prometheus import _register_prometheus_datasets
from torch_measure.datasets.summeval import _register_summeval_datasets
from torch_measure.datasets.ultrafeedback import _register_ultrafeedback_datasets
from torch_measure.datasets.vl_rewardbench import _register_vl_rewardbench_datasets
from torch_measure.datasets.intervention import _register_intervention_datasets
from torch_measure.datasets.safety import _register_safety_datasets
from torch_measure.datasets.wmt_mqm import _register_wmt_mqm_datasets

# ---------------------------------------------------------------------------
# Global registry: name -> DatasetInfo
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DatasetInfo] = {}
_REGISTRY.update(_register_helm_datasets())
_REGISTRY.update(_register_openllm_datasets())
_REGISTRY.update(_register_arena_datasets())
_REGISTRY.update(_register_agentic_datasets())
_REGISTRY.update(_register_metr_datasets())
_REGISTRY.update(_register_bench_datasets())
_REGISTRY.update(_register_biggen_datasets())
_REGISTRY.update(_register_mtbench_datasets())
_REGISTRY.update(_register_nectar_datasets())
_REGISTRY.update(_register_oasst_datasets())
_REGISTRY.update(_register_arena_hard_datasets())
_REGISTRY.update(_register_helpsteer2_datasets())
_REGISTRY.update(_register_indeterminacy_datasets())
_REGISTRY.update(_register_hh_rlhf_datasets())
_REGISTRY.update(_register_preference_dissection_datasets())
_REGISTRY.update(_register_rewardbench_datasets())
_REGISTRY.update(_register_rewardbench2_datasets())
_REGISTRY.update(_register_shp2_datasets())
_REGISTRY.update(_register_ultrafeedback_datasets())
_REGISTRY.update(_register_beavertails_datasets())
_REGISTRY.update(_register_flask_datasets())
_REGISTRY.update(_register_judgebench_datasets())
_REGISTRY.update(_register_personalllm_datasets())
_REGISTRY.update(_register_pickapic_datasets())
_REGISTRY.update(_register_pku_saferlhf_datasets())
_REGISTRY.update(_register_prism_datasets())
_REGISTRY.update(_register_prm800k_datasets())
_REGISTRY.update(_register_prometheus_datasets())
_REGISTRY.update(_register_summeval_datasets())
_REGISTRY.update(_register_vl_rewardbench_datasets())
_REGISTRY.update(_register_wmt_mqm_datasets())
_REGISTRY.update(_register_intervention_datasets())
_REGISTRY.update(_register_safety_datasets())


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_datasets(family: str | None = None) -> list[str]:
    """List all available dataset names.

    Datasets come from two sources, in priority order:

    1. The **manifest** — ``manifest.json`` fetched from the HuggingFace
       Hub bucket managed by the ``measurement-db`` data-curation
       repository. When the manifest is reachable, this is the source of
       truth: it always reflects the latest uploaded ``.pt`` files.
    2. The **hardcoded registry** — baked into the library at install
       time. Used as an offline fallback for legacy dataset names when
       the manifest is unavailable (no network, ``huggingface_hub`` not
       installed, etc.).

    Both sources are unioned so that no dataset the user has seen before
    disappears on a library upgrade.

    Parameters
    ----------
    family : str | None
        If provided, filter to only datasets in this family. Family
        matching is case-sensitive and only considers the hardcoded
        registry (manifest entries use free-form tags, not families).

    Returns
    -------
    list[str]
        Sorted, deduplicated list of dataset names.
    """
    if family is not None:
        return sorted(
            name for name, entry in _REGISTRY.items() if entry.family == family
        )
    names = set(_REGISTRY)
    names.update(manifest_dataset_names())
    return sorted(names)


def info(name: str) -> DatasetInfo:
    """Get metadata about a dataset without downloading it.

    Looks up ``name`` in the manifest first (the current source of truth,
    fetched from HuggingFace Hub), then falls back to the hardcoded
    registry for offline use / legacy names.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"swebench"``).

    Returns
    -------
    DatasetInfo

    Raises
    ------
    ValueError
        If the dataset name is not found in either source.
    """
    # 1. Try the dynamic manifest (most up-to-date).
    entry = manifest_info(name)
    if entry is not None:
        return entry

    # 2. Fall back to the hardcoded registry (offline-safe).
    if name in _REGISTRY:
        return _REGISTRY[name]

    available = sorted(set(_REGISTRY) | set(manifest_dataset_names()))
    raise ValueError(
        f"Unknown dataset: {name!r}. Available datasets: {', '.join(available)}"
    )
