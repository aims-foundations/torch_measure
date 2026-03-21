# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Central dataset registry."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo
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
from torch_measure.datasets.ultrafeedback import _register_ultrafeedback_datasets

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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_datasets(family: str | None = None) -> list[str]:
    """List all available dataset names.

    Parameters
    ----------
    family : str | None
        If provided, filter to only datasets in this family (e.g., ``"helm"``).

    Returns
    -------
    list[str]
        Sorted list of dataset names.
    """
    if family is not None:
        return sorted(name for name, entry in _REGISTRY.items() if entry.family == family)
    return sorted(_REGISTRY)


def info(name: str) -> DatasetInfo:
    """Get metadata about a dataset without downloading it.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"helm/mmlu"``).

    Returns
    -------
    DatasetInfo

    Raises
    ------
    ValueError
        If the dataset name is not found in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown dataset: {name!r}. Available datasets: {available}")
    return _REGISTRY[name]
