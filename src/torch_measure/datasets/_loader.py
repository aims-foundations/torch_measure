# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Long-form dataset loader for the measurement-db HF bucket.

``load(name)`` returns a :class:`LongFormData` containing the raw long-form
responses, the item + subject registries filtered to this benchmark, an
optional traces table, and the ``benchmarks.parquet`` row as a dict. Wide-form
:class:`~torch_measure.data.response_matrix.ResponseMatrix` is now opt-in via
:meth:`LongFormData.to_response_matrix`.
"""

from __future__ import annotations

from pathlib import Path

from torch_measure.datasets._long_form import LongFormData
from torch_measure.datasets._manifest import MANIFEST_REPO, manifest_entry


def load(
    name: str,
    *,
    force_download: bool = False,
    local_dir: str | Path | None = None,
) -> LongFormData:
    """Load a dataset by name from the measurement-db HF bucket.

    The bucket holds one parquet per dataset (long-form rows with columns
    ``subject_id, item_id, benchmark_id, trial, test_condition, response,
    correct_answer, trace``) at the root, plus shared registry files
    (``benchmarks.parquet``, ``subjects.parquet``, ``items.parquet``) also at
    the root, and optionally ``{name}_traces.parquet`` for benchmarks that
    published traces separately. This loader fetches all of these and wraps
    them in a :class:`LongFormData`.

    Parameters
    ----------
    name : str
        Dataset name (e.g., ``"mtbench"``, ``"swebench"``). Use
        :func:`list_datasets` to see available names.
    force_download : bool
        Bypass the HF cache and re-download the parquets.
    local_dir : str | Path | None
        If provided, load ``{local_dir}/{name}.parquet`` +
        ``{local_dir}/{subjects,items,benchmarks}.parquet`` from the local
        filesystem instead of downloading. Useful for the data-curation repo
        itself (``measurement-db`` / ``measurement-db-private``) where those
        parquets live at the root after ``merge_registry.py``.

    Returns
    -------
    LongFormData
        Container wrapping the long-form responses, benchmark-scoped item and
        subject registries, optional traces, and the benchmark-registry row.
    """
    responses, items, subjects, traces, info = _load_frames(
        name, force_download=force_download, local_dir=local_dir,
    )
    return LongFormData(
        name=name,
        responses=responses,
        items=items,
        subjects=subjects,
        traces=traces,
        info=info,
    )


def _load_frames(
    name: str,
    *,
    force_download: bool,
    local_dir: str | Path | None,
):
    """Return (responses, items_for_bench, subjects_for_bench, traces, info)."""
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError(
            "torch_measure.datasets.load requires pandas. "
            "Install with: pip install pandas"
        ) from err

    if local_dir is not None:
        root = Path(local_dir)
        responses = pd.read_parquet(root / f"{name}.parquet")
        items = pd.read_parquet(root / "items.parquet")
        subjects = pd.read_parquet(root / "subjects.parquet")
        traces_path = root / f"{name}_traces.parquet"
        traces = pd.read_parquet(traces_path) if traces_path.exists() else None
        benchmarks_path = root / "benchmarks.parquet"
        info: dict = {}
        if benchmarks_path.exists():
            benchmarks_df = pd.read_parquet(benchmarks_path)
            info = _row_for_name(benchmarks_df, name)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ImportError(
                "Loading datasets requires huggingface_hub. "
                "Install with: pip install torch_measure[data]"
            ) from err

        def _fetch(filename: str, *, optional: bool = False) -> Path | None:
            try:
                return Path(hf_hub_download(
                    repo_id=MANIFEST_REPO,
                    filename=filename,
                    repo_type="dataset",
                    force_download=force_download,
                ))
            except Exception:
                if optional:
                    return None
                raise

        responses_path = _fetch(f"{name}.parquet")
        items_path = _fetch("items.parquet")
        subjects_path = _fetch("subjects.parquet")
        traces_path = _fetch(f"{name}_traces.parquet", optional=True)

        responses = pd.read_parquet(responses_path)
        items = pd.read_parquet(items_path)
        subjects = pd.read_parquet(subjects_path)
        traces = pd.read_parquet(traces_path) if traces_path is not None else None

        manifest_row = manifest_entry(name)
        info = manifest_row if manifest_row is not None else {}

    # Filter registries down to just this benchmark.
    if "benchmark_id" in items.columns:
        items_for_bench = items[items["benchmark_id"] == name].reset_index(drop=True)
    else:
        items_for_bench = items.reset_index(drop=True)

    present_subjects = set(responses["subject_id"].unique()) if "subject_id" in responses.columns else set()
    if present_subjects and "subject_id" in subjects.columns:
        subjects_for_bench = subjects[subjects["subject_id"].isin(present_subjects)].reset_index(drop=True)
    else:
        subjects_for_bench = subjects.reset_index(drop=True)

    return responses, items_for_bench, subjects_for_bench, traces, info


def _row_for_name(benchmarks_df, name: str) -> dict:
    """Pull the benchmarks.parquet row for ``name`` as a plain dict."""
    if "benchmark_id" not in benchmarks_df.columns:
        return {}
    row = benchmarks_df[benchmarks_df["benchmark_id"] == name]
    if row.empty:
        return {}
    record = row.iloc[0].to_dict()
    return record
