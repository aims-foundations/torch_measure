from __future__ import annotations

import hashlib
from typing import Optional

from ._cache import AnnotationCache, make_cache_key
from ._client import GeminiClient
from ._parsers import extract_demand_level
from ._prompts import get_full_instruction
from ._rubrics import RubricsCatalog
from ._types import (
    AnnotationJob,
    CacheEntry,
    DemandAnnotation,
    DemandVector,
    ItemAnnotation,
    Rubric,
)
from ._ug import UGAnnotator


class DemandAnnotator:
    """Runs the full 19-call ADeLe annotation pipeline for one benchmark item.

    One API call per demand rubric (18 sequential calls) plus one UG call.
    Results are cached to avoid redundant API calls across runs.

    ``client`` can be any object implementing
    ``generate(prompt: str) -> tuple[str, str]``. The bundled
    :class:`GeminiClient` is the default, but any LLM provider can be used
    by wrapping it in a class with that single method.
    """

    def __init__(
        self,
        client: GeminiClient,
        rubrics: RubricsCatalog,
        cache: Optional[AnnotationCache] = None,
    ) -> None:
        self._client = client
        self._rubrics = rubrics
        self._cache = cache
        self._ug = UGAnnotator(client, rubrics, cache)

    def annotate(self, job: AnnotationJob) -> ItemAnnotation:
        """Annotate one item across all 18 demand rubrics plus UG."""
        demands: dict[str, DemandAnnotation] = {}
        for rubric in self._rubrics.all_demand_rubrics():
            demands[rubric.acronym] = self._annotate_one(job, rubric)
        ug = self._ug.annotate(job)
        return ItemAnnotation(item_id=job.item_id, demands=demands, ug=ug)

    def annotate_dataset(self, jobs: list[AnnotationJob]) -> DemandVector:
        """Annotate all items and return a (n_items × 19) tensor.

        Row ordering in the returned ``DemandVector.tensor`` mirrors the order
        of ``jobs``. To pass the result to ``DemandAssessor.fit()``, supply
        ``jobs`` in the same order as ``data.to_fit_tensors()["item_ids"]``::

            item_ids = data.to_fit_tensors()["item_ids"]          # canonical order
            jobs = [AnnotationJob(iid, content[iid], ref[iid]) for iid in item_ids]
            dv   = annotator.annotate_dataset(jobs)
            model.fit(data, item_features=dv.tensor)
        """
        import torch

        item_ids: list[str] = []
        rows: list[list[float]] = []
        for job in jobs:
            item_ann = self.annotate(job)
            item_ids.append(job.item_id)
            rows.append(item_ann.to_feature_vector())

        tensor = torch.tensor(rows, dtype=torch.float32)
        return DemandVector(item_ids=item_ids, tensor=tensor)

    def _annotate_one(self, job: AnnotationJob, rubric: Rubric) -> DemandAnnotation:
        key = make_cache_key(
            content=job.content,
            acronym=rubric.acronym,
            model_id=self._client.model,
            rubric_hash=rubric.rubric_hash,
        )

        if self._cache is not None:
            entry = self._cache.get(key)
            if entry is not None:
                return DemandAnnotation(
                    item_id=job.item_id,
                    demand=rubric.acronym,
                    level=entry.level,
                    finish_reason=entry.finish_reason,
                    model_response=entry.model_response,
                )

        prompt = get_full_instruction(
            dimension=rubric.dimension_name,
            rubric_content=rubric.content,
            item_text=job.content,
        )
        model_response, finish_reason = self._client.generate(prompt)
        level = extract_demand_level(model_response)

        annotation = DemandAnnotation(
            item_id=job.item_id,
            demand=rubric.acronym,
            level=level,
            finish_reason=finish_reason,
            model_response=model_response,
        )

        if self._cache is not None:
            content_hash = hashlib.sha256(job.content.encode()).hexdigest()[:16]
            self._cache.put(CacheEntry(
                key=key,
                item_id=job.item_id,
                demand=rubric.acronym,
                level=level,
                finish_reason=finish_reason,
                model_response=model_response,
                rubric_hash=rubric.rubric_hash,
                model_id=self._client.model,
                content_hash=content_hash,
                timestamp=AnnotationCache.now_iso(),
            ))

        return annotation
