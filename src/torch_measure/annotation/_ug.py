from __future__ import annotations

import hashlib
from typing import Optional

from ._cache import AnnotationCache, make_cache_key
from ._client import GeminiClient
from ._parsers import extract_ug_score
from ._prompts import get_ug_instruction
from ._rubrics import RubricsCatalog
from ._types import AnnotationJob, CacheEntry, UGAnnotation


class UGAnnotator:
    """Classifies benchmark items as MCQ or open-ended and computes the UG score.

    UG (Unguessability) is separate from the 18 demand rubrics:
      - MCQ with N choices → ug_score = (1 - 1/N) * 100
      - open-ended         → ug_score = 100.0
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

    def annotate(self, job: AnnotationJob) -> UGAnnotation:
        key = make_cache_key(
            content=job.content,
            acronym="UG",
            model_id=self._client.model,
            rubric_hash=self._rubrics.ug_hash,
        )

        if self._cache is not None:
            entry = self._cache.get(key)
            if entry is not None:
                return UGAnnotation(
                    item_id=job.item_id,
                    raw_output=entry.raw_output,
                    ug_score=entry.level,
                    finish_reason=entry.finish_reason,
                    model_response=entry.model_response,
                )

        prompt = get_ug_instruction(
            item_text=job.content,
            reference_answer=job.reference_answer,
            ug_rubric_content=self._rubrics.ug_content,
        )
        model_response, finish_reason = self._client.generate(prompt)
        raw_output, ug_score = extract_ug_score(model_response)

        annotation = UGAnnotation(
            item_id=job.item_id,
            raw_output=raw_output,
            ug_score=ug_score,
            finish_reason=finish_reason,
            model_response=model_response,
        )

        if self._cache is not None:
            content_hash = hashlib.sha256(job.content.encode()).hexdigest()[:16]
            self._cache.put(CacheEntry(
                key=key,
                item_id=job.item_id,
                demand="UG",
                level=ug_score,
                finish_reason=finish_reason,
                model_response=model_response,
                rubric_hash=self._rubrics.ug_hash,
                model_id=self._client.model,
                content_hash=content_hash,
                timestamp=AnnotationCache.now_iso(),
                raw_output=raw_output,
            ))

        return annotation
