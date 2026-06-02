# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Experimental / negative-result models.

This namespace collects models that are documented for completeness and
reproducibility but are **not** part of the supported public API of
``torch_measure``. They typically carry self-disclosed caveats (negative
results, brittle dependencies, or narrowly-scoped fork-only motivations)
that disqualify them from the curated ``torch_measure.models`` surface.

Members
-------
- :class:`LLMJudgeIRT` -- 1-PL IRT with LLM-augmented item difficulty.
  Documents a convergent negative result across three iterations on the
  Stanford CS321M Predictive AI Evaluation Challenge data; the simpler
  :class:`torch_measure.models.ColdStartLookupPredictor` is the recommended
  deployment target.
- :func:`build_difficulty_prompt` -- prompt builder used by the judge
  function in :class:`LLMJudgeIRT`.

See ``docs/source/api/experimental.rst`` for the rendered API page and
``docs/SPLIT_MERGE_TARGETS.md`` for the upstream-eligible vs fork-only
file split.
"""

from torch_measure.experimental.llm_judge_irt import (
    LLMJudgeIRT,
    build_difficulty_prompt,
)

__all__ = ["LLMJudgeIRT", "build_difficulty_prompt"]
