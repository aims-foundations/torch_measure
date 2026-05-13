# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Hierarchical empirical-Bayes predictor for cold-start AI evaluation.

This module implements the model deployed in the Stanford CS321M Predictive AI
Evaluation Challenge (Codabench leaderboard NLL = -0.59). Unlike standard
``torch_measure.Predictor`` subclasses, this class operates on raw textual
records rather than integer ``(subject_idx, item_idx)`` tuples. The reason is
intrinsic to the cold-start setting: at competition test time, item IDs are
NEW (never seen during training), so integer-based lookup is impossible. The
model pools subject identities (which ARE known at test time, since the same
909 subjects are evaluated across both training and test) with benchmark and
condition priors learned from training responses.

The prediction stack is a 6-level fallback hierarchy:

1. (subject_name, benchmark_id, condition)              -- triple match
2. (subject_name, benchmark_id)                         -- Bayesian-shrunk pair
3. sigmoid(logit(subj_prior) + logit(bench_prior) - logit(global))  -- 1-PL IRT blend
4. benchmark_id                                         -- unknown subject
5. subject_name                                         -- unknown benchmark
6. global mean                                          -- both unknown

The IRT blend at level 3 is the classical 1-parameter Rasch formulation
``P = sigmoid(theta - delta)`` rewritten in logit space against the global
mean as a reference. Bayesian shrinkage at level 2 protects low-count cells
from collapsing to 0 or 1.

References
----------
- Rasch, G. (1960). Probabilistic models for some intelligence and
  attainment tests. Danish Institute for Educational Research.
- Platt, J. (1999). Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods. Adv. Large Margin Class.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Provider prefixes that subjects sometimes appear with in subject_content
# but not in display_name (e.g. "meta-llama/Llama-2-7b-chat" vs "Llama-2-7b-chat").
_DEFAULT_PROVIDER_PREFIXES: tuple[str, ...] = (
    "meta-llama/", "openai/", "mistralai/", "google/", "anthropic/",
    "microsoft/", "huggingface/", "tiiuae/", "deepmind/", "cohere/",
    "allenai/", "01-ai/", "qwen/", "baichuan-inc/",
)

# Default probability clipping. Avoids inf in logit() and keeps NLL bounded.
_CLIP_LO: float = 0.05
_CLIP_HI: float = 0.95

# Default Platt-shift cap. Stops a single benchmark's K=5 labeled examples
# from producing extreme shifts. See ``calibrate`` for details.
_DEFAULT_PLATT_SHIFT_CAP: float = 1.5


def _logit(p: float) -> float:
    """Numerically safe logit. Inputs are clipped away from 0 and 1."""
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid that avoids overflow for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clip(p: float, lo: float = _CLIP_LO, hi: float = _CLIP_HI) -> float:
    return max(lo, min(hi, float(p)))


def parse_subject_name(subject_content: str) -> str:
    """Extract the model name from a subject_content block.

    The competition runtime renders subject metadata as::

        Name: <display_name>
        Organization: <provider>
        Parameters: <params>
        ...

    This helper returns the first line after the ``Name:`` label, stripped.
    Falls back to the first non-empty line if no ``Name:`` is present.
    """
    if not subject_content:
        return ""
    m = re.match(r"\s*Name:\s*(.+)", subject_content)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return subject_content.split("\n", 1)[0].strip()


def resolve_subject_name(
    raw_name: str,
    subj_lookup: dict[str, float],
    sb_lookup: dict[str, float],
    sbc_lookup: dict[str, float],
    aliases: dict[str, str],
    name_lc: dict[str, str],
    provider_prefixes: Iterable[str] = _DEFAULT_PROVIDER_PREFIXES,
) -> str:
    """Resolve a raw display name to a canonical key found in the lookup tables.

    The competition's ``subject_content`` field sometimes contains a
    provider-prefixed name (``meta-llama/Llama-2-7b-chat``) while the
    training data stores the bare display name (``Llama-2-7b-chat``). This
    helper tries each candidate (the raw name and its prefix-stripped
    variant) through three resolution paths in order: direct lookup,
    alias map, case-insensitive map.

    Parameters
    ----------
    raw_name : str
        Whatever ``parse_subject_name`` produced for the input record.
    subj_lookup, sb_lookup, sbc_lookup : dict
        Lookup tables. Used here only to test ``in`` membership.
    aliases : dict[str, str]
        Map from prefix-stripped name to canonical display name.
    name_lc : dict[str, str]
        Map from ``display_name.lower()`` to display name.
    provider_prefixes : Iterable[str]
        Prefixes to strip from the raw name.

    Returns
    -------
    str
        The canonical name to use as a lookup key, or ``raw_name`` if no
        match was found at any level.
    """
    candidates = [raw_name]
    raw_lower = raw_name.lower()
    for prefix in provider_prefixes:
        if raw_lower.startswith(prefix):
            candidates.append(raw_name[len(prefix):].strip())
            break

    for cand in candidates:
        if cand in subj_lookup or cand in sb_lookup or cand in sbc_lookup:
            return cand
        if cand in aliases:
            return aliases[cand]
        lc = cand.lower()
        if lc in name_lc:
            return name_lc[lc]
    return raw_name


class ColdStartLookupPredictor:
    """Hierarchical empirical-Bayes predictor for cold-start AI evaluation.

    This is the model deployed in the CS321M Predictive AI Evaluation
    Challenge. It predicts ``P(subject answers item correctly)`` for cells
    where the item is unseen but the subject identity is known.

    Parameters
    ----------
    sbc : dict[str, float]
        Map ``"{subject_name}||{benchmark_id}||{condition}" -> P``, the
        most specific level (level 1). Built from cells with n >= 3
        observations in training.
    sb : dict[str, float]
        Map ``"{subject_name}||{benchmark_id}" -> P``, Bayesian-shrunk
        toward the IRT blend with pseudo-count alpha (level 2).
    subj : dict[str, float]
        Map ``subject_name -> P``, subject ability prior (level 5).
        Restricted to subjects with n >= 10 training observations.
    bench : dict[str, float]
        Map ``benchmark_id -> P``, benchmark difficulty prior (level 4).
    global_mean : float
        Overall mean correctness in training (level 6).
    name_aliases : dict[str, str]
        Map from prefix-stripped name to canonical display name.
    name_lc : dict[str, str]
        Map from ``display_name.lower()`` to canonical display name.
    platt_shift_cap : float
        Maximum absolute logit shift applied during ``calibrate``.
        Stops K=5 labeled examples from producing extreme shifts.
    output_clip : tuple[float, float]
        Output probability is clipped to this range.

    Examples
    --------
    >>> # Build from in-memory lookup tables
    >>> predictor = ColdStartLookupPredictor(
    ...     sbc={"gpt-4||mmlupro||cot": 0.85},
    ...     sb={"gpt-4||mmlupro": 0.80},
    ...     subj={"gpt-4": 0.75},
    ...     bench={"mmlupro": 0.50},
    ...     global_mean=0.65,
    ...     name_aliases={},
    ...     name_lc={"gpt-4": "gpt-4"},
    ... )
    >>> p = predictor.predict({
    ...     "benchmark": "mmlupro",
    ...     "condition": "cot",
    ...     "subject_content": "Name: gpt-4",
    ...     "item_content": "What is 2+2?",
    ... })
    >>> abs(p - 0.85) < 1e-6
    True
    """

    def __init__(
        self,
        sbc: dict[str, float],
        sb: dict[str, float],
        subj: dict[str, float],
        bench: dict[str, float],
        global_mean: float,
        name_aliases: dict[str, str] | None = None,
        name_lc: dict[str, str] | None = None,
        platt_shift_cap: float = _DEFAULT_PLATT_SHIFT_CAP,
        output_clip: tuple[float, float] = (0.02, 0.98),
    ) -> None:
        self.sbc = dict(sbc)
        self.sb = dict(sb)
        self.subj = dict(subj)
        self.bench = dict(bench)
        self.global_mean = float(global_mean)
        self.name_aliases = dict(name_aliases or {})
        self.name_lc = dict(name_lc or {})
        self.platt_shift_cap = float(platt_shift_cap)
        self.output_clip = (float(output_clip[0]), float(output_clip[1]))

        # Pre-compute case-insensitive views for fast lookup.
        self._sbc_ci = {k.lower(): v for k, v in self.sbc.items()}
        self._sb_ci = {k.lower(): v for k, v in self.sb.items()}
        self._subj_ci = {k.lower(): v for k, v in self.subj.items()}
        self._bench_ci = {k.lower(): v for k, v in self.bench.items()}

        # Per-benchmark Platt calibration state. Populated by ``calibrate``.
        # Each entry is ``(slope, intercept)``; slope is fixed at 1.0 by design.
        self._platt: dict[str, tuple[float, float]] = {}
        self._platt_fit_key: int = -1

    # ---- Persistence -----------------------------------------------------

    @classmethod
    def from_lookup_json(cls, path: str | Path, **overrides: Any) -> "ColdStartLookupPredictor":
        """Load from the ``lookup.json`` artifact produced by the build script.

        Parameters
        ----------
        path : str | Path
            Path to a JSON file matching the schema written by the
            ``m3_build_colab.py`` script in the competition repo. Required
            keys: ``sbc``, ``sb``, ``subj``, ``bench``, ``global``.
        **overrides
            Keyword arguments passed through to ``__init__`` to override
            defaults (e.g. ``platt_shift_cap``).
        """
        data = json.loads(Path(path).read_text())
        return cls(
            sbc=data["sbc"],
            sb=data["sb"],
            subj=data["subj"],
            bench=data["bench"],
            global_mean=data["global"],
            name_aliases=data.get("name_aliases", {}),
            name_lc=data.get("name_lc", {}),
            **overrides,
        )

    def to_lookup_json(self, path: str | Path) -> None:
        """Write lookup tables to a JSON file matching the build-script schema."""
        payload = {
            "sbc": self.sbc,
            "sb": self.sb,
            "subj": self.subj,
            "bench": self.bench,
            "global": self.global_mean,
            "name_aliases": self.name_aliases,
            "name_lc": self.name_lc,
        }
        Path(path).write_text(json.dumps(payload))

    # ---- Core lookup -----------------------------------------------------

    def _lookup_p(self, subj_name: str, benchmark: str, condition: str) -> float:
        """Walk the 6-level fallback hierarchy and return the first match.

        Levels (each tried case-insensitively as a fallback within the level):

        1. ``sbc[subject||benchmark||condition]``
        2. ``sb[subject||benchmark]``
        3. IRT blend: ``sigmoid(logit(subj) + logit(bench) - logit(global))``
        4. ``bench[benchmark]``
        5. ``subj[subject]``
        6. ``global_mean``
        """
        condition = condition or "none"

        # Level 1: full triple (subject, benchmark, condition).
        key3 = f"{subj_name}||{benchmark}||{condition}"
        if key3 in self.sbc:
            return self.sbc[key3]
        if key3.lower() in self._sbc_ci:
            return self._sbc_ci[key3.lower()]
        key3_none = f"{subj_name}||{benchmark}||none"
        if key3_none in self.sbc:
            return self.sbc[key3_none]

        # Level 2: pair (subject, benchmark).
        key2 = f"{subj_name}||{benchmark}"
        if key2 in self.sb:
            return self.sb[key2]
        if key2.lower() in self._sb_ci:
            return self._sb_ci[key2.lower()]

        # Level 3: IRT blend if both single priors are known.
        subj_p = self.subj.get(subj_name) or self._subj_ci.get(subj_name.lower())
        bench_p = self.bench.get(benchmark) or self._bench_ci.get(benchmark.lower())
        if subj_p is not None and bench_p is not None:
            return _clip(_sigmoid(_logit(subj_p) + _logit(bench_p) - _logit(self.global_mean)))

        # Level 4: benchmark only.
        if bench_p is not None:
            return bench_p

        # Level 5: subject only.
        if subj_p is not None:
            return subj_p

        # Level 6: global mean.
        return self.global_mean

    # ---- Public prediction API -------------------------------------------

    def predict(
        self,
        record: dict[str, Any],
        labeled: list[dict[str, Any]] | None = None,
    ) -> float:
        """Predict ``P(correct)`` for a single competition-format record.

        Parameters
        ----------
        record : dict
            Must contain ``subject_content`` (string) and ``benchmark``
            (string). May contain ``condition`` (string) and
            ``item_content`` (string, unused by this predictor but
            accepted for interface compatibility).
        labeled : list[dict] | None
            Optional list of labeled examples in the same format with an
            additional ``label`` field. If provided, calibration is
            (re)fit per-benchmark and applied to the prediction.

        Returns
        -------
        float
            Predicted probability of correctness, clipped to ``output_clip``.
        """
        subject_content = record.get("subject_content", "") or ""
        benchmark = (record.get("benchmark", "") or "").strip()
        condition = (record.get("condition", "") or "none").strip() or "none"

        raw_name = parse_subject_name(subject_content)
        subj_name = resolve_subject_name(
            raw_name, self.subj, self.sb, self.sbc,
            self.name_aliases, self.name_lc,
        )
        p = self._lookup_p(subj_name, benchmark, condition)

        if labeled:
            self.calibrate(labeled)
            if benchmark in self._platt:
                slope, intercept = self._platt[benchmark]
                p = _sigmoid(slope * _logit(p) + intercept)

        lo, hi = self.output_clip
        return max(lo, min(hi, float(p)))

    def predict_batch(
        self,
        records: list[dict[str, Any]],
        labeled: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """Vectorised wrapper around ``predict``. Calibration is fit once."""
        if labeled:
            self.calibrate(labeled)
        return [self.predict(r) for r in records]

    # ---- Adaptive calibration --------------------------------------------

    def calibrate(self, labeled: list[dict[str, Any]]) -> None:
        """Fit intercept-only Platt scaling per benchmark from labeled records.

        With only K=5 labeled examples per benchmark, fitting both slope and
        intercept produces a near-flat calibrator (slope ~ 0.1) that destroys
        subject ordering. We fix slope = 1 and fit only the intercept shift,
        capped at +/- ``platt_shift_cap`` logit units. The shift is the
        difference between the empirical label-mean's logit and the mean
        predicted logit on the labeled examples.

        Benchmarks with fewer than 2 labeled examples are not calibrated.
        There is intentionally NO global fallback calibrator: applying one
        benchmark's labeled distribution to another benchmark introduces
        more noise than signal at this K.

        This method is idempotent if called with the same labeled list
        (cached by length).
        """
        new_key = len(labeled)
        if new_key == self._platt_fit_key:
            return
        self._platt_fit_key = new_key
        self._platt.clear()

        by_bench: dict[str, list[tuple[float, float]]] = {}
        for ex in labeled:
            if "label" not in ex:
                continue
            bench = (ex.get("benchmark") or "").strip()
            cond = (ex.get("condition") or "none").strip() or "none"
            subject_content = ex.get("subject_content", "") or ""
            raw_name = parse_subject_name(subject_content)
            subj_name = resolve_subject_name(
                raw_name, self.subj, self.sb, self.sbc,
                self.name_aliases, self.name_lc,
            )
            p = self._lookup_p(subj_name, bench, cond)
            by_bench.setdefault(bench, []).append((_logit(p), float(ex["label"])))

        for bench, pairs in by_bench.items():
            if len(pairs) < 2:
                continue
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            mean_y = sum(ys) / len(ys)
            if mean_y <= 0 or mean_y >= 1:
                continue  # No information for a shift.
            mean_x = sum(xs) / len(xs)
            target_logit = math.log(mean_y / (1.0 - mean_y))
            shift = target_logit - mean_x
            shift = max(-self.platt_shift_cap, min(self.platt_shift_cap, shift))
            self._platt[bench] = (1.0, shift)
