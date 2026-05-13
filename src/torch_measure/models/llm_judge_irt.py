# Copyright (c) 2026 AIMS Foundations. MIT License.

"""1-PL IRT with LLM-augmented item difficulty (experimental).

This module implements the M4.5 architecture explored in the Stanford
CS321M Predictive AI Evaluation Challenge. The model formulation is::

    P(correct | subject, item)  =  sigmoid( theta - delta )
        theta = logit( ColdStartLookupPredictor(subject, bench, cond) )
        delta = alpha * (logP_yes - logP_no | item, bench)  +  beta

where the ``(logP_yes - logP_no)`` term is produced by a frozen
instruction-tuned LLM judge given a per-item difficulty prompt.

Negative-result disclosure
--------------------------
Across three rigorously validated iterations on the competition data
(v1: 600 fit rows, v2: 600 fit rows with validated alpha, v3: 1500 fit
rows with validated alpha), the judge consistently extracted a real but
very small signal::

    Raw judge_logit -- label correlation  r = -0.257  (n=1500, stable from n=600)
    Validation-set NLL lift over M3 baseline  = +0.002
    Held-out smoke-set NLL lift over M3 baseline  = -0.006

The effect size is roughly five times smaller than the sample-to-sample
variance at deployment-relevant sample sizes (sigma_NLL ~ 0.020 on
200-row samples). The model is therefore documented here for
completeness and reproducibility but is NOT the recommended deployment
target -- the simpler :class:`ColdStartLookupPredictor` is preferred for
its robustness.

The pedagogical value of this module is the documentation of an honest
iteration log: hypothesis, three principled iterations, convergent
negative result. See ``tutorials/predictive_evaluation_challenge.ipynb``
for the full narrative.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable

from torch_measure.models.cold_start_lookup import (
    ColdStartLookupPredictor,
    _logit,
    _sigmoid,
    parse_subject_name,
    resolve_subject_name,
)


def build_difficulty_prompt(
    item_content: str,
    benchmark: str,
    max_item_chars: int = 1500,
) -> str:
    """Build the difficulty-rating prompt for the LLM judge.

    The prompt asks the judge a SINGLE binary question about intrinsic
    item difficulty, deliberately without mentioning any specific
    subject. The judge is rating the item, not predicting subject
    behaviour, which is a measurable competence at this scale.
    """
    item = (item_content or "")[:max_item_chars]
    return (
        f"Below is a question from the benchmark `{benchmark}`. "
        f"Without solving it, rate whether this question would be difficult "
        f"for a typical AI assistant.\n\n"
        f"Question:\n{item}\n\n"
        f"Is this question difficult? Answer with one word, yes or no.\n"
        f"Answer:"
    )


class LLMJudgeIRT:
    """1-PL IRT predictor with LLM-augmented item difficulty (experimental).

    Combines a :class:`ColdStartLookupPredictor` (for ``theta``) with a
    user-supplied judge function (for ``delta``) under the Rasch model.

    Parameters
    ----------
    lookup : ColdStartLookupPredictor
        Provides ``theta_subject`` via ``logit(lookup.predict(record))``.
    judge_fn : Callable[[str, str], float]
        Function mapping ``(item_content, benchmark)`` to a signed
        difficulty logit (``logP_yes - logP_no`` from the judge). May
        raise; raised exceptions are converted to a logit of 0.0 so
        that the prediction degrades gracefully toward ``theta``.
    alpha : float
        Slope on the judge logit. Should be fit by held-out
        log-likelihood maximization with L2 regularization; see
        :meth:`fit_alpha_beta`.
    beta : float
        Intercept on the judge logit.
    logit_cap : float
        Symmetric clip on ``theta - delta`` before sigmoid. Prevents
        runaway logits from short prompts producing extreme yes/no
        ratios.

    Notes
    -----
    See module docstring for the negative-result disclosure. Production
    code should prefer :class:`ColdStartLookupPredictor` directly.
    """

    def __init__(
        self,
        lookup: ColdStartLookupPredictor,
        judge_fn: Callable[[str, str], float],
        alpha: float,
        beta: float = 0.0,
        logit_cap: float = 3.0,
    ) -> None:
        self.lookup = lookup
        self.judge_fn = judge_fn
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.logit_cap = float(logit_cap)

    def predict(self, record: dict[str, Any]) -> float:
        """Predict ``P(correct)`` for a single competition-format record."""
        subject_content = record.get("subject_content", "") or ""
        benchmark = (record.get("benchmark", "") or "").strip()
        condition = (record.get("condition", "") or "none").strip() or "none"
        item_content = record.get("item_content", "") or ""

        raw_name = parse_subject_name(subject_content)
        subj_name = resolve_subject_name(
            raw_name,
            self.lookup.subj,
            self.lookup.sb,
            self.lookup.sbc,
            self.lookup.name_aliases,
            self.lookup.name_lc,
        )
        p_lookup = self.lookup._lookup_p(subj_name, benchmark, condition)
        theta = _logit(p_lookup)

        try:
            judge_logit = float(self.judge_fn(item_content, benchmark))
        except Exception:
            judge_logit = 0.0

        delta = self.alpha * judge_logit + self.beta
        z = max(-self.logit_cap, min(self.logit_cap, theta - delta))
        p = _sigmoid(z)
        lo, hi = self.lookup.output_clip
        return max(lo, min(hi, float(p)))

    # ---- Parameter fitting -------------------------------------------------

    @staticmethod
    def fit_alpha_beta(
        thetas: Sequence[float],
        judge_logits: Sequence[float],
        labels: Sequence[int],
        l2_penalty: float = 2.0,
        max_iter: int = 800,
    ) -> tuple[float, float, float]:
        """Fit ``(alpha, beta)`` by L2-regularised maximum likelihood.

        Optimization objective::

            argmin  NLL(theta, judge_logits, labels; alpha, beta)
                    +  l2_penalty * alpha^2

        L2 is applied only to ``alpha``. ``beta`` is the intercept and is
        not regularised. The regularisation strength should be tuned on a
        held-out validation set; see the v3 build script for an L2 sweep.

        Returns
        -------
        (alpha, beta, fit_nll) : tuple of floats
            Fitted parameters and the un-penalised mean log-likelihood
            achieved on the training set.

        Raises
        ------
        ImportError
            If scipy is not installed.
        """
        try:
            import numpy as np
            from scipy.optimize import minimize
        except ImportError as e:
            raise ImportError(
                "fit_alpha_beta requires scipy and numpy. "
                "Install with `pip install scipy numpy`."
            ) from e

        theta_arr = np.asarray(thetas, dtype=float)
        jl_arr = np.asarray(judge_logits, dtype=float)
        y_arr = np.asarray(labels, dtype=float)

        def neg_log_lik(params):
            alpha, beta = params
            delta = alpha * jl_arr + beta
            z = np.clip(theta_arr - delta, -3.0, 3.0)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            nll = -float(np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))
            return nll + l2_penalty * alpha * alpha

        res = minimize(
            neg_log_lik,
            x0=[0.0, 0.0],
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": max_iter},
        )
        alpha, beta = float(res.x[0]), float(res.x[1])

        # Compute the un-penalised fit NLL for reporting.
        delta = alpha * jl_arr + beta
        z = np.clip(theta_arr - delta, -3.0, 3.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        fit_nll = float(np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))
        return alpha, beta, fit_nll
