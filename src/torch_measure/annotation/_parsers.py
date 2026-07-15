"""Pure parsing functions, verbatim from adgomant/delean-batch-manager/parse.py."""
from __future__ import annotations

import math
import re


def extract_demand_level(response: str) -> float:
    """Verbatim from extract_demand_level_from_response() in the paper repo.

    Splits on blank lines, takes the last paragraph as the conclusion,
    extracts the last integer, validates 0-5, rejects leading section numbers.
    Returns math.nan on any failure.
    """
    segments = response.split("\n\n")
    conclusion = segments[-1]

    digits = re.findall(r"\d+", conclusion)
    if not digits:
        return math.nan

    score = int(digits[-1])
    if not 0 <= score <= 5:
        return math.nan

    # Reject if the only integer found is a leading section-header number
    # (e.g. "4. Conclusion:" where 4 is not the actual score).
    if len(digits) == 1 and re.search(rf"^{score}\.", conclusion, re.MULTILINE):
        return math.nan

    return float(score)


def extract_ug_score(response: str) -> tuple[str, float]:
    """Parse a UG classification response into (raw_output, ug_score).

    Model is instructed to output a single line: an integer N or the word "open".
    Formula: ug_score = (1 - 1/N) * 100  for MCQ, or 100.0 for open-ended.
    Returns math.nan as ug_score on any parse failure.
    """
    raw = response.strip().split("\n")[0].strip()

    if raw.lower() == "open":
        return raw, 100.0

    try:
        n = int(raw)
    except ValueError:
        return raw, math.nan

    if n < 1:
        return raw, math.nan

    return raw, round((1.0 - 1.0 / n) * 100.0, 6)
