"""Normalized Levenshtein similarity."""

from __future__ import annotations

from typing import Any

from rapidfuzz.distance import Levenshtein

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric, reference_text


class LevenshteinMetric(ScalarMetric):
    name = "normalized_levenshtein"

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.config = {"threshold": threshold}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        value = float(Levenshtein.normalized_similarity(gen.output, reference))
        return value, {"threshold": self.threshold}
