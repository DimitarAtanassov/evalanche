"""SQuAD-style token-overlap F1."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric, reference_text, tokens


class SquadMetric(ScalarMetric):
    name = "squad_f1"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        predicted, expected = tokens(gen.output), tokens(reference)
        common = Counter(predicted) & Counter(expected)
        overlap = sum(common.values())
        precision = overlap / len(predicted) if predicted else float(not expected)
        recall = overlap / len(expected) if expected else float(not predicted)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, {"precision": precision, "recall": recall, "f1": f1}
