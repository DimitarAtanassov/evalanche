"""chrF2++ character n-gram F-score."""

from __future__ import annotations

from typing import Any

import sacrebleu

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric, reference_text


class ChrfMetric(ScalarMetric):
    name = "chrf_pp"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        score = sacrebleu.sentence_chrf(gen.output, [reference], word_order=2)
        return score.score / 100, {"signature": "chrF2++", "raw_score": score.score}
