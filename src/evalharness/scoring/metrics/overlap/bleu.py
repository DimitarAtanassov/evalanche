"""SacreBLEU: sentence score per case, corpus score on aggregate."""

from __future__ import annotations

from typing import Any

import sacrebleu
from sacrebleu.metrics.bleu import BLEU

from evalharness.domain import (
    OVERALL_SLICE,
    AggregateValue,
    Case,
    Generation,
    ScoreValue,
    ScoringContext,
)
from evalharness.scoring.base import ScalarMetric, reference_text


class BleuMetric(ScalarMetric):
    name = "sacrebleu"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        score = sacrebleu.sentence_bleu(gen.output, [reference])
        return score.score / 100, {
            "sentence_score": score.score,
            "hypothesis": gen.output,
            "reference": reference,
        }

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [
            value for value in values if value.value is not None and "hypothesis" in value.detail
        ]
        metric = BLEU()
        score = (
            metric.corpus_score(
                [str(value.detail["hypothesis"]) for value in valid],
                [[str(value.detail["reference"]) for value in valid]],
            )
            if valid
            else None
        )
        return AggregateValue(
            self.name,
            self.version,
            OVERALL_SLICE,
            len(valid),
            (score.score / 100) if score else 0.0,
            None,
            None,
            None,
            f"corpus BLEU; {metric.get_signature()}" if score else "corpus BLEU",
        )
