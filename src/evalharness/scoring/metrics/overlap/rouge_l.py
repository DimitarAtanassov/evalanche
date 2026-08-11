"""ROUGE-L, with a deterministic in-tree fallback when ``rouge_score`` is unavailable."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric, reference_text, tokens

try:
    from rouge_score import rouge_scorer as _rouge_scorer
except ImportError:  # pragma: no cover - optional path when rouge_score is unavailable
    _rouge_scorer = None


class RougeLMetric(ScalarMetric):
    name = "rouge_l"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        if _rouge_scorer is not None:
            scores = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"]).score(
                reference, gen.output
            )
            detail = {
                name: {
                    "precision": value.precision,
                    "recall": value.recall,
                    "fmeasure": value.fmeasure,
                }
                for name, value in scores.items()
            }
            return float(scores["rougeL"].fmeasure), detail
        # Deterministic fallback when rouge_score cannot be imported (e.g. NLTK
        # blocked on some Python 3.14 paths).
        predicted = tokens(gen.output)
        expected = tokens(reference)
        fallback_detail: dict[str, dict[str, float]] = {}
        for size, name in ((1, "rouge1"), (2, "rouge2")):
            predicted_ngrams = Counter(
                zip(*(predicted[index:] for index in range(size)), strict=False)
            )
            expected_ngrams = Counter(
                zip(*(expected[index:] for index in range(size)), strict=False)
            )
            overlap = sum((predicted_ngrams & expected_ngrams).values())
            fallback_detail[name] = _prf(
                overlap, sum(predicted_ngrams.values()), sum(expected_ngrams.values())
            )
        lcs = _lcs_length(predicted, expected)
        fallback_detail["rougeL"] = _prf(lcs, len(predicted), len(expected))
        fallback_detail["rougeLsum"] = fallback_detail["rougeL"]
        return fallback_detail["rougeL"]["fmeasure"], fallback_detail


def _prf(overlap: int, predicted: int, expected: int) -> dict[str, float]:
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / expected if expected else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1 if token == other else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]
