"""Mean average precision over the graded relevance judgements."""

from __future__ import annotations

from typing import Any

from evalharness.domain import Case, Generation, Requirement, ScoringContext, TaskType
from evalharness.scoring.base import ScalarMetric
from evalharness.scoring.metrics.retrieval.ranking import graded_relevance, parse_ranking


class RetrievalMapMetric(ScalarMetric):
    """Average precision: precision at each relevant hit, divided by the relevant total.

    Dividing by the relevant total rather than by the number of hits means an unretrieved
    relevant document costs the score, so truncating the ranking cannot inflate it.
    """

    name = "retrieval_map"
    task_types = frozenset({TaskType.RETRIEVAL, TaskType.RAG})
    requires = frozenset({Requirement.QRELS})
    config = {"threshold": 0.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if not case.qrels:
            return None, {"excluded": "zero_relevance"}
        relevant = graded_relevance(case.qrels)
        if not relevant:
            return None, {"excluded": "zero_relevance"}
        ranking = parse_ranking(gen.output)
        precisions = [
            sum(item in relevant for item in ranking[:rank]) / rank
            for rank, item in enumerate(ranking, start=1)
            if item in relevant
        ]
        return sum(precisions) / len(relevant), {
            "relevant": len(relevant),
            "retrieved_relevant": len(precisions),
        }
