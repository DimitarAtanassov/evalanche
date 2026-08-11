"""Mean reciprocal rank of the first relevant document."""

from __future__ import annotations

from typing import Any

from evalharness.domain import Case, Generation, Requirement, ScoringContext, TaskType
from evalharness.scoring.base import ScalarMetric
from evalharness.scoring.metrics.retrieval.ranking import graded_relevance, parse_ranking


class RetrievalMrrMetric(ScalarMetric):
    """Reciprocal rank of the first relevant hit; 0.0 when the ranking misses entirely."""

    name = "retrieval_mrr"
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
        ranks = [index + 1 for index, doc in enumerate(ranking) if doc in relevant]
        first = min(ranks) if ranks else None
        return (1 / first if first else 0.0), {
            "first_relevant_rank": first,
            "relevant": len(relevant),
            "retrieved": len(ranking),
        }
