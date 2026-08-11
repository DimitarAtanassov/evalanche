"""Precision@k over the graded relevance judgements."""

from __future__ import annotations

from typing import Any

from evalharness.domain import Case, Generation, Requirement, ScoringContext, TaskType
from evalharness.scoring.base import ScalarMetric
from evalharness.scoring.metrics.retrieval.ranking import (
    DEFAULT_CUTOFFS,
    DEFAULT_PRIMARY_CUTOFF,
    graded_relevance,
    parse_ranking,
)


class RetrievalPrecisionAtKMetric(ScalarMetric):
    """Precision at the primary cutoff, with every configured cutoff in the detail.

    Precision divides by the cutoff, not by the number of documents returned, so a short
    ranking is penalised rather than flattered.
    """

    name = "retrieval_precision_at_k"
    task_types = frozenset({TaskType.RETRIEVAL, TaskType.RAG})
    requires = frozenset({Requirement.QRELS})
    config = {"threshold": 0.0, "k": DEFAULT_PRIMARY_CUTOFF, "cutoffs": DEFAULT_CUTOFFS}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if not case.qrels:
            return None, {"excluded": "zero_relevance"}
        relevant = graded_relevance(case.qrels)
        if not relevant:
            return None, {"excluded": "zero_relevance"}
        ranking = parse_ranking(gen.output)
        detail: dict[str, Any] = {
            f"precision@{cutoff}": sum(doc in relevant for doc in ranking[:cutoff]) / cutoff
            for cutoff in self.config["cutoffs"]
        }
        primary = int(self.config["k"])
        detail["k"] = primary
        return sum(doc in relevant for doc in ranking[:primary]) / primary, detail
