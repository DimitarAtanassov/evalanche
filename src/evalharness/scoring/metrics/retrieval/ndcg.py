"""NDCG@10 with exponential gain."""

from __future__ import annotations

import math
from typing import Any

from evalharness.domain import Case, Generation, Requirement, ScoringContext, TaskType
from evalharness.scoring.base import ScalarMetric
from evalharness.scoring.metrics.retrieval.ranking import (
    DEFAULT_CUTOFFS,
    graded_relevance,
    parse_ranking,
)


class RetrievalMetric(ScalarMetric):
    """NDCG@10 as the primary value.

    The per-cutoff precision/recall/hit, MRR, and MAP keys in ``detail`` are retained for
    one release now that ``retrieval_precision_at_k``, ``retrieval_mrr``, and
    ``retrieval_map`` publish them as first-class metrics. Read the siblings, not the
    detail, for anything you plan to gate on.
    """

    name = "retrieval_ndcg_10"
    task_types = frozenset({TaskType.RETRIEVAL, TaskType.RAG})
    requires = frozenset({Requirement.QRELS})
    config = {"threshold": 0.0, "cutoffs": DEFAULT_CUTOFFS}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if not case.qrels:
            return None, {"excluded": "zero_relevance"}
        ranking = parse_ranking(gen.output)
        relevant = graded_relevance(case.qrels)
        detail: dict[str, Any] = {}
        for cutoff in self.config["cutoffs"]:
            selected = ranking[:cutoff]
            hits = sum(doc in relevant for doc in selected)
            detail[f"precision@{cutoff}"] = hits / cutoff
            detail[f"recall@{cutoff}"] = hits / len(relevant)
            detail[f"hit@{cutoff}"] = float(hits > 0)
        ranks = [index + 1 for index, doc in enumerate(ranking) if doc in relevant]
        detail["mrr"] = 1 / min(ranks) if ranks else 0.0
        precisions = [
            sum(item in relevant for item in ranking[:rank]) / rank
            for rank, item in enumerate(ranking, start=1)
            if item in relevant
        ]
        detail["map"] = sum(precisions) / len(relevant)
        dcg = sum(
            (2 ** relevant.get(doc, 0) - 1) / math.log2(index + 2)
            for index, doc in enumerate(ranking[:10])
        )
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        ndcg = dcg / idcg if idcg else 0.0
        detail["recall_ceiling"] = min(1.0, len(ranking) / len(relevant))
        return ndcg, detail
