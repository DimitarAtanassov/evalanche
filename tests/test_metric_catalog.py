from __future__ import annotations

import math
import warnings

from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome, TaskType
from evalharness.domain.generation import Generation
from evalharness.domain.scoring import ScoringContext
from evalharness.scoring.metrics.classification.labels import ClassificationMetric
from evalharness.scoring.metrics.lexical.squad_f1 import SquadMetric
from evalharness.scoring.metrics.retrieval.average_precision import RetrievalMapMetric
from evalharness.scoring.metrics.retrieval.mrr import RetrievalMrrMetric
from evalharness.scoring.metrics.retrieval.ndcg import RetrievalMetric
from evalharness.scoring.metrics.retrieval.precision_at_k import RetrievalPrecisionAtKMetric
from evalharness.scoring.metrics.structured.json_field_f1 import JsonFieldF1Metric


def generation(output: str) -> Generation:
    return Generation(
        None,
        "run",
        "case",
        0,
        output,
        [],
        None,
        FailureOutcome.PASSED,
        None,
        None,
        0.0,
        None,
        None,
        None,
        1,
        [],
        False,
        None,
        None,
    )


def test_squad_metric_token_overlap() -> None:
    case = Case("case", TaskType.QA_SHORT, {}, reference_answer="the blue whale")
    score = SquadMetric().score(generation("blue whale"), case, ScoringContext("n"))[0]
    assert math.isclose(score.value or 0, 0.8)


def test_json_flattened_field_f1() -> None:
    case = Case("case", TaskType.EXTRACTION, {}, expected_json={"a": 1, "nested": {"b": 2}})
    score = JsonFieldF1Metric().score(
        generation('{"a":1,"nested":{"b":3}}'), case, ScoringContext("n")
    )[0]
    assert score.value == 0.5


def test_retrieval_ndcg_and_short_list_semantics() -> None:
    case = Case("case", TaskType.RETRIEVAL, {}, qrels={"d1": 3, "d2": 1})
    score = RetrievalMetric().score(generation('["d1","d2"]'), case, ScoringContext("n"))[0]
    assert score.value == 1.0
    assert score.detail["precision@5"] == 0.4
    assert score.detail["recall@5"] == 1.0


def test_retrieval_siblings_reproduce_the_ndcg_detail_values() -> None:
    """The orthogonal metrics must agree with the detail keys they are replacing."""
    case = Case("case", TaskType.RETRIEVAL, {}, qrels={"d1": 3, "d2": 1, "d3": 2})
    output = generation('["x","d1","d2"]')
    context = ScoringContext("n")
    detail = RetrievalMetric().score(output, case, context)[0].detail

    precision = RetrievalPrecisionAtKMetric().score(output, case, context)[0]
    mrr = RetrievalMrrMetric().score(output, case, context)[0]
    average_precision = RetrievalMapMetric().score(output, case, context)[0]

    assert precision.detail["precision@5"] == detail["precision@5"]
    assert precision.value == detail["precision@10"]
    assert mrr.value == detail["mrr"]
    assert average_precision.value == detail["map"]


def test_retrieval_siblings_score_a_partial_ranking() -> None:
    case = Case("case", TaskType.RETRIEVAL, {}, qrels={"d1": 1, "d2": 1})
    output = generation('["x","d1"]')
    context = ScoringContext("n")

    assert RetrievalPrecisionAtKMetric().score(output, case, context)[0].value == 0.1
    assert RetrievalMrrMetric().score(output, case, context)[0].value == 0.5
    assert RetrievalMapMetric().score(output, case, context)[0].value == 0.25


def test_retrieval_siblings_exclude_a_case_with_no_positive_judgement() -> None:
    case = Case("case", TaskType.RETRIEVAL, {}, qrels={"d1": 0})
    output = generation('["d1"]')
    context = ScoringContext("n")

    for metric in (RetrievalPrecisionAtKMetric(), RetrievalMrrMetric(), RetrievalMapMetric()):
        score = metric.score(output, case, context)[0]

        assert score.value is None
        assert score.detail == {"excluded": "zero_relevance"}


def test_classification_aggregate_suppresses_expected_unseen_prediction_warning() -> None:
    metric = ClassificationMetric()
    context = ScoringContext("n")
    scores = [
        metric.score(
            generation(output),
            Case(f"case-{index}", TaskType.CLASSIFICATION, {}, expected_label="A"),
            context,
        )[0]
        for index, output in enumerate(("A", "B"))
    ]

    with warnings.catch_warnings(record=True) as recorded:
        aggregate = metric.aggregate(scores)

    assert recorded == []
    assert aggregate.value == 0.5
