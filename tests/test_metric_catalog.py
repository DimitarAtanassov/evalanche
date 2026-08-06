from __future__ import annotations

import math
import warnings

from evalharness.core.enums import FailureOutcome, TaskType
from evalharness.core.models import Case, Generation, ScoringContext
from evalharness.scoring.catalog import (
    ClassificationMetric,
    JsonFieldF1Metric,
    RetrievalMetric,
    SquadMetric,
)


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
