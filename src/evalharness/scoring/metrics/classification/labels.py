"""Label agreement, aggregated as accuracy with an imbalance-aware detail payload."""

from __future__ import annotations

import json
import warnings
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from evalharness.domain import (
    OVERALL_SLICE,
    AggregateValue,
    Case,
    Generation,
    ScoreValue,
    ScoringContext,
    TaskType,
)
from evalharness.scoring.base import ScalarMetric
from evalharness.statistics import wilson_interval


class ClassificationMetric(ScalarMetric):
    name = "classification"
    task_types = frozenset({TaskType.CLASSIFICATION})
    requires = frozenset()
    config = {"threshold": 1.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if gen.output is None or case.expected_label is None:
            return None, {"reason": "missing"}
        predicted = gen.output.strip()
        return float(predicted == case.expected_label), {
            "predicted": predicted,
            "expected": case.expected_label,
        }

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        details = [value.detail for value in values if value.value is not None]
        expected = [str(item["expected"]) for item in details]
        predicted = [str(item["predicted"]) for item in details]
        accuracy = float(accuracy_score(expected, predicted)) if expected else 0.0
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                expected, predicted, average="weighted", zero_division=0
            )
            if expected
            else (0.0, 0.0, 0.0, None)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="y_pred contains classes not in y_true",
                category=UserWarning,
            )
            balanced_accuracy = (
                float(balanced_accuracy_score(expected, predicted)) if expected else 0.0
            )
        detail = {
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": float(f1_score(expected, predicted, average="macro")) if expected else 0.0,
            "micro_f1": float(f1_score(expected, predicted, average="micro")) if expected else 0.0,
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
            "mcc": float(matthews_corrcoef(expected, predicted)) if expected else 0.0,
            "cohen_kappa": float(cohen_kappa_score(expected, predicted)) if expected else 0.0,
        }
        low, high = wilson_interval(
            sum(x == y for x, y in zip(expected, predicted, strict=True)), len(expected)
        )
        return AggregateValue(
            self.name,
            self.version,
            OVERALL_SLICE,
            len(expected),
            accuracy,
            low,
            high,
            None,
            json.dumps(detail, sort_keys=True),
        )
