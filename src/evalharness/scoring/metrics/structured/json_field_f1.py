"""Field-level F1 between the predicted JSON and the case's expected JSON."""

from __future__ import annotations

import json
from typing import Any

from evalharness.domain import Case, Generation, ScoringContext, TaskType
from evalharness.scoring.base import ScalarMetric


class JsonFieldF1Metric(ScalarMetric):
    name = "json_field_f1"
    task_types = frozenset({TaskType.EXTRACTION, TaskType.GENERATION})
    requires = frozenset()

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if case.expected_json is None:
            return None, {"reason": "missing_expected_json"}
        try:
            predicted = _flatten(json.loads(gen.output or ""))
        except json.JSONDecodeError:
            return 0.0, {"reason": "invalid_json"}
        expected = _flatten(case.expected_json)
        matches = sum(predicted.get(key) == value for key, value in expected.items())
        precision = matches / len(predicted) if predicted else 0.0
        recall = matches / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, {"precision": precision, "recall": recall}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Collapse nested objects and arrays to dotted/indexed leaf paths."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}[{index}]"))
        return result
    return {prefix: value}
