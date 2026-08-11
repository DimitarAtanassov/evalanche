"""JSON parseability, optionally checked against the case's declared schema."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import ValidationError, validate

from evalharness.domain import Case, Generation, ScoringContext, TaskType
from evalharness.scoring.base import ScalarMetric


class JsonValidityMetric(ScalarMetric):
    name = "json_validity"
    task_types = frozenset({TaskType.EXTRACTION, TaskType.GENERATION, TaskType.TOOL_USE})
    requires = frozenset()
    config = {"threshold": 1.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        try:
            parsed = json.loads(gen.output or "")
            schema = case.inputs.get("json_schema")
            if schema:
                validate(parsed, schema)
            return 1.0, {"parsed": parsed, "schema_valid": True}
        except (json.JSONDecodeError, ValidationError) as exc:
            return 0.0, {"error": str(exc)}
