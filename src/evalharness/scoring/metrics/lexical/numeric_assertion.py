"""Numeric equality within tolerance between prediction and reference."""

from __future__ import annotations

import math
import re
from typing import Any

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric, reference_text

_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


class NumericAssertionMetric(ScalarMetric):
    name = "numeric_assertion"
    config = {"threshold": 1.0, "abs_tol": 1e-6, "rel_tol": 1e-6}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        predicted_numbers = [float(value) for value in _NUMBER.findall(gen.output)]
        expected_numbers = [float(value) for value in _NUMBER.findall(reference)]
        passed = len(predicted_numbers) == len(expected_numbers) and all(
            math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
            for left, right in zip(predicted_numbers, expected_numbers, strict=True)
        )
        return float(passed), {"prediction": predicted_numbers, "reference": expected_numbers}
