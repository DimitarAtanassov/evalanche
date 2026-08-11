"""Required and forbidden term assertions."""

from __future__ import annotations

from typing import Any

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric


class AssertionMetric(ScalarMetric):
    name = "assertions"
    requires = frozenset()
    config = {"threshold": 1.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if gen.output is None:
            return 0.0, {"reason": "missing_output"}
        folded = gen.output.casefold()
        required = {term: term.casefold() in folded for term in case.must_contain}
        forbidden = {term: term.casefold() not in folded for term in case.must_not_contain}
        checks = [*required.values(), *forbidden.values()]
        return float(all(checks)), {"contains": required, "forbidden": forbidden}
