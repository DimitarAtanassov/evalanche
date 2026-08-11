"""Base class and shared helpers for the built-in scalar metrics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from evalharness.domain import (
    OVERALL_SLICE,
    AggregateValue,
    Case,
    Generation,
    Requirement,
    ScoreValue,
    ScoringContext,
    TaskType,
)
from evalharness.hashing import sha256_canonical
from evalharness.statistics import wilson_interval

ALL_TEXT_TASKS = frozenset(
    {TaskType.GENERATION, TaskType.QA_SHORT, TaskType.SUMMARIZATION, TaskType.RAG}
)


class ScalarMetric:
    """A metric whose per-case opinion is one optional float plus a detail payload.

    Subclasses implement :meth:`value`; ``score`` and ``aggregate`` supply the
    threshold-based pass decision and the mean-with-Wilson-interval rollup that most
    built-ins share. Override ``aggregate`` when the rollup is not a mean of the
    per-case values (corpus BLEU, classification accuracy).
    """

    name = "scalar"
    version = "1.0.0"
    task_types = ALL_TEXT_TASKS
    requires = frozenset({Requirement.REFERENCE})
    config: dict[str, Any] = {}

    @property
    def config_id(self) -> str:
        return sha256_canonical({"metric": self.name, "version": self.version, **self.config})

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        raise NotImplementedError

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]:
        value, detail = self.value(gen, case, ctx)
        return [
            ScoreValue(
                metric_name=self.name,
                metric_version=self.version,
                metric_config_sha256=self.config_id,
                value=value,
                passed=None if value is None else value >= float(self.config.get("threshold", 0.5)),
                detail=detail,
            )
        ]

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [float(value.value) for value in values if value.value is not None]
        mean = float(np.mean(valid)) if valid else 0.0
        low, high = (
            wilson_interval(
                sum(value >= float(self.config.get("threshold", 0.5)) for value in valid),
                len(valid),
            )
            if valid
            else (0.0, 0.0)
        )
        return AggregateValue(
            self.name,
            self.version,
            OVERALL_SLICE,
            len(valid),
            mean,
            low,
            high,
            float(np.std(valid)) if valid else None,
            "mean+wilson",
        )


def reference_text(case: Case) -> str | None:
    """The single reference a scalar metric compares against, or ``None`` if the case has none."""
    return case.reference_answer or (case.references[0] if case.references else None)


def tokens(text: str) -> list[str]:
    """Case-folded word tokens; the shared tokenization for overlap-style metrics."""
    return re.findall(r"\w+", text.casefold())
