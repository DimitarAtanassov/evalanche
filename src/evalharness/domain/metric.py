"""Metric protocol."""

from __future__ import annotations

from typing import Protocol

from evalharness.domain.dataset import Case
from evalharness.domain.enums import Requirement, TaskType
from evalharness.domain.generation import Generation
from evalharness.domain.scoring import AggregateValue, ScoreValue, ScoringContext


class Metric(Protocol):
    name: str
    version: str
    task_types: frozenset[TaskType]
    requires: frozenset[Requirement]

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]: ...

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue: ...
