"""Protocol definitions for providers and metrics."""

from __future__ import annotations

from typing import Protocol

from evalharness.core.enums import ErrorClass, Requirement, TaskType
from evalharness.core.models import (
    AggregateValue,
    Capabilities,
    Case,
    Generation,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
    ScoreValue,
    ScoringContext,
)


class Provider(Protocol):
    name: str

    async def resolve_version(self, model: str) -> ModelVersion: ...

    def capabilities(self, model: str) -> Capabilities: ...

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse: ...

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...

    def classify_error(self, exc: Exception) -> ErrorClass: ...

    async def aclose(self) -> None: ...


class Metric(Protocol):
    name: str
    version: str
    task_types: frozenset[TaskType]
    requires: frozenset[Requirement]

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]: ...

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue: ...
