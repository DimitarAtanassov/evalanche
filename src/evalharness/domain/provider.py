"""Provider protocol."""

from __future__ import annotations

from typing import Protocol

from evalharness.domain.enums import ErrorClass
from evalharness.domain.generation import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)


class Provider(Protocol):
    name: str

    async def resolve_version(self, model: str) -> ModelVersion: ...

    def capabilities(self, model: str) -> Capabilities: ...

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse: ...

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...

    def classify_error(self, exc: Exception) -> ErrorClass: ...

    async def aclose(self) -> None: ...
