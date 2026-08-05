"""Test fixtures and helpers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from evalharness.config import get_settings
from evalharness.core.enums import ErrorClass, FinishReason
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.store.db import get_engine, init_db


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://evalharness:evalharness@localhost:5432/evalharness",
        ),
    )
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def db_ready() -> AsyncIterator[None]:
    try:
        await init_db()
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")
    yield
    engine = get_engine()
    await engine.dispose()
    get_settings.cache_clear()


class MockProvider:
    name = "mock"

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, GenerationRequest]] = []

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version="mock-digest-abc123",
            quantization="Q4",
            capabilities=self.capabilities(model),
        )

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_seed=True,
            supports_logprobs=False,
            supports_tools=False,
            supports_json_schema=False,
            supports_streaming=False,
            supports_system_role=True,
            max_context_tokens=4096,
        )

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse:
        self.calls.append((model, req))
        prompt = req.messages[-1].content
        text = self.responses.get(prompt, "unknown")
        return GenerationResponse(
            text=text,
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            prompt_tokens=10,
            completion_tokens=5,
            logprobs=None,
            ttft_ms=5.0,
            total_ms=20.0,
            raw={"mock": True, "prompt": prompt},
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.RETRYABLE_TRANSIENT

    async def aclose(self) -> None:
        return None
