"""Deterministic mock provider for CI and offline PoC."""

from __future__ import annotations

import re
from typing import Any

from evalharness.core.enums import ErrorClass, FinishReason
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)

_PLUS_ONE = re.compile(r"What is (\d+) plus one\?", re.IGNORECASE)

MOCK_DIGEST = "mock-digest-poc-0001"
MOCK_ADAPTER_VERSION = "mock-v1"


class MockProvider:
    """Offline provider that answers the synthetic QA fixture deterministically."""

    name = "mock"

    def __init__(self, base_url: str | None = None, **_: Any) -> None:
        # base_url accepted for CLI/registry symmetry with OllamaProvider
        self.base_url = base_url

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version=MOCK_DIGEST,
            quantization=None,
            params_b=None,
            context_window=4096,
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
        prompt = req.messages[-1].content if req.messages else ""
        text = self._answer(prompt)
        return GenerationResponse(
            text=text,
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            prompt_tokens=12,
            completion_tokens=max(1, len(text.split())),
            logprobs=None,
            ttft_ms=1.0,
            total_ms=5.0,
            raw={
                "provider": self.name,
                "model": model,
                "adapter": MOCK_ADAPTER_VERSION,
                "prompt": prompt,
                "deterministic": True,
            },
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.NON_RETRYABLE_REQUEST

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _answer(prompt: str) -> str:
        match = _PLUS_ONE.search(prompt)
        if match:
            return str(int(match.group(1)) + 1)
        return "unknown"
