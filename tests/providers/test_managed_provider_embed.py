"""ManagedProvider embed capacity and circuit-open classification (PR10 / D5)."""

from __future__ import annotations

import pytest

from evalharness.domain.enums import ErrorClass, FinishReason
from evalharness.domain.generation import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelVersion,
)
from evalharness.providers.call_policy import (
    ProviderCallError,
    ProviderCallPolicy,
    generate_with_policy,
)
from evalharness.providers.runtime import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ManagedProvider,
    estimate_embed_tokens,
)


class _RecordingProvider:
    """Minimal Provider double that records embed/generate calls and can fail on demand."""

    name = "recording"

    def __init__(self, *, fail_embed: bool = False) -> None:
        self.embed_calls: list[tuple[str, list[str]]] = []
        self.generate_calls: list[str] = []
        self.fail_embed = fail_embed

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version="v1",
            quantization=None,
            params_b=None,
            context_window=4096,
            capabilities=self.capabilities(model),
        )

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_seed=False,
            supports_logprobs=False,
            supports_tools=False,
            supports_json_schema=False,
            supports_streaming=False,
            supports_system_role=True,
            max_context_tokens=4096,
        )

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse:
        self.generate_calls.append(model)
        return GenerationResponse(
            text="ok",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            prompt_tokens=1,
            completion_tokens=1,
            logprobs=None,
            ttft_ms=1.0,
            total_ms=1.0,
            raw={},
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append((model, list(texts)))
        if self.fail_embed:
            raise RuntimeError("embed backend failed")
        return [[float(len(t))] for t in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.RETRYABLE_TRANSIENT


def _managed(
    provider: _RecordingProvider,
    *,
    breaker: CircuitBreaker | None = None,
    tpm: int = 10_000,
) -> ManagedProvider:
    return ManagedProvider(
        provider,
        rpm=1000,
        tpm=tpm,
        concurrency=4,
        breaker=breaker,
    )


def _generation_request() -> GenerationRequest:
    return GenerationRequest(
        messages=[Message(role="user", content="hi")],
        max_tokens=8,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=0,
        stop=[],
        response_format=None,
        tools=None,
        timeout_s=1.0,
    )


def test_estimate_embed_tokens_matches_generate_char_style() -> None:
    assert estimate_embed_tokens(["abcdef"]) == 2  # (6 + 2) // 3
    assert estimate_embed_tokens([]) == 1
    assert estimate_embed_tokens(["a", "bb", "ccc"]) == 2  # 6 chars


async def test_embed_acquires_tpm_and_records_success_on_breaker() -> None:
    inner = _RecordingProvider()
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0)
    managed = _managed(inner, breaker=breaker, tpm=100)

    before = managed.tokens.tokens
    vectors = await managed.embed("embed-model", ["hello world"])

    assert vectors == [[11.0]]
    assert inner.embed_calls == [("embed-model", ["hello world"])]
    assert managed.tokens.tokens < before
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failures == 0


async def test_embed_fails_closed_when_circuit_open_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr("evalharness.providers.runtime.time.monotonic", lambda: now)
    inner = _RecordingProvider()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=30.0)
    breaker.failure()
    assert breaker.state == CircuitState.OPEN
    managed = _managed(inner, breaker=breaker)

    with pytest.raises(CircuitOpenError, match="provider circuit is open"):
        await managed.embed("embed-model", ["should not reach inner"])

    assert inner.embed_calls == []


async def test_embed_records_breaker_failure_on_inner_error() -> None:
    inner = _RecordingProvider(fail_embed=True)
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=30.0)
    managed = _managed(inner, breaker=breaker)

    with pytest.raises(RuntimeError, match="embed backend failed"):
        await managed.embed("embed-model", ["x"])

    assert breaker.failures == 1
    assert breaker.state == CircuitState.CLOSED
    assert len(inner.embed_calls) == 1


def test_circuit_open_error_classifies_as_circuit_open_not_retryable() -> None:
    managed = _managed(_RecordingProvider())
    classified = managed.classify_error(CircuitOpenError("provider circuit is open"))

    assert classified == ErrorClass.CIRCUIT_OPEN
    assert classified is not ErrorClass.RETRYABLE_TRANSIENT
    assert classified is not ErrorClass.RETRYABLE_RATE_LIMIT


async def test_circuit_open_is_not_retried_by_call_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CIRCUIT_OPEN must terminate on first attempt; call_policy must not sleep/retry."""
    now = 100.0
    monkeypatch.setattr("evalharness.providers.runtime.time.monotonic", lambda: now)

    async def _sleep_must_not_run(_: float) -> None:
        raise AssertionError("circuit_open must not schedule a retry sleep")

    monkeypatch.setattr(
        "evalharness.providers.call_policy.asyncio.sleep",
        _sleep_must_not_run,
    )

    inner = _RecordingProvider()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=30.0)
    breaker.failure()
    assert breaker.state == CircuitState.OPEN
    managed = _managed(inner, breaker=breaker)
    policy = ProviderCallPolicy(
        request_timeout_s=1.0,
        max_retries=5,
        retry_base_s=0.0,
        retry_cap_s=0.0,
    )

    with pytest.raises(ProviderCallError, match="circuit_open") as exc_info:
        await generate_with_policy(
            managed,
            model="m",
            request=_generation_request(),
            policy=policy,
        )

    assert exc_info.value.attempts == 1
    assert inner.generate_calls == []
    assert isinstance(exc_info.value.__cause__, CircuitOpenError)
