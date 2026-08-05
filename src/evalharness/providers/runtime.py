"""Provider rate limiting, concurrency control, and circuit breaking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from enum import StrEnum

from evalharness.core.enums import ErrorClass
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.core.protocols import Provider
from evalharness.observability import exception_summary, get_logger

logger = get_logger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    pass


class TokenBucket:
    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self.capacity = capacity
        self.tokens = capacity
        self.refill_per_second = refill_per_second
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> float:
        started = time.monotonic()
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
                self.updated_at = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return (time.monotonic() - started) * 1000
                delay = (amount - self.tokens) / max(self.refill_per_second, 1e-9)
            await asyncio.sleep(delay)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.opened_at: float | None = None

    def before_call(self) -> CircuitState:
        if self.state == CircuitState.OPEN:
            assert self.opened_at is not None
            if time.monotonic() - self.opened_at < self.recovery_timeout_s:
                logger.warning(
                    "provider_circuit_rejected",
                    state=self.state.value,
                    failures=self.failures,
                )
                raise CircuitOpenError("provider circuit is open")
            self.state = CircuitState.HALF_OPEN
            logger.info(
                "provider_circuit_state_changed",
                previous=CircuitState.OPEN.value,
                current=self.state.value,
                failures=self.failures,
            )
        return self.state

    def success(self) -> None:
        previous = self.state
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.opened_at = None
        if previous != self.state:
            logger.info(
                "provider_circuit_state_changed",
                previous=previous.value,
                current=self.state.value,
                failures=0,
            )

    def failure(self) -> None:
        previous = self.state
        self.failures += 1
        if self.state == CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
        if previous != self.state:
            logger.warning(
                "provider_circuit_state_changed",
                previous=previous.value,
                current=self.state.value,
                failures=self.failures,
                threshold=self.failure_threshold,
            )


def estimate_tokens(req: GenerationRequest) -> int:
    """Conservative tokenizer-independent estimate (roughly three chars/token)."""
    text = "".join(message.content for message in req.messages)
    return max(1, (len(text) + 2) // 3) + (req.max_tokens or 512)


class ManagedProvider:
    def __init__(
        self,
        provider: Provider,
        *,
        rpm: int,
        tpm: int,
        concurrency: int,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.provider = provider
        self.name = provider.name
        self.requests = TokenBucket(float(rpm), rpm / 60)
        self.tokens = TokenBucket(float(tpm), tpm / 60)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.breaker = breaker or CircuitBreaker()

    async def resolve_version(self, model: str) -> ModelVersion:
        return await self.provider.resolve_version(model)

    def capabilities(self, model: str) -> Capabilities:
        return self.provider.capabilities(model)

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse:
        state = self.breaker.before_call()
        queue_wait = await self.requests.acquire()
        queue_wait += await self.tokens.acquire(estimate_tokens(req))
        logger.debug(
            "provider_capacity_acquired",
            provider=self.name,
            model=model,
            queue_wait_ms=round(queue_wait, 2),
            estimated_tokens=estimate_tokens(req),
            breaker_state=state.value,
        )
        async with self.semaphore:
            try:
                response = await self.provider.generate(model, req)
            except Exception as exc:
                self.breaker.failure()
                logger.warning(
                    "managed_provider_call_failed",
                    provider=self.name,
                    model=model,
                    breaker_state=self.breaker.state.value,
                    **exception_summary(exc),
                )
                raise
            self.breaker.success()
        raw = dict(response.raw)
        raw["runtime"] = {
            "queue_wait_ms": queue_wait,
            "breaker_state_before": state.value,
            "estimated_tokens": estimate_tokens(req),
        }
        return replace(response, raw=raw)

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        await self.requests.acquire()
        async with self.semaphore:
            return await self.provider.embed(model, texts)

    def classify_error(self, exc: Exception) -> ErrorClass:
        if isinstance(exc, CircuitOpenError):
            return ErrorClass.RETRYABLE_TRANSIENT
        return self.provider.classify_error(exc)

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()
