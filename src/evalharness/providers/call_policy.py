"""Bounded provider calls for non-generation inference workflows."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from evalharness.core.enums import ErrorClass
from evalharness.core.models import GenerationRequest, GenerationResponse, ModelVersion
from evalharness.core.protocols import Provider
from evalharness.observability import StageTimer, exception_summary, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProviderCallPolicy:
    """Executor-equivalent bounds for idempotent inference calls."""

    request_timeout_s: float
    max_retries: int
    retry_base_s: float
    retry_cap_s: float


@dataclass(frozen=True)
class ProviderCallResult:
    """Successful provider response and retry accounting."""

    response: GenerationResponse
    attempts: int


class ProviderCallError(RuntimeError):
    """Terminal provider failure after applying the bounded call policy."""

    def __init__(self, message: str, *, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(message)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=UTC)
        except ValueError:
            return None
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


async def _call_with_policy[ResultT](
    provider: Provider,
    operation: Callable[[], Awaitable[ResultT]],
    policy: ProviderCallPolicy,
    *,
    model: str,
    operation_name: str,
) -> tuple[ResultT, int]:
    # Case, claim, and swap identifiers reach these events through the caller's
    # bound structlog context, so they stay out of this generic signature.
    call = {"operation": operation_name, "provider": provider.name, "model": model}
    for attempt in range(policy.max_retries + 1):
        timer = StageTimer()
        logger.debug(
            "provider_call_attempt_started",
            **call,
            attempt=attempt + 1,
            max_attempts=policy.max_retries + 1,
            timeout_s=policy.request_timeout_s,
        )
        try:
            result = await asyncio.wait_for(operation(), timeout=policy.request_timeout_s)
        except (TimeoutError, httpx.TimeoutException) as exc:
            logger.error(
                "provider_call_failed",
                **call,
                attempts=attempt + 1,
                error_class="timeout",
                duration_ms=timer.elapsed_ms,
                **exception_summary(exc),
            )
            raise ProviderCallError("provider request timed out", attempts=attempt + 1) from exc
        except Exception as exc:
            error_class = provider.classify_error(exc)
            retryable = error_class in {
                ErrorClass.RETRYABLE_TRANSIENT,
                ErrorClass.RETRYABLE_RATE_LIMIT,
            }
            if not retryable or attempt >= policy.max_retries:
                logger.error(
                    "provider_call_failed",
                    **call,
                    attempts=attempt + 1,
                    error_class=error_class.value,
                    duration_ms=timer.elapsed_ms,
                    **exception_summary(exc),
                )
                raise ProviderCallError(
                    f"provider request failed: {error_class.value}",
                    attempts=attempt + 1,
                ) from exc
            exponential_cap = min(policy.retry_cap_s, policy.retry_base_s * (2**attempt))
            jitter = random.uniform(0.0, exponential_cap)
            delay = max(jitter, _retry_after_seconds(exc) or 0.0)
            logger.warning(
                "provider_retry_scheduled",
                **call,
                attempt=attempt + 1,
                next_attempt=attempt + 2,
                error_class=error_class.value,
                delay_s=round(delay, 3),
                duration_ms=timer.elapsed_ms,
                **exception_summary(exc),
            )
            await asyncio.sleep(delay)
        else:
            logger.debug(
                "provider_call_finished",
                **call,
                attempts=attempt + 1,
                duration_ms=timer.elapsed_ms,
            )
            return result, attempt + 1
    raise RuntimeError("unreachable provider retry state")


async def generate_with_policy(
    provider: Provider,
    *,
    model: str,
    request: GenerationRequest,
    policy: ProviderCallPolicy,
) -> ProviderCallResult:
    """Generate with explicit timeout and bounded executor-equivalent retries."""

    response, attempts = await _call_with_policy(
        provider,
        lambda: provider.generate(model, request),
        policy,
        model=model,
        operation_name="generate",
    )
    return ProviderCallResult(response=response, attempts=attempts)


async def resolve_version_with_policy(
    provider: Provider,
    *,
    model: str,
    policy: ProviderCallPolicy,
) -> ModelVersion:
    """Resolve a real model digest under the same provider call bounds."""

    version, _ = await _call_with_policy(
        provider,
        lambda: provider.resolve_version(model),
        policy,
        model=model,
        operation_name="resolve_version",
    )
    return version


async def bounded_map[InputT, ResultT](
    inputs: Sequence[InputT],
    *,
    concurrency: int,
    operation: Callable[[InputT], Awaitable[ResultT]],
) -> list[ResultT]:
    """Map async work through a fixed-size worker pool."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    queue: asyncio.Queue[tuple[int, InputT] | None] = asyncio.Queue(maxsize=concurrency * 2)
    results: list[ResultT | None] = [None] * len(inputs)

    async def produce() -> None:
        for index, item in enumerate(inputs):
            await queue.put((index, item))
        for _ in range(concurrency):
            await queue.put(None)

    async def worker() -> None:
        while True:
            queued = await queue.get()
            try:
                if queued is None:
                    return
                index, item = queued
                results[index] = await operation(item)
            finally:
                queue.task_done()

    producer = asyncio.create_task(produce())
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(producer, *workers)
    except BaseException:
        producer.cancel()
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(producer, *workers, return_exceptions=True)
        raise
    if any(result is None for result in results):
        raise RuntimeError("provider worker pool produced an incomplete result")
    return [result for result in results if result is not None]


@dataclass(frozen=True)
class CostSummary:
    """Provider-reported cost and how many responses carried no price at all.

    ``known_usd_total`` is a floor rather than the true spend: a provider that
    returns no cost field contributes nothing to it. ``unpriced_responses`` is
    what tells an operator the total understates reality, since the published
    artifact schemas require a non-null number here.
    """

    known_usd_total: float
    unpriced_responses: int


def _response_cost_usd(response: GenerationResponse) -> float | None:
    """Provider-reported cost, or ``None`` when the provider priced nothing."""

    candidates: list[object] = [
        response.raw.get("cost_usd"),
        response.raw.get("cost"),
    ]
    usage = response.raw.get("usage")
    if isinstance(usage, dict):
        candidates.append(usage.get("cost_usd"))
        candidates.append(usage.get("cost"))
    chunks = response.raw.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, dict):
                candidates.append(chunk.get("cost_usd"))
                candidates.append(chunk.get("cost"))
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    candidates.append(chunk_usage.get("cost_usd"))
                    candidates.append(chunk_usage.get("cost"))
    for value in reversed(candidates):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
    return None


def summarize_cost(responses: Sequence[GenerationResponse]) -> CostSummary:
    """Sum only provider-reported cost, counting responses that reported none."""

    costs = [_response_cost_usd(response) for response in responses]
    return CostSummary(
        known_usd_total=float(sum(cost for cost in costs if cost is not None)),
        unpriced_responses=sum(1 for cost in costs if cost is None),
    )
