"""Provider call loop for one case: retries, backoff, and the case budget."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from opentelemetry.trace import Tracer

from evalharness.app.settings import Settings
from evalharness.domain.enums import ErrorClass
from evalharness.domain.generation import GenerationRequest, GenerationResponse, Message
from evalharness.domain.provider import Provider
from evalharness.execution.models import AttemptOutcome, RunConfig
from evalharness.observability import StageTimer, exception_summary, get_logger
from evalharness.providers.retry import retry_after_seconds, retry_delay_seconds

logger = get_logger(__name__)

RETRYABLE = (ErrorClass.RETRYABLE_TRANSIENT, ErrorClass.RETRYABLE_RATE_LIMIT)


def build_request(config: RunConfig, rendered: str) -> GenerationRequest:
    return GenerationRequest(
        messages=[Message(role="user", content=rendered)],
        max_tokens=config.decode_params.get("max_tokens"),
        temperature=float(config.decode_params.get("temperature", 0.0)),
        top_p=config.decode_params.get("top_p"),
        top_k=config.decode_params.get("top_k"),
        seed=config.decode_params.get("seed"),
        stop=config.decode_params.get("stop", []),
        response_format=None,
        tools=None,
        timeout_s=config.request_timeout_s,
    )


async def _call_provider(
    *,
    provider: Provider,
    model: str,
    tracer: Tracer,
    req: GenerationRequest,
    attempt: int,
    timeout_s: float,
) -> GenerationResponse:
    with tracer.start_as_current_span("provider.call") as provider_span:
        provider_span.set_attribute("gen_ai.request.model", model)
        provider_span.set_attribute("eval.attempt", attempt + 1)
        response = await asyncio.wait_for(provider.generate(model, req), timeout=timeout_s)
        if response.prompt_tokens is not None:
            provider_span.set_attribute("gen_ai.usage.input_tokens", response.prompt_tokens)
        if response.completion_tokens is not None:
            provider_span.set_attribute("gen_ai.usage.output_tokens", response.completion_tokens)
    return response


async def _sleep_before_retry(
    *,
    attempt: int,
    error_class: str,
    exc: Exception,
    settings: Settings,
    remaining_s: float,
    retry_after_s: float | None,
) -> None:
    delay = retry_delay_seconds(
        attempt,
        base_s=settings.default_retry_base_s,
        cap_s=settings.default_retry_cap_s,
        retry_after_s=retry_after_s,
        remaining_budget_s=remaining_s,
    )
    logger.warning(
        "provider_retry_scheduled",
        attempt=attempt + 1,
        next_attempt=attempt + 2,
        error_class=error_class,
        delay_s=round(delay, 3),
        **exception_summary(exc),
    )
    if delay > 0:
        await asyncio.sleep(delay)


async def generate_with_retries(
    *,
    provider: Provider,
    model: str,
    tracer: Tracer,
    settings: Settings,
    config: RunConfig,
    rendered: str,
    case_deadline: float,
) -> AttemptOutcome:
    """Call the provider until it answers, the retries run out, or the case budget expires.

    The budget is checked inside the loop rather than by wrapping the whole call, so an
    expiry still returns the attempt log instead of cancelling it away.
    """
    req = build_request(config, rendered)
    attempt_log: list[dict[str, Any]] = []
    response: GenerationResponse | None = None
    harness_error = False
    harness_timeout = False

    for attempt in range(config.max_retries + 1):
        remaining = case_deadline - time.monotonic()
        if remaining <= 0:
            harness_timeout = True
            break
        start = datetime.now(UTC)
        attempt_timer = StageTimer()
        logger.debug("provider_attempt_started", attempt=attempt + 1)
        try:
            response = await _call_provider(
                provider=provider,
                model=model,
                tracer=tracer,
                req=req,
                attempt=attempt,
                timeout_s=min(config.request_timeout_s, remaining),
            )
            attempt_log.append(
                {
                    "attempt": attempt + 1,
                    "error_class": None,
                    "duration_ms": response.total_ms,
                    "at": start.isoformat(),
                }
            )
            logger.debug(
                "provider_attempt_finished",
                attempt=attempt + 1,
                duration_ms=attempt_timer.elapsed_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                finish_reason=response.finish_reason.value,
            )
            break
        except (TimeoutError, httpx.TimeoutException) as exc:
            attempt_log.append(
                {
                    "attempt": attempt + 1,
                    "error_class": "timeout",
                    "duration_ms": None,
                    "at": start.isoformat(),
                    **exception_summary(exc),
                }
            )
            logger.warning(
                "provider_attempt_finished",
                attempt=attempt + 1,
                duration_ms=attempt_timer.elapsed_ms,
                error_class="timeout",
                **exception_summary(exc),
            )
            remaining = case_deadline - time.monotonic()
            if attempt >= config.max_retries or remaining <= 0:
                harness_timeout = True
                break
            await _sleep_before_retry(
                attempt=attempt,
                error_class="timeout",
                exc=exc,
                settings=settings,
                remaining_s=remaining,
                retry_after_s=None,
            )
        except Exception as exc:
            error_class = provider.classify_error(exc)
            attempt_log.append(
                {
                    "attempt": attempt + 1,
                    "error_class": error_class.value,
                    "duration_ms": None,
                    "at": start.isoformat(),
                    **exception_summary(exc),
                }
            )
            if error_class not in RETRYABLE:
                harness_error = True
                break
            remaining = case_deadline - time.monotonic()
            if remaining <= 0:
                harness_timeout = True
                break
            if attempt >= config.max_retries:
                harness_error = True
                break
            await _sleep_before_retry(
                attempt=attempt,
                error_class=error_class.value,
                exc=exc,
                settings=settings,
                remaining_s=remaining,
                retry_after_s=retry_after_seconds(exc),
            )

    return AttemptOutcome(
        response=response,
        attempt_log=attempt_log,
        harness_error=harness_error,
        harness_timeout=harness_timeout,
        cached=False,
    )
