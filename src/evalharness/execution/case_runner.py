"""Per-case generation: cache, retries, and persist."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from opentelemetry.trace import Tracer

from evalharness.config import Settings
from evalharness.core.enums import ErrorClass, FailureOutcome
from evalharness.core.models import (
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelVersion,
)
from evalharness.core.ports import RunStoreFactory
from evalharness.core.protocols import Provider
from evalharness.execution.helpers import (
    ExecutionResult,
    RunConfig,
    RunPlanItem,
    classify_outcome,
    render_prompt,
    response_cache_key,
    response_from_cache,
)
from evalharness.execution.shutdown import GracefulShutdown
from evalharness.observability import (
    StageTimer,
    exception_summary,
    get_logger,
    log_context,
    payload_summary,
)
from evalharness.providers.call_policy import _response_cost_usd
from evalharness.providers.retry import retry_after_seconds, retry_delay_seconds
from evalharness.store.db import session_scope

logger = get_logger(__name__)


@dataclass(frozen=True)
class _AttemptOutcome:
    """Result of cache lookup or the provider retry loop for one case."""

    response: GenerationResponse | None
    attempt_log: list[dict[str, Any]]
    harness_error: bool
    harness_timeout: bool
    cached: bool


class CaseRunner:
    """Runs a single plan item through cache / generate / classify / persist."""

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        model_version: ModelVersion,
        template_body: str,
        settings: Settings,
        run_store: RunStoreFactory,
        tracer: Tracer,
        shutdown: GracefulShutdown,
    ) -> None:
        self.provider = provider
        self.model = model
        self.model_version = model_version
        self.template_body = template_body
        self.settings = settings
        self.run_store = run_store
        self.tracer = tracer
        self.shutdown = shutdown

    async def run_one(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        item: RunPlanItem,
    ) -> ExecutionResult:
        if self.shutdown.requested:
            return ExecutionResult(
                item.case_db_id,
                item.case.external_id,
                item.repeat_idx,
                FailureOutcome.HARNESS_ERROR,
                0,
                False,
                None,
                persisted=False,
            )
        timer = StageTimer()
        with log_context(
            case_id=item.case_db_id,
            case_external_id=item.case.external_id,
            repeat_idx=item.repeat_idx,
        ):
            logger.debug("case_started")
            with self.tracer.start_as_current_span("case") as span:
                span.set_attribute("gen_ai.request.model", self.model)
                span.set_attribute("case.external_id", item.case.external_id)
                trace_id = format(span.get_span_context().trace_id, "032x")
                case_deadline = time.monotonic() + config.case_timeout_s
                try:
                    # Case budget is enforced inside the retry loop so attempt_log
                    # survives expiry (outer wait_for would cancel and drop it).
                    return await self._execute_case(
                        run_id,
                        config,
                        item,
                        trace_id,
                        case_deadline=case_deadline,
                    )
                except Exception as exc:
                    logger.exception(
                        "case_execution_failed",
                        duration_ms=timer.elapsed_ms,
                        **exception_summary(exc),
                    )
                    return await self._save_terminal_failure(
                        run_id,
                        item,
                        FailureOutcome.HARNESS_ERROR,
                        trace_id,
                        type(exc).__name__,
                    )

    async def _save_terminal_failure(
        self,
        run_id: uuid.UUID,
        item: RunPlanItem,
        outcome: FailureOutcome,
        trace_id: str,
        reason: str,
    ) -> ExecutionResult:
        async with session_scope() as session:
            await self.run_store(session).save_generation(
                run_id=run_id,
                case_id=item.case_db_id,
                repeat_idx=item.repeat_idx,
                output=None,
                tool_calls=[],
                finish_reason=None,
                outcome=outcome,
                prompt_tokens=None,
                completion_tokens=None,
                cost_usd=None,
                ttft_ms=None,
                total_ms=None,
                queue_wait_ms=None,
                attempts=1,
                attempt_log=[{"attempt": 1, "error_class": reason}],
                cached=False,
                raw_response=None,
                trace_id=trace_id,
            )
        logger.warning(
            "case_finished",
            outcome=outcome.value,
            attempts=1,
            cached=False,
            reason=reason,
            prompt=payload_summary(render_prompt(self.template_body, item.case)),
            output=payload_summary(None),
            trace_id=trace_id,
        )
        return ExecutionResult(
            item.case_db_id,
            item.case.external_id,
            item.repeat_idx,
            outcome,
            1,
            False,
            None,
        )

    async def _execute_case(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        item: RunPlanItem,
        trace_id: str,
        *,
        case_deadline: float,
    ) -> ExecutionResult:
        rendered = render_prompt(self.template_body, item.case)
        case_timer = StageTimer()
        logger.debug("case_input_ready", prompt=payload_summary(rendered))
        cache_key = response_cache_key(
            provider=self.model_version.provider,
            resolved_version=self.model_version.resolved_version,
            rendered_prompt=rendered,
            decode_params=config.decode_params,
        )
        cache_enabled = float(config.decode_params.get("temperature", 0.0)) == 0.0

        cached_response = await self._load_cached_response(cache_key) if cache_enabled else None
        if cached_response is not None:
            attempt = _AttemptOutcome(
                response=cached_response,
                attempt_log=[],
                harness_error=False,
                harness_timeout=False,
                cached=True,
            )
        else:
            attempt = await self._generate_with_retries(
                config=config,
                rendered=rendered,
                cache_key=cache_key,
                cache_enabled=cache_enabled,
                case_deadline=case_deadline,
            )

        response = attempt.response
        outcome = classify_outcome(
            output=response.text if response else None,
            finish_reason=response.finish_reason if response else None,
            harness_error=attempt.harness_error,
            harness_timeout=attempt.harness_timeout,
        )
        await self._persist_generation(
            run_id=run_id,
            item=item,
            response=response,
            outcome=outcome,
            attempt_log=attempt.attempt_log,
            cached=attempt.cached,
            trace_id=trace_id,
        )
        attempts = len(attempt.attempt_log) or 1
        case_log = logger.debug if outcome == FailureOutcome.PASSED else logger.warning
        case_log(
            "case_finished",
            outcome=outcome.value,
            attempts=attempts,
            cached=attempt.cached,
            duration_ms=case_timer.elapsed_ms,
            prompt_tokens=response.prompt_tokens if response else None,
            completion_tokens=response.completion_tokens if response else None,
            finish_reason=response.finish_reason.value if response else None,
            prompt=payload_summary(rendered),
            output=payload_summary(response.text if response else None),
            trace_id=trace_id,
        )
        return ExecutionResult(
            item.case_db_id,
            item.case.external_id,
            item.repeat_idx,
            outcome,
            attempts,
            attempt.cached,
            response.total_ms if response else None,
        )

    async def _load_cached_response(self, cache_key: str) -> GenerationResponse | None:
        async with session_scope() as session:
            cached_payload = await self.run_store(session).get_cache(cache_key)
        if not cached_payload:
            return None
        logger.debug("cache_hit", cache_key=cache_key)
        return response_from_cache(cached_payload)

    async def _put_cached_response(self, cache_key: str, response: GenerationResponse) -> None:
        async with session_scope() as session:
            await self.run_store(session).put_cache(
                cache_key,
                {
                    "text": response.text,
                    "tool_calls": [asdict(call) for call in response.tool_calls],
                    "finish_reason": response.finish_reason.value,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "logprobs": None,
                    "ttft_ms": response.ttft_ms,
                    "total_ms": response.total_ms,
                    "raw": response.raw,
                },
            )

    async def _generate_with_retries(
        self,
        *,
        config: RunConfig,
        rendered: str,
        cache_key: str,
        cache_enabled: bool,
        case_deadline: float,
    ) -> _AttemptOutcome:
        messages = [Message(role="user", content=rendered)]
        req = GenerationRequest(
            messages=messages,
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
            request_budget = min(config.request_timeout_s, remaining)
            try:
                with self.tracer.start_as_current_span("provider.call") as provider_span:
                    provider_span.set_attribute("gen_ai.request.model", self.model)
                    provider_span.set_attribute("eval.attempt", attempt + 1)
                    response = await asyncio.wait_for(
                        self.provider.generate(self.model, req),
                        timeout=request_budget,
                    )
                    if response.prompt_tokens is not None:
                        provider_span.set_attribute(
                            "gen_ai.usage.input_tokens", response.prompt_tokens
                        )
                    if response.completion_tokens is not None:
                        provider_span.set_attribute(
                            "gen_ai.usage.output_tokens", response.completion_tokens
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
                if cache_enabled:
                    await self._put_cached_response(cache_key, response)
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
                delay = retry_delay_seconds(
                    attempt,
                    base_s=self.settings.default_retry_base_s,
                    cap_s=self.settings.default_retry_cap_s,
                    remaining_budget_s=remaining,
                )
                logger.warning(
                    "provider_retry_scheduled",
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    error_class="timeout",
                    delay_s=round(delay, 3),
                    **exception_summary(exc),
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            except Exception as exc:
                error_class = self.provider.classify_error(exc)
                attempt_log.append(
                    {
                        "attempt": attempt + 1,
                        "error_class": error_class.value,
                        "duration_ms": None,
                        "at": start.isoformat(),
                        **exception_summary(exc),
                    }
                )
                if error_class not in (
                    ErrorClass.RETRYABLE_TRANSIENT,
                    ErrorClass.RETRYABLE_RATE_LIMIT,
                ):
                    harness_error = True
                    break
                remaining = case_deadline - time.monotonic()
                if remaining <= 0:
                    harness_timeout = True
                    break
                if attempt >= config.max_retries:
                    harness_error = True
                    break
                delay = retry_delay_seconds(
                    attempt,
                    base_s=self.settings.default_retry_base_s,
                    cap_s=self.settings.default_retry_cap_s,
                    retry_after_s=retry_after_seconds(exc),
                    remaining_budget_s=remaining,
                )
                logger.warning(
                    "provider_retry_scheduled",
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    error_class=error_class.value,
                    delay_s=round(delay, 3),
                    **exception_summary(exc),
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        return _AttemptOutcome(
            response=response,
            attempt_log=attempt_log,
            harness_error=harness_error,
            harness_timeout=harness_timeout,
            cached=False,
        )

    async def _persist_generation(
        self,
        *,
        run_id: uuid.UUID,
        item: RunPlanItem,
        response: GenerationResponse | None,
        outcome: FailureOutcome,
        attempt_log: list[dict[str, Any]],
        cached: bool,
        trace_id: str,
    ) -> None:
        async with session_scope() as session:
            await self.run_store(session).save_generation(
                run_id=run_id,
                case_id=item.case_db_id,
                repeat_idx=item.repeat_idx,
                output=response.text if response else None,
                tool_calls=[asdict(call) for call in response.tool_calls] if response else [],
                finish_reason=response.finish_reason if response else None,
                outcome=outcome,
                prompt_tokens=response.prompt_tokens if response else None,
                completion_tokens=response.completion_tokens if response else None,
                cost_usd=_response_cost_usd(response) if response else None,
                ttft_ms=response.ttft_ms if response else None,
                total_ms=response.total_ms if response else None,
                queue_wait_ms=(response.raw.get("runtime") or {}).get("queue_wait_ms")
                if response
                else None,
                attempts=len(attempt_log) or 1,
                attempt_log=attempt_log,
                cached=cached,
                raw_response=response.raw if response else None,
                trace_id=trace_id,
            )
