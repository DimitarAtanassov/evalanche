"""Per-case generation: cache, retries, classify, persist."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

from opentelemetry.trace import Tracer

from evalharness.app.settings import Settings
from evalharness.domain.enums import FailureOutcome
from evalharness.domain.generation import GenerationResponse, ModelVersion
from evalharness.domain.ports import RunStoreFactory
from evalharness.domain.provider import Provider
from evalharness.execution.attempts import generate_with_retries
from evalharness.execution.cache import (
    cache_enabled_for,
    load_cached_response,
    response_cache_key,
    store_cached_response,
)
from evalharness.execution.helpers import classify_outcome, render_prompt
from evalharness.execution.models import AttemptOutcome, ExecutionResult, RunConfig, RunPlanItem
from evalharness.execution.shutdown import GracefulShutdown
from evalharness.observability import (
    StageTimer,
    exception_summary,
    get_logger,
    log_context,
    payload_summary,
)
from evalharness.providers.call_policy import _response_cost_usd
from evalharness.db.session import session_scope

logger = get_logger(__name__)


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
                try:
                    # Case budget is enforced inside the retry loop so attempt_log
                    # survives expiry (outer wait_for would cancel and drop it).
                    return await self._execute_case(
                        run_id,
                        config,
                        item,
                        trace_id,
                        case_deadline=time.monotonic() + config.case_timeout_s,
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
        attempt = await self._attempt(config, rendered, case_deadline=case_deadline)

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

    async def _attempt(
        self,
        config: RunConfig,
        rendered: str,
        *,
        case_deadline: float,
    ) -> AttemptOutcome:
        """Serve the case from cache when it is reusable, otherwise call the provider."""
        cache_key = response_cache_key(
            provider=self.model_version.provider,
            resolved_version=self.model_version.resolved_version,
            rendered_prompt=rendered,
            decode_params=config.decode_params,
        )
        cache_enabled = cache_enabled_for(config.decode_params)

        if cache_enabled:
            cached_response = await load_cached_response(self.run_store, cache_key)
            if cached_response is not None:
                return AttemptOutcome(
                    response=cached_response,
                    attempt_log=[],
                    harness_error=False,
                    harness_timeout=False,
                    cached=True,
                )

        attempt = await generate_with_retries(
            provider=self.provider,
            model=self.model,
            tracer=self.tracer,
            settings=self.settings,
            config=config,
            rendered=rendered,
            case_deadline=case_deadline,
        )
        if cache_enabled and attempt.response is not None:
            await store_cached_response(self.run_store, cache_key, attempt.response)
        return attempt

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
