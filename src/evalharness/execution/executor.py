"""Execution planner and resilient executor."""

from __future__ import annotations

import asyncio
import random
import signal
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from jinja2 import Environment

from evalharness.config import Settings, get_settings
from evalharness.core.enums import ErrorClass, FailureOutcome, FinishReason
from evalharness.core.models import (
    Case,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelVersion,
    ToolCall,
)
from evalharness.core.ports import RunStoreFactory
from evalharness.core.protocols import Provider
from evalharness.execution.errors import ResumeError as ResumeError
from evalharness.execution.helpers import validate_decode_params
from evalharness.hashing import config_hash, sha256_canonical
from evalharness.observability import (
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    StageTimer,
    emit_progress,
    exception_summary,
    get_logger,
    get_tracer,
    log_context,
    payload_summary,
)
from evalharness.providers.retry import retry_after_seconds
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository

logger = get_logger(__name__)


def response_cache_key(
    *,
    provider: str,
    resolved_version: str,
    rendered_prompt: str,
    decode_params: dict[str, Any],
) -> str:
    """Key for the shared response cache; callers that purge must use this same derivation."""
    return sha256_canonical(
        {
            "provider": provider,
            "model_version": resolved_version,
            "prompt": rendered_prompt,
            "decode": decode_params,
            "adapter": f"{provider}-v1",
        }
    )


def _response_from_cache(payload: dict[str, Any]) -> GenerationResponse:
    return GenerationResponse(
        text=payload["text"],
        tool_calls=[ToolCall(**call) for call in payload.get("tool_calls", [])],
        finish_reason=FinishReason(payload["finish_reason"]),
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        logprobs=None,
        ttft_ms=payload.get("ttft_ms"),
        total_ms=payload["total_ms"],
        raw=payload.get("raw", {}),
    )


@dataclass(frozen=True)
class RunPlanItem:
    case_db_id: int
    case: Case
    repeat_idx: int


@dataclass(frozen=True)
class RunConfig:
    dataset_id: int
    prompt_template_id: int
    model_version_id: int
    config_sha256: str
    decode_params: dict[str, Any]
    repeats: int
    concurrency: int
    case_timeout_s: float
    request_timeout_s: float
    run_timeout_s: float
    drain_timeout_s: float
    max_retries: int
    coverage_floor: float


@dataclass(frozen=True)
class ExecutionResult:
    case_id: int
    external_id: str
    repeat_idx: int
    outcome: FailureOutcome
    attempts: int
    cached: bool
    duration_ms: float | None


@dataclass(frozen=True)
class _AttemptOutcome:
    """Result of cache lookup or the provider retry loop for one case."""

    response: GenerationResponse | None
    attempt_log: list[dict[str, Any]]
    harness_error: bool
    harness_timeout: bool
    cached: bool


class GracefulShutdown:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.reason: str | None = None

    @property
    def requested(self) -> bool:
        return self.event.is_set()

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle)

    def _handle(self) -> None:
        self.request("signal")

    def request(self, reason: str) -> None:
        if self.event.is_set():
            return
        self.reason = reason
        self.event.set()
        logger.info("shutdown_requested")


def render_prompt(template: str, case: Case) -> str:
    """Render a trusted local template against case inputs."""
    return Environment(autoescape=False).from_string(template).render(**case.inputs)


def classify_outcome(
    *,
    output: str | None,
    finish_reason: FinishReason | None,
    harness_error: bool,
    harness_timeout: bool,
) -> FailureOutcome:
    if harness_timeout:
        return FailureOutcome.HARNESS_TIMEOUT
    if harness_error:
        return FailureOutcome.HARNESS_ERROR
    if not output or not output.strip():
        return FailureOutcome.EMPTY_OUTPUT
    if finish_reason == FinishReason.LENGTH:
        return FailureOutcome.TRUNCATED
    if finish_reason == FinishReason.CONTENT_FILTER:
        return FailureOutcome.CONTENT_FILTERED
    if output.strip().lower().startswith("i can't") or output.strip().lower().startswith(
        "i cannot"
    ):
        return FailureOutcome.REFUSED
    return FailureOutcome.PASSED


async def _retry_delay(attempt: int, base: float, cap: float) -> float:
    exp = min(cap, base * (2**attempt))
    return random.uniform(0, exp)


class Executor:
    def __init__(
        self,
        provider: Provider,
        model: str,
        model_version: ModelVersion,
        template_body: str,
        *,
        settings: Settings | None = None,
        run_store: RunStoreFactory | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.model_version = model_version
        self.template_body = template_body
        self.settings = settings or get_settings()
        self.run_store: RunStoreFactory = run_store or RunRepository
        self.tracer = get_tracer()
        self.shutdown = GracefulShutdown()

    async def create_run(
        self,
        *,
        bundle_dataset_id: int,
        prompt_template_id: int,
        model_version_id: int,
        dataset_sha256: str,
        prompt_template_sha256: str,
        decode_params: dict[str, Any],
        repeats: int,
        tenant_id: str,
        run_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        validate_decode_params(decode_params)
        cfg_sha = config_hash(
            dataset_sha256=dataset_sha256,
            prompt_template_sha256=prompt_template_sha256,
            provider=self.model_version.provider,
            model=self.model_version.model,
            resolved_version=self.model_version.resolved_version,
            decode_params=decode_params,
            harness_version=self.settings.harness_version,
        )
        async with session_scope() as session:
            repo = self.run_store(session)
            rid = await repo.create_run(
                dataset_id=bundle_dataset_id,
                prompt_template_id=prompt_template_id,
                model_version_id=model_version_id,
                decode_params=decode_params,
                config_sha256=cfg_sha,
                harness_version=self.settings.harness_version,
                git_sha=self.settings.git_sha,
                repeats=repeats,
                tenant_id=tenant_id,
                run_id=run_id,
            )
            await repo.update_run_status(rid, "running")
            logger.info(
                "run_created",
                run_id=str(rid),
                tenant_id=tenant_id,
                dataset_id=bundle_dataset_id,
                model=self.model,
                model_digest=self.model_version.resolved_version,
                repeats=repeats,
                config_sha256=cfg_sha,
            )
            return rid

    async def plan(self, run_id: uuid.UUID) -> tuple[RunConfig, list[RunPlanItem]]:
        async with session_scope() as session:
            repo = self.run_store(session)
            run = await repo.get_run(run_id)
            if not run:
                raise ValueError(f"Run not found: {run_id}")
            completed = await repo.get_completed_keys(run_id)
            cases = await repo.get_cases_for_dataset(run.dataset_id)
            items: list[RunPlanItem] = []
            for case_db_id, case in cases:
                for repeat_idx in range(run.repeats):
                    if (case_db_id, repeat_idx) not in completed:
                        items.append(
                            RunPlanItem(case_db_id=case_db_id, case=case, repeat_idx=repeat_idx)
                        )
            config = RunConfig(
                dataset_id=run.dataset_id,
                prompt_template_id=run.prompt_template_id,
                model_version_id=run.model_version_id,
                config_sha256=run.config_sha256,
                decode_params=run.decode_params,
                repeats=run.repeats,
                concurrency=self.settings.default_concurrency,
                case_timeout_s=self.settings.default_case_timeout_s,
                request_timeout_s=self.settings.default_request_timeout_s,
                run_timeout_s=self.settings.default_run_timeout_s,
                drain_timeout_s=self.settings.default_shutdown_drain_timeout_s,
                max_retries=self.settings.default_max_retries,
                coverage_floor=self.settings.default_coverage_floor,
            )
            return config, items

    async def validate_resume(
        self,
        run_id: uuid.UUID,
        *,
        dataset_id: int,
        prompt_template_id: int,
        model_version_id: int,
        decode_params: dict[str, Any],
        repeats: int,
        tenant_id: str,
    ) -> None:
        """Refuse a resume when any generation-affecting input changed."""
        async with session_scope() as session:
            run = await self.run_store(session).get_run(run_id)
            if run is None:
                raise ValueError(f"Run not found: {run_id}")
            expected = {
                "dataset_id": dataset_id,
                "prompt_template_id": prompt_template_id,
                "model_version_id": model_version_id,
                "decode_params": decode_params,
                "repeats": repeats,
                "tenant_id": tenant_id,
            }
            actual = {key: getattr(run, key) for key in expected}
            mismatches = [key for key in expected if actual[key] != expected[key]]
            if mismatches:
                raise ValueError(
                    "Resume configuration mismatch for: " + ", ".join(sorted(mismatches))
                )

    async def execute_run(
        self,
        run_id: uuid.UUID,
        concurrency: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Execute a run under context that is guaranteed not to leak to its caller."""
        with log_context(
            run_id=str(run_id),
            provider=self.model_version.provider,
            model=self.model,
        ):
            await self._execute_run_with_context(run_id, concurrency, progress)

    async def _execute_run_with_context(
        self,
        run_id: uuid.UUID,
        concurrency: int | None,
        progress: ProgressCallback | None,
    ) -> None:
        self.shutdown.install()
        config, items = await self.plan(run_id)
        worker_count = max(1, concurrency or config.concurrency)
        timer = StageTimer()
        logger.info(
            "generation_started",
            planned=len(items),
            concurrency=worker_count,
        )
        emit_progress(
            progress,
            ProgressEvent(PipelineStage.GENERATING, 0, len(items), "Generating responses"),
        )
        with self.tracer.start_as_current_span("run.generate") as run_span:
            run_span.set_attribute("eval.run_id", str(run_id))
            run_span.set_attribute("eval.planned_generations", len(items))
            pipeline = asyncio.create_task(
                self._run_worker_pool(run_id, config, items, worker_count, progress),
                name=f"run-{run_id}-pipeline",
            )
            shutdown_wait = asyncio.create_task(self.shutdown.event.wait(), name="shutdown-wait")
            worker_failures = False
            try:
                done, _ = await asyncio.wait(
                    {pipeline, shutdown_wait},
                    timeout=config.run_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if pipeline in done:
                    results = await pipeline
                    worker_failures = any(isinstance(result, BaseException) for result in results)
                elif shutdown_wait in done:
                    try:
                        results = await asyncio.wait_for(pipeline, timeout=config.drain_timeout_s)
                        worker_failures = any(
                            isinstance(result, BaseException) for result in results
                        )
                    except TimeoutError:
                        pipeline.cancel()
                        await asyncio.gather(pipeline, return_exceptions=True)
                else:
                    self.shutdown.request("run_deadline")
                    pipeline.cancel()
                    await asyncio.gather(pipeline, return_exceptions=True)
            finally:
                shutdown_wait.cancel()
                await asyncio.gather(shutdown_wait, return_exceptions=True)

        async with session_scope() as session:
            repo = self.run_store(session)
            remaining = len((await self.plan(run_id))[1])
            if remaining == 0 and not worker_failures:
                status = "completed"
            elif self.shutdown.requested:
                status = "cancelled"
            else:
                status = "failed"
            await repo.update_run_status(run_id, status)
        logger.info(
            "generation_finished",
            status=status,
            remaining=remaining,
            duration_ms=timer.elapsed_ms,
        )

    async def _run_worker_pool(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        items: list[RunPlanItem],
        worker_count: int,
        progress: ProgressCallback | None,
    ) -> list[Any]:
        queue: asyncio.Queue[RunPlanItem | None] = asyncio.Queue(maxsize=worker_count * 2)
        total = len(items)
        completed = 0
        outcomes: Counter[str] = Counter()
        retries = 0
        cache_hits = 0

        async def produce() -> None:
            for item in items:
                if self.shutdown.requested:
                    break
                await queue.put(item)
            for _ in range(worker_count):
                await queue.put(None)

        async def worker() -> None:
            nonlocal completed, retries, cache_hits
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    result = await self._run_one(run_id, config, item)
                    completed += 1
                    outcomes[result.outcome.value] += 1
                    retries += max(0, result.attempts - 1)
                    cache_hits += int(result.cached)
                    if completed == total or completed % self.settings.log_progress_every == 0:
                        logger.info(
                            "generation_progress",
                            completed=completed,
                            total=total,
                            valid_outputs=outcomes[FailureOutcome.PASSED.value],
                            other_outcomes=completed - outcomes[FailureOutcome.PASSED.value],
                            retries=retries,
                            cache_hits=cache_hits,
                        )
                    emit_progress(
                        progress,
                        ProgressEvent(
                            PipelineStage.GENERATING,
                            completed,
                            total,
                            result.external_id,
                            {
                                "valid_outputs": outcomes[FailureOutcome.PASSED.value],
                                "other_outcomes": completed - outcomes[FailureOutcome.PASSED.value],
                                "retries": retries,
                                "cache_hits": cache_hits,
                            },
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "case_worker_failed",
                        run_id=str(run_id),
                        case_id=item.case_db_id if item else None,
                    )
                finally:
                    queue.task_done()

        producer = asyncio.create_task(produce(), name=f"run-{run_id}-producer")
        workers = [
            asyncio.create_task(worker(), name=f"run-{run_id}-worker-{idx}")
            for idx in range(worker_count)
        ]
        return await asyncio.gather(producer, *workers, return_exceptions=True)

    async def _run_one(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        item: RunPlanItem,
        sem: asyncio.Semaphore | None = None,
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
            )
        if sem is not None:
            await sem.acquire()
        timer = StageTimer()
        try:
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
                        result = await asyncio.wait_for(
                            self._execute_case(run_id, config, item, trace_id),
                            timeout=config.case_timeout_s,
                        )
                        return result
                    except TimeoutError:
                        result = await self._save_terminal_failure(
                            run_id,
                            item,
                            FailureOutcome.HARNESS_TIMEOUT,
                            trace_id,
                            "case_timeout",
                        )
                        return result
                    except Exception as exc:
                        logger.exception(
                            "case_execution_failed",
                            duration_ms=timer.elapsed_ms,
                            **exception_summary(exc),
                        )
                        result = await self._save_terminal_failure(
                            run_id,
                            item,
                            FailureOutcome.HARNESS_ERROR,
                            trace_id,
                            type(exc).__name__,
                        )
                        return result
        finally:
            if sem is not None:
                sem.release()

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
                cost_usd=0.0,
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
        return _response_from_cache(cached_payload)

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
            start = datetime.now(UTC)
            attempt_timer = StageTimer()
            logger.debug("provider_attempt_started", attempt=attempt + 1)
            try:
                with self.tracer.start_as_current_span("provider.call") as provider_span:
                    provider_span.set_attribute("gen_ai.request.model", self.model)
                    provider_span.set_attribute("eval.attempt", attempt + 1)
                    response = await asyncio.wait_for(
                        self.provider.generate(self.model, req),
                        timeout=config.request_timeout_s,
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
                harness_timeout = True
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
                break
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
                if attempt < config.max_retries:
                    jitter = await _retry_delay(
                        attempt,
                        self.settings.default_retry_base_s,
                        self.settings.default_retry_cap_s,
                    )
                    retry_after = retry_after_seconds(exc)
                    delay = max(jitter, retry_after or 0.0)
                    logger.warning(
                        "provider_retry_scheduled",
                        attempt=attempt + 1,
                        next_attempt=attempt + 2,
                        error_class=error_class.value,
                        delay_s=round(delay, 3),
                        **exception_summary(exc),
                    )
                    await asyncio.sleep(delay)
                else:
                    harness_error = True

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
                cost_usd=0.0,
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
