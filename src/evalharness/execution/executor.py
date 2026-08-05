"""Execution planner and resilient executor."""

from __future__ import annotations

import asyncio
import random
import signal
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update

from evalharness.config import get_settings
from evalharness.core.enums import ErrorClass, FailureOutcome, FinishReason
from evalharness.core.models import (
    Case,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelVersion,
    ScoringContext,
)
from evalharness.core.protocols import Provider
from evalharness.hashing import config_hash, sha256_canonical
from evalharness.observability import get_logger, get_tracer
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig
from evalharness.store.blob import BlobStore, blob_key_for_raw, get_blob_store
from evalharness.store.db import session_scope
from evalharness.store.models import GenerationRow
from evalharness.store.repository import RunRepository

logger = get_logger(__name__)


def _response_from_cache(payload: dict[str, Any]) -> GenerationResponse:
    return GenerationResponse(
        text=payload["text"],
        tool_calls=[],
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
    max_retries: int
    coverage_floor: float


class GracefulShutdown:
    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle)

    def _handle(self) -> None:
        self.requested = True
        logger.info("shutdown_requested")


def render_prompt(template: str, case: Case) -> str:
    rendered = template
    for key, value in case.inputs.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


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
        blob_store: BlobStore | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.model_version = model_version
        self.template_body = template_body
        self.blob_store = blob_store or get_blob_store()
        self.settings = get_settings()
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
            repo = RunRepository(session)
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
            return rid

    async def plan(self, run_id: uuid.UUID) -> tuple[RunConfig, list[RunPlanItem]]:
        async with session_scope() as session:
            repo = RunRepository(session)
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
                max_retries=self.settings.default_max_retries,
                coverage_floor=self.settings.default_coverage_floor,
            )
            return config, items

    async def execute_run(self, run_id: uuid.UUID, concurrency: int | None = None) -> None:
        self.shutdown.install()
        config, items = await self.plan(run_id)
        sem = asyncio.Semaphore(concurrency or config.concurrency)
        tasks = [self._run_one(run_id, config, item, sem) for item in items]
        await asyncio.gather(*tasks)
        async with session_scope() as session:
            repo = RunRepository(session)
            await repo.update_run_status(run_id, "completed")

    async def _run_one(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        item: RunPlanItem,
        sem: asyncio.Semaphore,
    ) -> None:
        if self.shutdown.requested:
            return
        async with sem:
            with self.tracer.start_as_current_span("case") as span:
                span.set_attribute("gen_ai.request.model", self.model)
                span.set_attribute("case.external_id", item.case.external_id)
                trace_id = format(span.get_span_context().trace_id, "032x")
                await asyncio.wait_for(
                    self._execute_case(run_id, config, item, trace_id),
                    timeout=config.case_timeout_s,
                )

    async def _execute_case(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        item: RunPlanItem,
        trace_id: str,
    ) -> None:
        rendered = render_prompt(self.template_body, item.case)
        cache_key = sha256_canonical(
            {
                "provider": self.model_version.provider,
                "model_version": self.model_version.resolved_version,
                "prompt": rendered,
                "decode": config.decode_params,
                "adapter": "ollama-v1",
            }
        )

        attempt_log: list[dict[str, Any]] = []
        cached = False
        response = None
        harness_error = False
        harness_timeout = False

        async with session_scope() as session:
            repo = RunRepository(session)
            cached_payload = await repo.get_cache(cache_key)
            if cached_payload:
                cached = True
                response = _response_from_cache(cached_payload)

        if response is None:
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
            for attempt in range(config.max_retries + 1):
                start = datetime.now(UTC)
                try:
                    with self.tracer.start_as_current_span("provider.call"):
                        response = await self.provider.generate(self.model, req)
                    attempt_log.append(
                        {
                            "attempt": attempt + 1,
                            "error_class": None,
                            "duration_ms": response.total_ms,
                            "at": start.isoformat(),
                        }
                    )
                    async with session_scope() as session:
                        repo = RunRepository(session)
                        await repo.put_cache(
                            cache_key,
                            {
                                "text": response.text,
                                "tool_calls": [],
                                "finish_reason": response.finish_reason.value,
                                "prompt_tokens": response.prompt_tokens,
                                "completion_tokens": response.completion_tokens,
                                "logprobs": None,
                                "ttft_ms": response.ttft_ms,
                                "total_ms": response.total_ms,
                                "raw": response.raw,
                            },
                        )
                    break
                except TimeoutError:
                    harness_timeout = True
                    attempt_log.append(
                        {
                            "attempt": attempt + 1,
                            "error_class": ErrorClass.RETRYABLE_TRANSIENT.value,
                            "duration_ms": None,
                            "at": start.isoformat(),
                        }
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
                            "message": str(exc),
                        }
                    )
                    if error_class not in (
                        ErrorClass.RETRYABLE_TRANSIENT,
                        ErrorClass.RETRYABLE_RATE_LIMIT,
                    ):
                        harness_error = True
                        break
                    if attempt < config.max_retries:
                        await _retry_delay(
                            attempt,
                            self.settings.default_retry_base_s,
                            self.settings.default_retry_cap_s,
                        )
                    else:
                        harness_error = True

        outcome = classify_outcome(
            output=response.text if response else None,
            finish_reason=response.finish_reason if response else None,
            harness_error=harness_error,
            harness_timeout=harness_timeout,
        )

        raw_uri = None
        if response and response.raw:
            key = blob_key_for_raw(str(run_id), item.case.external_id, item.repeat_idx)
            raw_uri = await self.blob_store.put_json(key, response.raw)

        async with session_scope() as session:
            repo = RunRepository(session)
            gen_id = await repo.save_generation(
                run_id=run_id,
                case_id=item.case_db_id,
                repeat_idx=item.repeat_idx,
                output=response.text if response else None,
                tool_calls=[],
                finish_reason=response.finish_reason if response else None,
                outcome=outcome,
                prompt_tokens=response.prompt_tokens if response else None,
                completion_tokens=response.completion_tokens if response else None,
                cost_usd=0.0,
                ttft_ms=response.ttft_ms if response else None,
                total_ms=response.total_ms if response else None,
                queue_wait_ms=None,
                attempts=len(attempt_log) or 1,
                attempt_log=attempt_log,
                cached=cached,
                raw_uri=raw_uri,
                trace_id=trace_id,
            )

            if response and outcome not in (
                FailureOutcome.HARNESS_ERROR,
                FailureOutcome.HARNESS_TIMEOUT,
            ):
                gen_row = await session.get(GenerationRow, gen_id)
                if gen_row is None:
                    return
                gen = await repo.generation_to_domain(gen_row, item.case.external_id)
                normalizer = Normalizer(NormalizerConfig())
                metric = ExactMatchMetric(normalizer)
                ctx = ScoringContext(normalizer_id=normalizer.config_id)
                scores = metric.score(gen, item.case, ctx)
                for score in scores:
                    if outcome == FailureOutcome.PASSED and score.passed is False:
                        await session.execute(
                            update(GenerationRow)
                            .where(GenerationRow.id == gen_id)
                            .values(outcome=FailureOutcome.FAILED_SCORE.value)
                        )
                    await repo.save_score(
                        generation_id=gen_id,
                        metric_name=score.metric_name,
                        metric_version=score.metric_version,
                        metric_config_sha256=score.metric_config_sha256,
                        value=score.value,
                        passed=score.passed,
                        detail=score.detail,
                    )
