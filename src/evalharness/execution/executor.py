"""Run lifecycle orchestration: create, plan, resume, and drive the case workers."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from evalharness.app.settings import Settings, get_settings
from evalharness.domain.enums import FailureOutcome
from evalharness.domain.generation import ModelVersion
from evalharness.domain.ports import RunStoreFactory
from evalharness.domain.provider import Provider
from evalharness.execution.cache import response_cache_key as response_cache_key
from evalharness.execution.case_runner import CaseRunner
from evalharness.execution.errors import ResumeError as ResumeError
from evalharness.execution.helpers import classify_outcome as classify_outcome
from evalharness.execution.helpers import render_prompt as render_prompt
from evalharness.execution.helpers import validate_decode_params
from evalharness.execution.models import ExecutionResult as ExecutionResult
from evalharness.execution.models import RunConfig as RunConfig
from evalharness.execution.models import RunPlanItem as RunPlanItem
from evalharness.execution.plan import build_run_plan
from evalharness.execution.shutdown import GracefulShutdown as GracefulShutdown
from evalharness.hashing import config_hash
from evalharness.observability import (
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    StageTimer,
    emit_progress,
    get_logger,
    get_tracer,
    log_context,
)
from evalharness.db.session import session_scope
from evalharness.repositories import RunStoreUow

logger = get_logger(__name__)


@dataclass
class _ProgressTally:
    """Running counts a generation pass reports to logs and the progress callback."""

    total: int
    completed: int = 0
    passed: int = 0
    retries: int = 0
    cache_hits: int = 0

    def record(self, result: ExecutionResult) -> None:
        self.completed += 1
        self.passed += int(result.outcome == FailureOutcome.PASSED)
        self.retries += max(0, result.attempts - 1)
        self.cache_hits += int(result.cached)

    @property
    def other_outcomes(self) -> int:
        return self.completed - self.passed

    def as_fields(self) -> dict[str, int | float | str]:
        return {
            "valid_outputs": self.passed,
            "other_outcomes": self.other_outcomes,
            "retries": self.retries,
            "cache_hits": self.cache_hits,
        }


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
        self.run_store: RunStoreFactory = run_store or RunStoreUow
        self.tracer = get_tracer()
        self.shutdown = GracefulShutdown()
        self.case_runner = CaseRunner(
            provider=provider,
            model=model,
            model_version=model_version,
            template_body=template_body,
            settings=self.settings,
            run_store=self.run_store,
            tracer=self.tracer,
            shutdown=self.shutdown,
        )

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
        return await build_run_plan(run_id, run_store=self.run_store, settings=self.settings)

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
            worker_failures = await self._drive_workers(
                run_id, config, items, worker_count, progress
            )

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

    async def _drive_workers(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        items: list[RunPlanItem],
        worker_count: int,
        progress: ProgressCallback | None,
    ) -> bool:
        """Run the pool under the run deadline and shutdown signal; report worker failures.

        Shutdown gets a bounded drain so in-flight cases can persist; the run deadline
        does not, because the budget it enforces has already been spent.
        """
        pipeline = asyncio.create_task(
            self._run_worker_pool(run_id, config, items, worker_count, progress),
            name=f"run-{run_id}-pipeline",
        )
        shutdown_wait = asyncio.create_task(self.shutdown.event.wait(), name="shutdown-wait")
        try:
            done, _ = await asyncio.wait(
                {pipeline, shutdown_wait},
                timeout=config.run_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pipeline in done:
                return _has_failures(await pipeline)
            if shutdown_wait in done:
                try:
                    results = await asyncio.wait_for(pipeline, timeout=config.drain_timeout_s)
                    return _has_failures(results)
                except TimeoutError:
                    await _cancel(pipeline)
                    return False
            self.shutdown.request("run_deadline")
            await _cancel(pipeline)
            return False
        finally:
            await _cancel(shutdown_wait)

    async def _run_worker_pool(
        self,
        run_id: uuid.UUID,
        config: RunConfig,
        items: list[RunPlanItem],
        worker_count: int,
        progress: ProgressCallback | None,
    ) -> list[Any]:
        queue: asyncio.Queue[RunPlanItem | None] = asyncio.Queue(maxsize=worker_count * 2)
        tally = _ProgressTally(total=len(items))

        async def produce() -> None:
            for item in items:
                if self.shutdown.requested:
                    break
                await queue.put(item)
            for _ in range(worker_count):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    result = await self.case_runner.run_one(run_id, config, item)
                    # A shutdown short-circuit wrote nothing, so it is not progress.
                    if not result.persisted:
                        continue
                    tally.record(result)
                    if (
                        tally.completed == tally.total
                        or tally.completed % self.settings.log_progress_every == 0
                    ):
                        logger.info(
                            "generation_progress",
                            completed=tally.completed,
                            total=tally.total,
                            **tally.as_fields(),
                        )
                    emit_progress(
                        progress,
                        ProgressEvent(
                            PipelineStage.GENERATING,
                            tally.completed,
                            tally.total,
                            result.external_id,
                            tally.as_fields(),
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


def _has_failures(results: list[Any]) -> bool:
    return any(isinstance(result, BaseException) for result in results)


async def _cancel(task: asyncio.Task[Any]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
