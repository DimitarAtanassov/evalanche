"""Evaluation use case: dataset and provider in, persisted run and report out."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalharness.app.settings import Settings
from evalharness.datasets import dataset_upsert_fields, load_dataset, validate_dataset
from evalharness.domain.ports import RunStoreFactory
from evalharness.execution.executor import Executor
from evalharness.hashing import config_hash, sha256_hex
from evalharness.observability import (
    ProgressCallback,
    StageTimer,
    exception_summary,
    get_logger,
    setup_logging,
    setup_otel,
)
from evalharness.providers.factory import ProviderBuilder
from evalharness.reporting.report import PRIMARY_METRIC, RunReport, write_report
from evalharness.scoring.engine import ScoringEngineFactory
from evalharness.db.session import init_db, session_scope

logger = get_logger(__name__)

type RunStartedCallback = Callable[[uuid.UUID, bool], None]
"""Notified with the run id and whether it was resumed, before execution begins."""


class DatasetValidationError(Exception):
    """The dataset is not fit to run; ``errors`` holds every validation failure."""

    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__(f"Dataset validation failed with {len(errors)} error(s)")
        self.errors: tuple[str, ...] = tuple(errors)


class ResumeError(Exception):
    """The requested run cannot be resumed with the supplied inputs."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """Identity of the executed run and the report written for it."""

    run_id: uuid.UUID
    report: RunReport


class EvaluationService:
    """Orchestrates validate → execute → score → report for one evaluation run."""

    def __init__(
        self,
        *,
        settings: Settings,
        build_provider: ProviderBuilder,
        scoring_engine: ScoringEngineFactory,
        run_store: RunStoreFactory,
    ) -> None:
        self._settings = settings
        self._build_provider = build_provider
        self._scoring_engine = scoring_engine
        self._run_store = run_store

    async def run(
        self,
        *,
        dataset_dir: Path,
        template: Path,
        model: str,
        provider: str,
        output_dir: Path,
        repeats: int,
        concurrency: int,
        temperature: float,
        max_tokens: int | None,
        seed: int | None,
        resume: str | None,
        final_eval: bool,
        coverage_floor: float,
        tenant_id: str,
        progress: ProgressCallback | None = None,
        on_run_started: RunStartedCallback | None = None,
    ) -> RunResult:
        """Validate, execute, score, and report one evaluation run.

        Raises ``DatasetValidationError`` when the dataset is unfit and ``ResumeError`` when
        ``resume`` names an unknown run or one built from different inputs. Publishability is
        reported, not enforced: the caller decides what an unpublishable report means.
        """
        setup_logging()
        setup_otel()
        pipeline_timer = StageTimer()
        await init_db()

        logger.info("dataset_validation_started", dataset_path=str(dataset_dir))
        bundle = load_dataset(dataset_dir)
        validation = validate_dataset(bundle, allow_holdout=final_eval)
        logger.info(
            "dataset_validated",
            dataset=bundle.manifest.name,
            version=bundle.manifest.version,
            split=bundle.manifest.split,
            cases=len(bundle.cases),
            content_sha256=bundle.content_sha256,
            valid=validation.valid,
            warnings=len(validation.warnings),
            errors=len(validation.errors),
        )
        if not validation.valid:
            raise DatasetValidationError(validation.errors)

        template_body = template.read_text(encoding="utf-8")
        template_sha = sha256_hex(template_body)
        decode_params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "top_p": None,
            "top_k": None,
            "stop": [],
        }

        prov = self._build_provider(provider, concurrency=concurrency)
        logger.info("provider_resolution_started", provider=provider, model=model)
        model_version = await prov.resolve_version(model)
        logger.info(
            "provider_resolved",
            provider=model_version.provider,
            model=model_version.model,
            model_digest=model_version.resolved_version,
            capabilities=dict(model_version.capabilities or {}),
        )
        resumed_run_id = uuid.UUID(resume) if resume else None

        logger.info(
            "pipeline_planning_started",
            resume=bool(resume),
            repeats=repeats,
            concurrency=concurrency,
        )
        async with session_scope() as session:
            repo = self._run_store(session)
            dataset_id = await repo.upsert_dataset(**dataset_upsert_fields(bundle))
            prompt_template_id = await repo.upsert_prompt_template(
                name=f"{bundle.manifest.name}-{template.stem}",
                version=bundle.manifest.version,
                body=template_body,
                sha256=template_sha,
            )
            model_version_id = await repo.upsert_model_version(
                provider=model_version.provider,
                model=model_version.model,
                resolved_version=model_version.resolved_version,
                quantization=model_version.quantization,
                capabilities=dict(model_version.capabilities or {}),
            )
            if resumed_run_id is not None:
                stored_run = await repo.get_run(resumed_run_id)
                if stored_run is None:
                    raise ResumeError(f"Run not found: {resumed_run_id}")
                supplied_config_sha = config_hash(
                    dataset_sha256=bundle.content_sha256,
                    prompt_template_sha256=template_sha,
                    provider=model_version.provider,
                    model=model_version.model,
                    resolved_version=model_version.resolved_version,
                    decode_params=decode_params,
                    harness_version=self._settings.harness_version,
                )
                mismatches = []
                if stored_run.dataset_id != dataset_id:
                    mismatches.append("dataset")
                if stored_run.prompt_template_id != prompt_template_id:
                    mismatches.append("prompt template")
                if stored_run.model_version_id != model_version_id:
                    mismatches.append("model version")
                if stored_run.config_sha256 != supplied_config_sha:
                    mismatches.append("configuration hash")
                if mismatches:
                    raise ResumeError(
                        "Resume inputs do not match stored run: " + ", ".join(mismatches)
                    )

        logger.info(
            "pipeline_planning_finished",
            dataset_id=dataset_id,
            prompt_template_id=prompt_template_id,
            model_version_id=model_version_id,
            resume=bool(resume),
        )
        executor = Executor(
            provider=prov,
            model=model,
            model_version=model_version,
            template_body=template_body,
            settings=self._settings,
            run_store=self._run_store,
        )

        if resume:
            assert resumed_run_id is not None
            run_id = resumed_run_id
            await executor.validate_resume(
                run_id,
                dataset_id=dataset_id,
                prompt_template_id=prompt_template_id,
                model_version_id=model_version_id,
                decode_params=decode_params,
                repeats=repeats,
                tenant_id=tenant_id,
            )
        else:
            run_id = await executor.create_run(
                bundle_dataset_id=dataset_id,
                prompt_template_id=prompt_template_id,
                model_version_id=model_version_id,
                dataset_sha256=bundle.content_sha256,
                prompt_template_sha256=template_sha,
                decode_params=decode_params,
                repeats=repeats,
                tenant_id=tenant_id,
            )
        if on_run_started is not None:
            on_run_started(run_id, bool(resume))

        # The headline pass rate must be computed from a metric this run actually scored.
        metric_names = list(bundle.manifest.task_metrics or [PRIMARY_METRIC])
        try:
            await executor.execute_run(run_id, concurrency=concurrency, progress=progress)
            await self._scoring_engine().rescore_run(
                run_id,
                metric_names,
                progress=progress,
            )
            report = await write_report(
                run_id,
                output_dir,
                coverage_floor=coverage_floor,
                progress=progress,
                primary_metric=metric_names[0],
                run_store=self._run_store,
            )
        except Exception as exc:
            logger.exception("pipeline_failed", **exception_summary(exc))
            raise
        finally:
            await prov.aclose()
        logger.info(
            "pipeline_finished",
            run_id=str(run_id),
            publishable=report.publishable,
            coverage=report.coverage,
            primary_metric=report.primary_metric,
            pass_rate=report.pass_rate,
            pass_rate_n=report.pass_rate_n,
            duration_ms=pipeline_timer.elapsed_ms,
        )
        return RunResult(run_id=run_id, report=report)
