"""Database repository for runs, generations, and scores."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.core.enums import FailureOutcome, FinishReason, TaskType
from evalharness.core.models import Case, Generation
from evalharness.datasets.loader import DatasetBundle
from evalharness.store.models import (
    CaseRow,
    DatasetRow,
    GenerationRow,
    MetricAggregateRow,
    ModelVersionRow,
    PromptTemplateRow,
    ResponseCacheRow,
    RunRow,
    ScoreRow,
)


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_dataset(self, bundle: DatasetBundle) -> int:
        stmt = select(DatasetRow).where(
            DatasetRow.name == bundle.manifest.name,
            DatasetRow.version == bundle.manifest.version,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing.id

        row = DatasetRow(
            name=bundle.manifest.name,
            version=bundle.manifest.version,
            content_sha256=bundle.content_sha256,
            split=bundle.manifest.split,
            manifest={
                "name": bundle.manifest.name,
                "version": bundle.manifest.version,
                "split": bundle.manifest.split,
                "license": bundle.manifest.license,
                "pii_scrubbed": bundle.manifest.pii_scrubbed,
                "created_at": bundle.manifest.created_at,
                "slices": bundle.manifest.slices,
            },
        )
        self.session.add(row)
        await self.session.flush()

        for case in bundle.cases:
            self.session.add(
                CaseRow(
                    dataset_id=row.id,
                    external_id=case.external_id,
                    task_type=case.task_type.value,
                    inputs=case.inputs,
                    reference={
                        "reference_answer": case.reference_answer,
                        "references": case.references,
                        "expected_label": case.expected_label,
                        "expected_json": case.expected_json,
                        "must_contain": case.must_contain,
                        "must_not_contain": case.must_not_contain,
                    },
                    qrels=case.qrels,
                    slices=case.slices,
                    weight=case.weight,
                )
            )
        await self.session.flush()
        return row.id

    async def upsert_prompt_template(self, name: str, version: str, body: str, sha256: str) -> int:
        stmt = select(PromptTemplateRow).where(
            PromptTemplateRow.name == name,
            PromptTemplateRow.version == version,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing.id
        row = PromptTemplateRow(name=name, version=version, body=body, content_sha256=sha256)
        self.session.add(row)
        await self.session.flush()
        return row.id

    async def upsert_model_version(
        self,
        provider: str,
        model: str,
        resolved_version: str,
        quantization: str | None,
        capabilities: dict[str, Any],
    ) -> int:
        stmt = select(ModelVersionRow).where(
            ModelVersionRow.provider == provider,
            ModelVersionRow.model == model,
            ModelVersionRow.resolved_version == resolved_version,
            ModelVersionRow.quantization == quantization,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing.id
        row = ModelVersionRow(
            provider=provider,
            model=model,
            resolved_version=resolved_version,
            quantization=quantization,
            capabilities=capabilities,
        )
        self.session.add(row)
        await self.session.flush()
        return row.id

    async def create_run(
        self,
        *,
        dataset_id: int,
        prompt_template_id: int,
        model_version_id: int,
        decode_params: dict[str, Any],
        config_sha256: str,
        harness_version: str,
        git_sha: str,
        repeats: int,
        tenant_id: str,
        run_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        run = RunRow(
            id=run_id or uuid.uuid4(),
            dataset_id=dataset_id,
            prompt_template_id=prompt_template_id,
            model_version_id=model_version_id,
            decode_params=decode_params,
            config_sha256=config_sha256,
            harness_version=harness_version,
            git_sha=git_sha,
            repeats=repeats,
            status="queued",
            tenant_id=tenant_id,
            started_at=datetime.now(UTC),
        )
        self.session.add(run)
        await self.session.flush()
        return run.id

    async def update_run_status(self, run_id: uuid.UUID, status: str) -> None:
        run = await self.session.get(RunRow, run_id)
        if run:
            run.status = status
            if status in ("completed", "failed", "cancelled"):
                run.finished_at = datetime.now(UTC)

    async def get_cases_for_dataset(self, dataset_id: int) -> list[tuple[int, Case]]:
        stmt = select(CaseRow).where(CaseRow.dataset_id == dataset_id)
        rows = (await self.session.execute(stmt)).scalars().all()
        result: list[tuple[int, Case]] = []
        for row in rows:
            ref = row.reference or {}
            result.append(
                (
                    row.id,
                    Case(
                        external_id=row.external_id,
                        task_type=TaskType(row.task_type),
                        inputs=row.inputs,
                        reference_answer=ref.get("reference_answer"),
                        references=ref.get("references", []),
                        expected_label=ref.get("expected_label"),
                        expected_json=ref.get("expected_json"),
                        qrels=row.qrels,
                        slices=row.slices or {},
                        must_contain=ref.get("must_contain", []),
                        must_not_contain=ref.get("must_not_contain", []),
                    ),
                )
            )
        return result

    async def get_completed_keys(self, run_id: uuid.UUID) -> set[tuple[int, int]]:
        stmt = select(GenerationRow.case_id, GenerationRow.repeat_idx).where(
            GenerationRow.run_id == run_id
        )
        rows = (await self.session.execute(stmt)).all()
        return {(case_id, repeat_idx) for case_id, repeat_idx in rows}

    async def save_generation(
        self,
        *,
        run_id: uuid.UUID,
        case_id: int,
        repeat_idx: int,
        output: str | None,
        tool_calls: list[dict[str, Any]] | None,
        finish_reason: FinishReason | None,
        outcome: FailureOutcome,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cost_usd: float | None,
        ttft_ms: float | None,
        total_ms: float | None,
        queue_wait_ms: float | None,
        attempts: int,
        attempt_log: list[dict[str, Any]],
        cached: bool,
        raw_response: dict[str, Any] | None,
        trace_id: str | None,
    ) -> int:
        row = GenerationRow(
            run_id=run_id,
            case_id=case_id,
            repeat_idx=repeat_idx,
            output=output,
            tool_calls=tool_calls,
            finish_reason=finish_reason.value if finish_reason else None,
            outcome=outcome.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            queue_wait_ms=queue_wait_ms,
            attempts=attempts,
            attempt_log=attempt_log,
            cached=cached,
            raw_response=raw_response,
            trace_id=trace_id,
        )
        self.session.add(row)
        await self.session.flush()
        return row.id

    async def save_score(
        self,
        *,
        generation_id: int,
        metric_name: str,
        metric_version: str,
        metric_config_sha256: str,
        value: float | None,
        passed: bool | None,
        detail: dict[str, Any] | None,
    ) -> None:
        row = ScoreRow(
            generation_id=generation_id,
            metric_name=metric_name,
            metric_version=metric_version,
            metric_config_sha256=metric_config_sha256,
            value=value,
            passed=passed,
            detail=detail,
        )
        self.session.add(row)

    async def save_metric_aggregate(
        self,
        *,
        run_id: uuid.UUID,
        metric_name: str,
        metric_version: str,
        slice_key: str,
        n: int,
        value: float,
        ci_low: float | None,
        ci_high: float | None,
        stddev: float | None,
        method: str,
    ) -> None:
        self.session.add(
            MetricAggregateRow(
                run_id=run_id,
                metric_name=metric_name,
                metric_version=metric_version,
                slice_key=slice_key,
                n=n,
                value=value,
                ci_low=ci_low,
                ci_high=ci_high,
                stddev=stddev,
                method=method,
            )
        )

    async def get_generations_for_run(self, run_id: uuid.UUID) -> list[GenerationRow]:
        stmt = select(GenerationRow).where(GenerationRow.run_id == run_id)
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_run(self, run_id: uuid.UUID) -> RunRow | None:
        return await self.session.get(RunRow, run_id)

    async def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = await self.session.get(ResponseCacheRow, cache_key)
        return row.response if row else None

    async def put_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        existing = await self.session.get(ResponseCacheRow, cache_key)
        if existing:
            return
        self.session.add(ResponseCacheRow(cache_key=cache_key, response=response))

    async def get_case_external_id(self, case_id: int) -> str:
        row = await self.session.get(CaseRow, case_id)
        if not row:
            raise KeyError(case_id)
        return row.external_id

    async def get_scores_for_run(self, run_id: uuid.UUID) -> list[ScoreRow]:
        stmt = (
            select(ScoreRow)
            .join(GenerationRow, ScoreRow.generation_id == GenerationRow.id)
            .where(GenerationRow.run_id == run_id)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def generation_to_domain(self, row: GenerationRow, case_external_id: str) -> Generation:
        return Generation(
            id=row.id,
            run_id=str(row.run_id),
            case_external_id=case_external_id,
            repeat_idx=row.repeat_idx,
            output=row.output,
            tool_calls=list(row.tool_calls) if isinstance(row.tool_calls, list) else [],
            finish_reason=FinishReason(row.finish_reason) if row.finish_reason else None,
            outcome=FailureOutcome(row.outcome),
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            cost_usd=float(row.cost_usd) if row.cost_usd is not None else None,
            ttft_ms=row.ttft_ms,
            total_ms=row.total_ms,
            queue_wait_ms=row.queue_wait_ms,
            attempts=row.attempts,
            attempt_log=list(row.attempt_log) if isinstance(row.attempt_log, list) else [],
            cached=row.cached,
            raw_response=row.raw_response,
            trace_id=row.trace_id,
        )
