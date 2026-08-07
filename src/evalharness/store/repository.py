"""Database repository for runs, generations, and scores."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.core.enums import FailureOutcome, FinishReason, TaskType
from evalharness.core.models import Case, Generation
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

    async def upsert_dataset(
        self,
        *,
        name: str,
        version: str,
        split: str,
        content_sha256: str,
        license: str,
        pii_scrubbed: bool,
        created_at: str,
        slices: Sequence[str],
        cases: Sequence[Case],
    ) -> int:
        stmt = select(DatasetRow).where(
            DatasetRow.name == name,
            DatasetRow.version == version,
        )
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing:
            if existing.content_sha256 != content_sha256:
                raise ValueError(f"Dataset {name}@{version} already exists with different content")
            return existing.id

        row = DatasetRow(
            name=name,
            version=version,
            content_sha256=content_sha256,
            split=split,
            manifest={
                "name": name,
                "version": version,
                "split": split,
                "license": license,
                "pii_scrubbed": pii_scrubbed,
                "created_at": created_at,
                "slices": list(slices),
            },
        )
        self.session.add(row)
        await self.session.flush()

        for case in cases:
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
            if existing.content_sha256 != sha256:
                raise ValueError(
                    f"Prompt template {name}@{version} already exists with different content"
                )
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
        stmt = select(CaseRow).where(CaseRow.dataset_id == dataset_id).order_by(CaseRow.id)
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
        values = {
            "run_id": run_id,
            "case_id": case_id,
            "repeat_idx": repeat_idx,
            "output": output,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason.value if finish_reason else None,
            "outcome": outcome.value,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "queue_wait_ms": queue_wait_ms,
            "attempts": attempts,
            "attempt_log": attempt_log,
            "cached": cached,
            "raw_response": raw_response,
            "trace_id": trace_id,
        }
        stmt = (
            insert(GenerationRow)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["run_id", "case_id", "repeat_idx"])
            .returning(GenerationRow.id)
        )
        generation_id = (await self.session.execute(stmt)).scalar_one_or_none()
        if generation_id is not None:
            return generation_id
        existing_stmt = select(GenerationRow.id).where(
            GenerationRow.run_id == run_id,
            GenerationRow.case_id == case_id,
            GenerationRow.repeat_idx == repeat_idx,
        )
        return (await self.session.execute(existing_stmt)).scalar_one()

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
        stmt = (
            insert(ScoreRow)
            .values(
                generation_id=generation_id,
                metric_name=metric_name,
                metric_version=metric_version,
                metric_config_sha256=metric_config_sha256,
                value=value,
                passed=passed,
                detail=detail,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    "generation_id",
                    "metric_name",
                    "metric_version",
                    "metric_config_sha256",
                ]
            )
        )
        await self.session.execute(stmt)

    async def save_metric_aggregate(
        self,
        *,
        run_id: uuid.UUID,
        metric_name: str,
        metric_version: str,
        metric_config_sha256: str,
        slice_key: str,
        n: int,
        value: float,
        ci_low: float | None,
        ci_high: float | None,
        stddev: float | None,
        method: str,
    ) -> None:
        stmt = insert(MetricAggregateRow).values(
            run_id=run_id,
            metric_name=metric_name,
            metric_version=metric_version,
            metric_config_sha256=metric_config_sha256,
            slice_key=slice_key,
            n=n,
            value=value,
            ci_low=ci_low,
            ci_high=ci_high,
            stddev=stddev,
            method=method,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_metric_aggregates_identity",
            set_={
                "n": stmt.excluded.n,
                "value": stmt.excluded.value,
                "ci_low": stmt.excluded.ci_low,
                "ci_high": stmt.excluded.ci_high,
                "stddev": stmt.excluded.stddev,
                "method": stmt.excluded.method,
            },
        )
        await self.session.execute(stmt)

    async def get_generations_for_run(self, run_id: uuid.UUID) -> list[GenerationRow]:
        stmt = (
            select(GenerationRow)
            .where(GenerationRow.run_id == run_id)
            .order_by(GenerationRow.case_id, GenerationRow.repeat_idx)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_run(self, run_id: uuid.UUID) -> RunRow | None:
        return await self.session.get(RunRow, run_id)

    async def get_dataset(self, dataset_id: int) -> DatasetRow | None:
        return await self.session.get(DatasetRow, dataset_id)

    async def get_model_version(self, model_version_id: int) -> ModelVersionRow | None:
        return await self.session.get(ModelVersionRow, model_version_id)

    async def get_prompt_template(self, prompt_template_id: int) -> PromptTemplateRow | None:
        return await self.session.get(PromptTemplateRow, prompt_template_id)

    async def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = await self.session.get(ResponseCacheRow, cache_key)
        return row.response if row else None

    async def put_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        stmt = (
            insert(ResponseCacheRow)
            .values(cache_key=cache_key, response=response)
            .on_conflict_do_nothing(index_elements=["cache_key"])
        )
        await self.session.execute(stmt)

    async def delete_cache(self, cache_keys: Sequence[str]) -> None:
        """Drop cached responses so a subsequent run executes cold."""
        if not cache_keys:
            return
        await self.session.execute(
            delete(ResponseCacheRow).where(ResponseCacheRow.cache_key.in_(list(cache_keys)))
        )

    async def get_planned_generation_count(self, run_id: uuid.UUID) -> int:
        run = await self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        case_count = (
            await self.session.execute(
                select(func.count(CaseRow.id)).where(CaseRow.dataset_id == run.dataset_id)
            )
        ).scalar_one()
        return int(case_count) * run.repeats

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
            .order_by(ScoreRow.generation_id, ScoreRow.metric_name, ScoreRow.metric_version)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_metric_aggregates(self, run_id: uuid.UUID) -> list[MetricAggregateRow]:
        stmt = (
            select(MetricAggregateRow)
            .where(MetricAggregateRow.run_id == run_id)
            .order_by(
                MetricAggregateRow.metric_name,
                MetricAggregateRow.metric_version,
                MetricAggregateRow.slice_key,
            )
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
