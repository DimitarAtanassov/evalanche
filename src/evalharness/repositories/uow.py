"""Session-scoped unit of work: the one implementation of the ``RunStore`` port."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.domain.dataset import Case, DatasetRef
from evalharness.domain.enums import FailureOutcome, FinishReason
from evalharness.domain.generation import StoredGeneration
from evalharness.domain.run import ModelVersionRef, PromptTemplateRef, RunRecord
from evalharness.domain.scoring import StoredAggregate, StoredScore
from evalharness.repositories.cases import CaseRepository
from evalharness.repositories.datasets import DatasetRepository
from evalharness.repositories.generations import GenerationRepository
from evalharness.repositories.metric_aggregates import MetricAggregateRepository
from evalharness.repositories.model_versions import ModelVersionRepository
from evalharness.repositories.prompt_templates import PromptTemplateRepository
from evalharness.repositories.response_cache import ResponseCacheRepository
from evalharness.repositories.runs import RunRepository
from evalharness.repositories.scores import ScoreRepository


class RunStoreUow:
    """Every persistence call the application layer makes, bound to one session.

    A facade over the per-table repositories: it owns no SQL and no transaction. The
    session's lifecycle, and therefore the commit boundary, stays with the caller that
    opened it, so a run's writes land or roll back together.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._cases = CaseRepository(session)
        self._datasets = DatasetRepository(session)
        self._prompt_templates = PromptTemplateRepository(session)
        self._model_versions = ModelVersionRepository(session)
        self._runs = RunRepository(session, self._cases)
        self._generations = GenerationRepository(session)
        self._scores = ScoreRepository(session)
        self._metric_aggregates = MetricAggregateRepository(session)
        self._response_cache = ResponseCacheRepository(session)

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
        return await self._datasets.upsert_dataset(
            name=name,
            version=version,
            split=split,
            content_sha256=content_sha256,
            license=license,
            pii_scrubbed=pii_scrubbed,
            created_at=created_at,
            slices=slices,
            cases=cases,
        )

    async def get_dataset(self, dataset_id: int) -> DatasetRef | None:
        return await self._datasets.get_dataset(dataset_id)

    async def get_cases_for_dataset(self, dataset_id: int) -> list[tuple[int, Case]]:
        return await self._cases.get_cases_for_dataset(dataset_id)

    async def get_case_external_id(self, case_id: int) -> str:
        return await self._cases.get_case_external_id(case_id)

    async def upsert_prompt_template(self, name: str, version: str, body: str, sha256: str) -> int:
        return await self._prompt_templates.upsert_prompt_template(name, version, body, sha256)

    async def get_prompt_template(self, prompt_template_id: int) -> PromptTemplateRef | None:
        return await self._prompt_templates.get_prompt_template(prompt_template_id)

    async def upsert_model_version(
        self,
        provider: str,
        model: str,
        resolved_version: str,
        quantization: str | None,
        capabilities: dict[str, Any],
    ) -> int:
        return await self._model_versions.upsert_model_version(
            provider, model, resolved_version, quantization, capabilities
        )

    async def get_model_version(self, model_version_id: int) -> ModelVersionRef | None:
        return await self._model_versions.get_model_version(model_version_id)

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
        return await self._runs.create_run(
            dataset_id=dataset_id,
            prompt_template_id=prompt_template_id,
            model_version_id=model_version_id,
            decode_params=decode_params,
            config_sha256=config_sha256,
            harness_version=harness_version,
            git_sha=git_sha,
            repeats=repeats,
            tenant_id=tenant_id,
            run_id=run_id,
        )

    async def update_run_status(self, run_id: uuid.UUID, status: str) -> None:
        await self._runs.update_run_status(run_id, status)

    async def get_run(self, run_id: uuid.UUID) -> RunRecord | None:
        return await self._runs.get_run(run_id)

    async def get_planned_generation_count(self, run_id: uuid.UUID) -> int:
        return await self._runs.get_planned_generation_count(run_id)

    async def get_completed_keys(self, run_id: uuid.UUID) -> set[tuple[int, int]]:
        return await self._generations.get_completed_keys(run_id)

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
        return await self._generations.save_generation(
            run_id=run_id,
            case_id=case_id,
            repeat_idx=repeat_idx,
            output=output,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            outcome=outcome,
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

    async def get_generations_for_run(self, run_id: uuid.UUID) -> list[StoredGeneration]:
        return await self._generations.get_generations_for_run(run_id)

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
        await self._scores.save_score(
            generation_id=generation_id,
            metric_name=metric_name,
            metric_version=metric_version,
            metric_config_sha256=metric_config_sha256,
            value=value,
            passed=passed,
            detail=detail,
        )

    async def get_scores_for_run(self, run_id: uuid.UUID) -> list[StoredScore]:
        return await self._scores.get_scores_for_run(run_id)

    async def get_paired_outcomes(
        self, run_id: uuid.UUID, metric: str
    ) -> dict[tuple[str, int], bool]:
        return await self._scores.get_paired_outcomes(run_id, metric)

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
        await self._metric_aggregates.save_metric_aggregate(
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

    async def get_metric_aggregates(self, run_id: uuid.UUID) -> list[StoredAggregate]:
        return await self._metric_aggregates.get_metric_aggregates(run_id)

    async def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        return await self._response_cache.get_cache(cache_key)

    async def put_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        await self._response_cache.put_cache(cache_key, response)

    async def delete_cache(self, cache_keys: Sequence[str]) -> None:
        """Not part of ``RunStore``: only cold-start tooling invalidates the cache."""
        await self._response_cache.delete_cache(cache_keys)
