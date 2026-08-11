"""Persistence port: domain types only, no ORM imports."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.domain.dataset import Case, DatasetRef
from evalharness.domain.enums import FailureOutcome, FinishReason
from evalharness.domain.generation import StoredGeneration
from evalharness.domain.run import ModelVersionRef, PromptTemplateRef, RunRecord
from evalharness.domain.scoring import StoredAggregate, StoredScore


class RunStore(Protocol):
    """Every persistence call the application layer makes against one session."""

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
    ) -> int: ...

    async def upsert_prompt_template(
        self, name: str, version: str, body: str, sha256: str
    ) -> int: ...

    async def upsert_model_version(
        self,
        provider: str,
        model: str,
        resolved_version: str,
        quantization: str | None,
        capabilities: dict[str, Any],
    ) -> int: ...

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
    ) -> uuid.UUID: ...

    async def update_run_status(self, run_id: uuid.UUID, status: str) -> None: ...

    async def get_run(self, run_id: uuid.UUID) -> RunRecord | None: ...

    async def get_dataset(self, dataset_id: int) -> DatasetRef | None: ...

    async def get_model_version(self, model_version_id: int) -> ModelVersionRef | None: ...

    async def get_prompt_template(self, prompt_template_id: int) -> PromptTemplateRef | None: ...

    async def get_cases_for_dataset(self, dataset_id: int) -> list[tuple[int, Case]]: ...

    async def get_completed_keys(self, run_id: uuid.UUID) -> set[tuple[int, int]]: ...

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
    ) -> int: ...

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
    ) -> None: ...

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
    ) -> None: ...

    async def get_generations_for_run(self, run_id: uuid.UUID) -> list[StoredGeneration]: ...

    async def get_scores_for_run(self, run_id: uuid.UUID) -> list[StoredScore]: ...

    async def get_metric_aggregates(self, run_id: uuid.UUID) -> list[StoredAggregate]: ...

    async def get_planned_generation_count(self, run_id: uuid.UUID) -> int: ...

    async def get_paired_outcomes(
        self, run_id: uuid.UUID, metric: str
    ) -> dict[tuple[str, int], bool]: ...

    async def get_cache(self, cache_key: str) -> dict[str, Any] | None: ...

    async def put_cache(self, cache_key: str, response: dict[str, Any]) -> None: ...


type RunStoreFactory = Callable[[AsyncSession], RunStore]
"""Binds a store to one session; the session lifecycle stays with the caller."""
