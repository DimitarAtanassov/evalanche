"""``generations`` table access."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import CaseRow, GenerationRow
from evalharness.domain.enums import FailureOutcome, FinishReason
from evalharness.domain.generation import StoredGeneration
from evalharness.repositories.mappers import _stored_generation


class GenerationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_completed_keys(self, run_id: uuid.UUID) -> set[tuple[int, int]]:
        """The (case, repeat) pairs already persisted, so a resume skips them."""
        stmt = select(GenerationRow.case_id, GenerationRow.repeat_idx).where(
            GenerationRow.run_id == run_id
        )
        rows = (await self._session.execute(stmt)).all()
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
        """Persist one generation idempotently and return its id.

        (run, case, repeat) is the identity, so a retried write loses the race rather
        than duplicating the attempt, and the existing id is read back instead.
        """
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
        generation_id = (await self._session.execute(stmt)).scalar_one_or_none()
        if generation_id is not None:
            return generation_id
        existing_stmt = select(GenerationRow.id).where(
            GenerationRow.run_id == run_id,
            GenerationRow.case_id == case_id,
            GenerationRow.repeat_idx == repeat_idx,
        )
        return (await self._session.execute(existing_stmt)).scalar_one()

    async def get_generations_for_run(self, run_id: uuid.UUID) -> list[StoredGeneration]:
        """Generations with their case's external id, ordered for stable reports."""
        stmt = (
            select(GenerationRow, CaseRow.external_id)
            .join(CaseRow, GenerationRow.case_id == CaseRow.id)
            .where(GenerationRow.run_id == run_id)
            .order_by(GenerationRow.case_id, GenerationRow.repeat_idx)
        )
        rows = (await self._session.execute(stmt)).all()
        return [_stored_generation(row, external_id) for row, external_id in rows]
