"""``scores`` table access."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import CaseRow, GenerationRow, ScoreRow
from evalharness.domain.scoring import StoredScore
from evalharness.repositories.mappers import _stored_score


class ScoreRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        """Persist one score idempotently.

        Generation, metric, metric version, and metric config are the identity, so
        re-scoring a run is a no-op rather than a second row that would double-count.
        """
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
        await self._session.execute(stmt)

    async def get_scores_for_run(self, run_id: uuid.UUID) -> list[StoredScore]:
        stmt = (
            select(ScoreRow)
            .join(GenerationRow, ScoreRow.generation_id == GenerationRow.id)
            .where(GenerationRow.run_id == run_id)
            .order_by(ScoreRow.generation_id, ScoreRow.metric_name, ScoreRow.metric_version)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_stored_score(row) for row in rows]

    async def get_paired_outcomes(
        self, run_id: uuid.UUID, metric: str
    ) -> dict[tuple[str, int], bool]:
        """Pass/fail keyed by (case external id, repeat) for a paired comparison.

        Keyed on the external id, not the row id, so two runs over different
        materializations of the same corpus still align. Scores with no verdict are
        excluded rather than read as failures.
        """
        statement = (
            select(CaseRow.external_id, GenerationRow.repeat_idx, ScoreRow.passed)
            .join(GenerationRow, ScoreRow.generation_id == GenerationRow.id)
            .join(CaseRow, GenerationRow.case_id == CaseRow.id)
            .where(
                GenerationRow.run_id == run_id,
                ScoreRow.metric_name == metric,
                ScoreRow.passed.is_not(None),
            )
        )
        return {
            (case_id, repeat): bool(passed)
            for case_id, repeat, passed in (await self._session.execute(statement)).all()
        }
