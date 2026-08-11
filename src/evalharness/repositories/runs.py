"""``runs`` table access."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import RunRow
from evalharness.domain.run import RunRecord
from evalharness.repositories.cases import CaseRepository
from evalharness.repositories.mappers import _run_record

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class RunRepository:
    """Run rows, plus the planned-work count derived from the run's dataset."""

    def __init__(self, session: AsyncSession, cases: CaseRepository) -> None:
        self._session = session
        self._cases = cases

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
        """Insert a queued run. A caller-supplied ``run_id`` makes the insert resumable."""
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
        self._session.add(run)
        await self._session.flush()
        return run.id

    async def update_run_status(self, run_id: uuid.UUID, status: str) -> None:
        run = await self._session.get(RunRow, run_id)
        if run:
            run.status = status
            if status in _TERMINAL_STATUSES:
                run.finished_at = datetime.now(UTC)

    async def get_run(self, run_id: uuid.UUID) -> RunRecord | None:
        row = await self._session.get(RunRow, run_id)
        return _run_record(row) if row else None

    async def get_planned_generation_count(self, run_id: uuid.UUID) -> int:
        """Generations a complete run owes: one per case per repeat.

        Coverage and resume both compare against this, so it is derived from the
        dataset's current case count rather than stored on the run.
        """
        run = await self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        return await self._cases.count_for_dataset(run.dataset_id) * run.repeats
