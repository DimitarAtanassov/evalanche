"""``cases`` table access."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import CaseRow
from evalharness.domain.dataset import Case
from evalharness.repositories.mappers import _case


class CaseRepository:
    """Reads over the cases of a dataset. Writes belong to ``DatasetRepository``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_cases_for_dataset(self, dataset_id: int) -> list[tuple[int, Case]]:
        """Cases paired with their row id, ordered by id so runs stay reproducible."""
        stmt = select(CaseRow).where(CaseRow.dataset_id == dataset_id).order_by(CaseRow.id)
        rows = (await self._session.execute(stmt)).scalars().all()
        return [(row.id, _case(row)) for row in rows]

    async def count_for_dataset(self, dataset_id: int) -> int:
        stmt = select(func.count(CaseRow.id)).where(CaseRow.dataset_id == dataset_id)
        return int((await self._session.execute(stmt)).scalar_one())

    async def get_case_external_id(self, case_id: int) -> str:
        row = await self._session.get(CaseRow, case_id)
        if not row:
            raise KeyError(case_id)
        return row.external_id
