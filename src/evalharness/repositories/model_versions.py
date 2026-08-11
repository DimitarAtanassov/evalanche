"""``model_versions`` table access."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import ModelVersionRow
from evalharness.domain.run import ModelVersionRef
from evalharness.repositories.mappers import _model_version_ref


class ModelVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_model_version(
        self,
        provider: str,
        model: str,
        resolved_version: str,
        quantization: str | None,
        capabilities: dict[str, Any],
    ) -> int:
        """Return the id of the resolved model identity, creating it if absent.

        Capabilities are not compared: the identity is provider, model, resolved
        version, and quantization, and a re-probe of the same identity is not evidence
        that the earlier probe was wrong.
        """
        stmt = select(ModelVersionRow).where(
            ModelVersionRow.provider == provider,
            ModelVersionRow.model == model,
            ModelVersionRow.resolved_version == resolved_version,
            ModelVersionRow.quantization == quantization,
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing.id
        row = ModelVersionRow(
            provider=provider,
            model=model,
            resolved_version=resolved_version,
            quantization=quantization,
            capabilities=capabilities,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def get_model_version(self, model_version_id: int) -> ModelVersionRef | None:
        row = await self._session.get(ModelVersionRow, model_version_id)
        return _model_version_ref(row) if row else None
