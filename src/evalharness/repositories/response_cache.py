"""``response_cache`` table access."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import ResponseCacheRow


class ResponseCacheRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = await self._session.get(ResponseCacheRow, cache_key)
        return row.response if row else None

    async def put_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        """Store a response, keeping the first writer's value on a concurrent insert."""
        stmt = (
            insert(ResponseCacheRow)
            .values(cache_key=cache_key, response=response)
            .on_conflict_do_nothing(index_elements=["cache_key"])
        )
        await self._session.execute(stmt)

    async def delete_cache(self, cache_keys: Sequence[str]) -> None:
        """Drop cached responses so a subsequent run executes cold."""
        if not cache_keys:
            return
        await self._session.execute(
            delete(ResponseCacheRow).where(ResponseCacheRow.cache_key.in_(list(cache_keys)))
        )
