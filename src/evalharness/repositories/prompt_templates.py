"""``prompt_templates`` table access."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import PromptTemplateRow
from evalharness.domain.run import PromptTemplateRef
from evalharness.repositories.mappers import _prompt_template_ref


class PromptTemplateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_prompt_template(self, name: str, version: str, body: str, sha256: str) -> int:
        """Return the id of ``name@version``, creating it if absent.

        A version is immutable: reusing one with a different body would silently
        re-interpret every run already recorded against it.
        """
        stmt = select(PromptTemplateRow).where(
            PromptTemplateRow.name == name,
            PromptTemplateRow.version == version,
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing:
            if existing.content_sha256 != sha256:
                raise ValueError(
                    f"Prompt template {name}@{version} already exists with different content"
                )
            return existing.id
        row = PromptTemplateRow(name=name, version=version, body=body, content_sha256=sha256)
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def get_prompt_template(self, prompt_template_id: int) -> PromptTemplateRef | None:
        row = await self._session.get(PromptTemplateRow, prompt_template_id)
        return _prompt_template_ref(row) if row else None
