"""``datasets`` table access."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import CaseRow, DatasetRow
from evalharness.domain.dataset import Case, DatasetRef
from evalharness.repositories.mappers import _case_reference, _dataset_ref


class DatasetRepository:
    """Datasets and the cases created with them.

    Cases are written here rather than in ``CaseRepository`` because a dataset and its
    cases are one immutable unit: the content hash covers both, so nothing may add a
    case to a dataset that already exists.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        """Return the id of the dataset at ``name@version``, creating it if absent.

        Raises ``ValueError`` when the name and version already exist with different
        content, since two different corpora under one identity would make every run
        recorded against it unreproducible.
        """
        stmt = select(DatasetRow).where(
            DatasetRow.name == name,
            DatasetRow.version == version,
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
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
        self._session.add(row)
        await self._session.flush()

        for case in cases:
            self._session.add(
                CaseRow(
                    dataset_id=row.id,
                    external_id=case.external_id,
                    task_type=case.task_type.value,
                    inputs=case.inputs,
                    reference=_case_reference(case),
                    qrels=case.qrels,
                    slices=case.slices,
                    weight=case.weight,
                )
            )
        await self._session.flush()
        return row.id

    async def get_dataset(self, dataset_id: int) -> DatasetRef | None:
        row = await self._session.get(DatasetRow, dataset_id)
        return _dataset_ref(row) if row else None
