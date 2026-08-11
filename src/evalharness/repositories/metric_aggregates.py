"""``metric_aggregates`` table access."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.db.models import MetricAggregateRow
from evalharness.domain.scoring import StoredAggregate
from evalharness.repositories.mappers import _stored_aggregate


class MetricAggregateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        """Write an aggregate, overwriting any earlier value for the same identity.

        Unlike a score, an aggregate is a recomputable summary: a re-aggregation over
        more generations must replace the stale number, not be dropped.
        """
        stmt = insert(MetricAggregateRow).values(
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
        stmt = stmt.on_conflict_do_update(
            constraint="uq_metric_aggregates_identity",
            set_={
                "n": stmt.excluded.n,
                "value": stmt.excluded.value,
                "ci_low": stmt.excluded.ci_low,
                "ci_high": stmt.excluded.ci_high,
                "stddev": stmt.excluded.stddev,
                "method": stmt.excluded.method,
            },
        )
        await self._session.execute(stmt)

    async def get_metric_aggregates(self, run_id: uuid.UUID) -> list[StoredAggregate]:
        stmt = (
            select(MetricAggregateRow)
            .where(MetricAggregateRow.run_id == run_id)
            .order_by(
                MetricAggregateRow.metric_name,
                MetricAggregateRow.metric_version,
                MetricAggregateRow.slice_key,
            )
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_stored_aggregate(row) for row in rows]
