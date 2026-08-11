"""Score and aggregate domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoreValue:
    metric_name: str
    metric_version: str
    metric_config_sha256: str
    value: float | None
    passed: bool | None
    detail: dict[str, Any]


@dataclass(frozen=True)
class AggregateValue:
    metric_name: str
    metric_version: str
    slice_key: str
    n: int
    value: float
    ci_low: float | None
    ci_high: float | None
    stddev: float | None
    method: str


@dataclass
class ScoringContext:
    normalizer_id: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredScore:
    """A persisted per-generation score returned by the store port."""

    id: int
    generation_id: int
    metric_name: str
    metric_version: str
    metric_config_sha256: str
    value: float | None
    passed: bool | None
    detail: dict[str, Any] | None


@dataclass(frozen=True)
class StoredAggregate:
    """A persisted metric aggregate returned by the store port."""

    id: int
    run_id: str
    metric_name: str
    metric_version: str
    metric_config_sha256: str
    slice_key: str
    n: int
    value: float
    ci_low: float | None
    ci_high: float | None
    stddev: float | None
    # Writes always name a method; the column is nullable, so rows written before it
    # was recorded read back without one rather than being relabelled with a guess.
    method: str | None
