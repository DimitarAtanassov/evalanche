"""MetricRegistry defaults / discover contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evalharness.core.enums import Requirement, TaskType
from evalharness.core.models import AggregateValue, Case, Generation, ScoreValue, ScoringContext
from evalharness.scoring.engine import ScoringEngine
from evalharness.scoring.registry import MetricRegistry

_FAKE_METRIC_NAME = "wave2_fake_discover_metric"


@dataclass(frozen=True)
class _FakeDiscoverMetric:
    """Minimal Metric stand-in loadable via a fake entry point."""

    name: str = _FAKE_METRIC_NAME
    version: str = "0.0.1"
    task_types: frozenset[TaskType] = frozenset({TaskType.GENERATION})
    requires: frozenset[Requirement] = frozenset()

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]:
        return []

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        return AggregateValue(
            metric_name=self.name,
            metric_version=self.version,
            slice_key="__overall__",
            n=0,
            value=0.0,
            ci_low=None,
            ci_high=None,
            stddev=None,
            method="none",
        )


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    _loaded: Any

    def load(self) -> Any:
        return self._loaded


def _fake_entry_points(*, group: str = "") -> tuple[_FakeEntryPoint, ...]:
    if group != "evalharness.metrics":
        return ()
    return (_FakeEntryPoint(name=_FAKE_METRIC_NAME, _loaded=_FakeDiscoverMetric),)


def test_discover_includes_defaults_and_does_not_crash() -> None:
    defaults = MetricRegistry.defaults()
    discovered = MetricRegistry.discover()

    assert "exact_match" in discovered.names()
    assert set(defaults.names()).issubset(set(discovered.names()))


def test_discover_registers_entry_point_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evalharness.scoring.registry.entry_points",
        _fake_entry_points,
    )

    discovered = MetricRegistry.discover()

    assert _FAKE_METRIC_NAME in discovered.names()
    assert "exact_match" in discovered.names()


def test_scoring_engine_defaults_to_discover_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evalharness.scoring.registry.entry_points",
        _fake_entry_points,
    )

    engine = ScoringEngine()

    assert _FAKE_METRIC_NAME in engine.registry.names()
    assert "exact_match" in engine.registry.names()
    assert set(MetricRegistry.defaults().names()).issubset(set(engine.registry.names()))
