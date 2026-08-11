"""MetricRegistry defaults / discover contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evalharness.app.settings import Settings
from evalharness.domain.dataset import Case
from evalharness.domain.enums import Requirement, TaskType
from evalharness.domain.generation import Generation
from evalharness.domain.scoring import AggregateValue, ScoreValue, ScoringContext
from evalharness.scoring.engine import ScoringEngine
from evalharness.scoring.families import METRIC_FAMILIES
from evalharness.scoring.registry import MetricRegistry

_FAKE_METRIC_NAME = "wave2_fake_discover_metric"


def _settings(*, families: str | None = None, enabled: str | None = None) -> Settings:
    """Settings with the metric flags pinned, so a developer's .env cannot steer a test."""
    return Settings(metric_families=families, metrics_enabled=enabled)


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


@dataclass(frozen=True)
class _MissingDependencyEntryPoint:
    """An entry point whose module cannot import because its extra is not installed."""

    name: str

    def load(self) -> Any:
        raise ImportError("No module named 'torch'")


def _fake_entry_points(*, group: str = "") -> tuple[_FakeEntryPoint, ...]:
    if group != "evalharness.metrics":
        return ()
    return (_FakeEntryPoint(name=_FAKE_METRIC_NAME, _loaded=_FakeDiscoverMetric),)


def _uninstalled_entry_points(*, group: str = "") -> tuple[_MissingDependencyEntryPoint, ...]:
    if group != "evalharness.metrics":
        return ()
    return (_MissingDependencyEntryPoint(name="bertscore_f1"),)


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


def test_discover_registers_every_metric_the_family_map_declares() -> None:
    """Fails on an entry point missing from pyproject, or a metric missing from the map."""
    discovered = MetricRegistry.discover(settings=_settings())

    assert discovered.names() == sorted(METRIC_FAMILIES)


def test_family_filter_drops_other_families_but_keeps_the_primary_metric() -> None:
    registry = MetricRegistry.discover(settings=_settings(families="retrieval"))

    assert "retrieval_ndcg_10" in registry.names()
    assert "exact_match" in registry.names()
    assert "rouge_l" not in registry.names()


def test_metrics_enabled_narrows_discovery_to_the_named_metrics() -> None:
    registry = MetricRegistry.discover(settings=_settings(enabled="squad_f1,retrieval_mrr"))

    assert registry.names() == ["exact_match", "retrieval_mrr", "squad_f1"]


def test_blank_flags_mean_every_discovered_metric() -> None:
    registry = MetricRegistry.discover(settings=_settings(families="  ", enabled=""))

    assert registry.names() == sorted(METRIC_FAMILIES)


def test_get_explains_a_metric_disabled_by_family() -> None:
    registry = MetricRegistry.discover(settings=_settings(families="retrieval"))

    with pytest.raises(ValueError, match="family 'overlap' is not in METRIC_FAMILIES"):
        registry.get("rouge_l")


def test_get_explains_a_metric_disabled_by_name() -> None:
    registry = MetricRegistry.discover(settings=_settings(enabled="squad_f1"))

    with pytest.raises(ValueError, match="not listed in METRICS_ENABLED"):
        registry.get("classification")


def test_get_distinguishes_an_unknown_metric_from_a_disabled_one() -> None:
    registry = MetricRegistry.discover(settings=_settings(families="retrieval"))

    with pytest.raises(ValueError, match="Unknown metric 'not_a_metric'"):
        registry.get("not_a_metric")


def test_uninstalled_metric_is_reported_rather_than_breaking_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evalharness.scoring.registry.entry_points",
        _uninstalled_entry_points,
    )

    registry = MetricRegistry.discover(settings=_settings())

    assert registry.names() == ["exact_match"]
    with pytest.raises(ValueError, match="dependency is not installed"):
        registry.get("bertscore_f1")


def test_statuses_list_enabled_and_disabled_metrics_with_their_family() -> None:
    registry = MetricRegistry.discover(settings=_settings(families="lexical"))

    statuses = {status.name: status for status in registry.statuses()}

    assert statuses["squad_f1"].enabled is True
    assert statuses["squad_f1"].family == "lexical"
    assert statuses["classification"].enabled is False
    assert statuses["classification"].reason == "family 'classification' is not in METRIC_FAMILIES"
