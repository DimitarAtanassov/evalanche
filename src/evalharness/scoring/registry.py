"""Entry-point-backed metric registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from evalharness.app.settings import Settings, get_settings
from evalharness.domain import Metric
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.families import METRIC_ENTRY_POINT_GROUP, family_of
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig


@dataclass(frozen=True, slots=True)
class MetricStatus:
    """A metric the entry-point group offers, and whether this process can use it."""

    name: str
    family: str
    enabled: bool
    reason: str | None = None


class MetricRegistry:
    """Name-to-metric lookup for the scoring engine.

    Every built-in metric except ``exact_match`` is registered on the
    ``evalharness.metrics`` entry-point group and loaded by :meth:`discover`. An entry
    point is called with no arguments, so ``exact_match`` (which needs an injected
    ``Normalizer``) stays in :meth:`defaults` along with anything else needing a
    collaborator.

    :meth:`discover` filters by ``METRIC_FAMILIES`` and ``METRICS_ENABLED`` and records
    why each metric it dropped is unavailable, so :meth:`get` can say whether a name is
    unknown, switched off, or missing its dependency instead of a bare KeyError.
    ``exact_match`` is exempt from the filters: it is the default primary metric, and a
    report without it has no headline.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}
        self._excluded: dict[str, MetricStatus] = {}

    def register(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Metric already registered: {metric.name}")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        metric = self._metrics.get(name)
        if metric is not None:
            return metric
        excluded = self._excluded.get(name)
        if excluded is not None:
            raise ValueError(f"Metric '{name}' is unavailable: {excluded.reason}")
        raise ValueError(f"Unknown metric '{name}'. Available: {sorted(self._metrics)}")

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def statuses(self) -> list[MetricStatus]:
        """Every metric this process discovered, enabled or not, for ``evalctl metrics list``."""
        enabled = [
            MetricStatus(name=name, family=family_of(name), enabled=True) for name in self._metrics
        ]
        return sorted([*enabled, *self._excluded.values()], key=lambda status: status.name)

    @classmethod
    def defaults(cls) -> MetricRegistry:
        """Only the metrics that need injected collaborators, so cannot come from an entry point."""
        registry = cls()
        registry.register(ExactMatchMetric(Normalizer(NormalizerConfig())))
        return registry

    @classmethod
    def discover(cls, *, settings: Settings | None = None) -> MetricRegistry:
        """Defaults plus every enabled entry-point metric whose dependencies are installed."""
        resolved = settings if settings is not None else get_settings()
        families = _selection(resolved.metric_families)
        allowed = _selection(resolved.metrics_enabled)
        registry = cls.defaults()
        for ep in entry_points(group=METRIC_ENTRY_POINT_GROUP):
            registry._load(ep, families=families, allowed=allowed)
        return registry

    def _load(
        self,
        ep: EntryPoint,
        *,
        families: frozenset[str] | None,
        allowed: frozenset[str] | None,
    ) -> None:
        if ep.name in self._metrics:
            return
        family = family_of(ep.name)
        if families is not None and family not in families:
            self._exclude(ep.name, f"family '{family}' is not in METRIC_FAMILIES")
            return
        if allowed is not None and ep.name not in allowed:
            self._exclude(ep.name, "not listed in METRICS_ENABLED")
            return
        try:
            loaded = ep.load()
        except ImportError as exc:
            self._exclude(ep.name, f"dependency is not installed ({exc})")
            return
        metric = loaded() if isinstance(loaded, type) else loaded
        self.register(metric)

    def _exclude(self, name: str, reason: str) -> None:
        self._excluded[name] = MetricStatus(
            name=name, family=family_of(name), enabled=False, reason=reason
        )


def _selection(raw: str | None) -> frozenset[str] | None:
    """Parse a comma-separated allowlist; unset or empty means no filter."""
    if raw is None:
        return None
    selected = frozenset(item.strip() for item in raw.split(",") if item.strip())
    return selected or None


type MetricRegistryFactory = Callable[[], MetricRegistry]
"""Defers entry-point loading until a caller actually scores something."""
