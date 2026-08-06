"""Entry-point-backed metric registry."""

from __future__ import annotations

from importlib.metadata import entry_points

from evalharness.core.protocols import Metric
from evalharness.scoring.catalog import builtin_metrics
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig


class MetricRegistry:
    """Name-to-metric lookup for the scoring engine.

    Built-in metrics register through :meth:`defaults`, which constructs the ones needing
    injected collaborators and takes the rest from ``scoring/catalog.py``. The
    ``evalharness.metrics`` entry-point group is for optional external extras that ship
    behind their own dependency group, today only ``bertscore`` behind ``metrics-ml``.
    :meth:`discover` calls an entry point with no arguments, so an entry-point metric must
    be constructible without collaborators; anything that needs one belongs in
    :meth:`defaults`.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Metric already registered: {metric.name}")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {sorted(self._metrics)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._metrics)

    @classmethod
    def defaults(cls) -> MetricRegistry:
        registry = cls()
        registry.register(ExactMatchMetric(Normalizer(NormalizerConfig())))
        for metric in builtin_metrics():
            registry.register(metric)
        return registry

    @classmethod
    def discover(cls) -> MetricRegistry:
        """Built-in metrics plus any optional external extra on the entry-point group."""
        registry = cls.defaults()
        for ep in entry_points(group="evalharness.metrics"):
            if ep.name in registry._metrics:
                continue
            loaded = ep.load()
            metric = loaded() if isinstance(loaded, type) else loaded
            registry.register(metric)
        return registry
