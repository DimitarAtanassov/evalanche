"""Entry-point-backed metric registry."""

from __future__ import annotations

from importlib.metadata import entry_points

from evalharness.core.protocols import Metric
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig


class MetricRegistry:
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
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(self._metrics)}") from exc

    def names(self) -> list[str]:
        return sorted(self._metrics)

    @classmethod
    def defaults(cls) -> MetricRegistry:
        registry = cls()
        registry.register(ExactMatchMetric(Normalizer(NormalizerConfig())))
        from evalharness.scoring.catalog import builtin_metrics

        for metric in builtin_metrics():
            registry.register(metric)
        return registry

    @classmethod
    def discover(cls) -> MetricRegistry:
        registry = cls.defaults()
        for ep in entry_points(group="evalharness.metrics"):
            if ep.name in registry._metrics:
                continue
            loaded = ep.load()
            metric = loaded() if isinstance(loaded, type) else loaded
            registry.register(metric)
        return registry
