"""Composition root: the one module that chooses concrete collaborators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalharness.app.container import AppContainer
from evalharness.app.settings import get_settings

if TYPE_CHECKING:
    from evalharness.domain.ports import RunStoreFactory
    from evalharness.providers.factory import ProviderBuilder


def build_container(
    *,
    build_provider: ProviderBuilder | None = None,
    run_store: RunStoreFactory | None = None,
) -> AppContainer:
    """Assemble the production wiring.

    Cheap enough to call per command: settings are cached and every other field is a
    factory or a service holding factories, so no connection, client, or engine is
    created here. The scoring engine is bound to the same store as the container, so a
    substituted store cannot be bypassed by a service that reaches for the engine
    instead, and to the settings-filtered metric registry, so ``METRIC_FAMILIES`` /
    ``METRICS_ENABLED`` apply to every command.

    ``build_provider`` and ``run_store`` default to the production choices; passing them
    keeps the caller's module the single substitution point. A caller that cannot inject,
    such as a CLI command, substitutes by rebinding the name on the module the defaults
    are resolved from, since the lazy imports below resolve per call rather than at import.
    """
    # Imported here rather than at module scope: the provider, scoring, and store
    # packages all read settings from this package, so importing them while
    # ``evalharness.app`` is still initializing would close an import cycle.
    from evalharness.providers.factory import build_managed_provider
    from evalharness.repositories import RunStoreUow
    from evalharness.scoring.engine import ScoringEngine
    from evalharness.scoring.registry import MetricRegistry
    from evalharness.services.compare import CompareService
    from evalharness.services.dataset import DatasetService
    from evalharness.services.evaluation import EvaluationService
    from evalharness.services.gates import GatesService
    from evalharness.services.judge import JudgeService
    from evalharness.services.matrix import MatrixService
    from evalharness.services.rag import RagService
    from evalharness.services.scoring import ScoringService
    from evalharness.services.suite import SuiteService

    provider_builder: ProviderBuilder = build_provider or build_managed_provider
    store: RunStoreFactory = run_store or RunStoreUow
    settings = get_settings()

    def metric_registry() -> MetricRegistry:
        return MetricRegistry.discover(settings=settings)

    def scoring_engine() -> ScoringEngine:
        return ScoringEngine(registry=metric_registry(), run_store=store)

    return AppContainer(
        settings=settings,
        build_provider=provider_builder,
        metric_registry=metric_registry,
        scoring_engine=scoring_engine,
        run_store=store,
        evaluation=EvaluationService(
            settings=settings,
            build_provider=provider_builder,
            scoring_engine=scoring_engine,
            run_store=store,
        ),
        compare=CompareService(run_store=store),
        scoring=ScoringService(scoring_engine=scoring_engine),
        judge=JudgeService(settings=settings, build_provider=provider_builder),
        rag=RagService(settings=settings, build_provider=provider_builder),
        dataset=DatasetService(),
        suite=SuiteService(),
        matrix=MatrixService(),
        gates=GatesService(),
    )
