"""The collaborators an entry point needs, resolved once per invocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalharness.app.settings import Settings
    from evalharness.domain.ports import RunStoreFactory
    from evalharness.providers.factory import ProviderBuilder
    from evalharness.scoring.engine import ScoringEngineFactory
    from evalharness.scoring.registry import MetricRegistryFactory
    from evalharness.services.compare import CompareService
    from evalharness.services.dataset import DatasetService
    from evalharness.services.evaluation import EvaluationService
    from evalharness.services.gates import GatesService
    from evalharness.services.judge import JudgeService
    from evalharness.services.matrix import MatrixService
    from evalharness.services.rag import RagService
    from evalharness.services.scoring import ScoringService
    from evalharness.services.suite import SuiteService


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Resolved dependencies for one CLI invocation or script run.

    Frozen, so nothing downstream can swap a collaborator after wiring. Factory fields
    create no connection, client, or engine until a service asks for one. Service
    instances are built here so entry points never construct collaborators themselves.
    """

    settings: Settings
    build_provider: ProviderBuilder
    metric_registry: MetricRegistryFactory
    scoring_engine: ScoringEngineFactory
    run_store: RunStoreFactory
    evaluation: EvaluationService
    compare: CompareService
    scoring: ScoringService
    judge: JudgeService
    rag: RagService
    dataset: DatasetService
    suite: SuiteService
    matrix: MatrixService
    gates: GatesService
