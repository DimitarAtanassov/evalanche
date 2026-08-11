"""Use-case layer: service classes an interface (CLI, script) drives.

Every service takes its collaborators in ``__init__``. Nothing here reads settings,
builds a provider, or picks a store: that is the composition root's job
(``evalharness.app.build_container``).
"""

from evalharness.services.compare import CompareService
from evalharness.services.dataset import DatasetService
from evalharness.services.evaluation import (
    DatasetValidationError,
    EvaluationService,
    ResumeError,
    RunResult,
    RunStartedCallback,
)
from evalharness.services.gates import GatesService
from evalharness.services.judge import JudgeService
from evalharness.services.matrix import MatrixService
from evalharness.services.rag import RagService
from evalharness.services.scoring import ScoredRow, ScoringService
from evalharness.services.suite import SuiteService

__all__ = [
    "CompareService",
    "DatasetService",
    "DatasetValidationError",
    "EvaluationService",
    "GatesService",
    "JudgeService",
    "MatrixService",
    "RagService",
    "ResumeError",
    "RunResult",
    "RunStartedCallback",
    "ScoredRow",
    "ScoringService",
    "SuiteService",
]
