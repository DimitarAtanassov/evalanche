"""Shim: re-exports the evaluation service. Prefer ``evalharness.services.evaluation``."""

from evalharness.services.evaluation import (
    DatasetValidationError,
    EvaluationService,
    ResumeError,
    RunResult,
    RunStartedCallback,
)

__all__ = [
    "DatasetValidationError",
    "EvaluationService",
    "ResumeError",
    "RunResult",
    "RunStartedCallback",
]
