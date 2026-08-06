"""Evaluation run pipeline, free of any CLI transport concern."""

from evalharness.pipeline.run import (
    DatasetValidationError,
    ResumeError,
    RunResult,
    RunStartedCallback,
    run_evaluation,
)

__all__ = [
    "DatasetValidationError",
    "ResumeError",
    "RunResult",
    "RunStartedCallback",
    "run_evaluation",
]
