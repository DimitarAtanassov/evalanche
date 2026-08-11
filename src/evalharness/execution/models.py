"""Value types passed between the execution planner, worker pool, and case runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome
from evalharness.domain.generation import GenerationResponse


@dataclass(frozen=True)
class RunPlanItem:
    case_db_id: int
    case: Case
    repeat_idx: int


@dataclass(frozen=True)
class RunConfig:
    dataset_id: int
    prompt_template_id: int
    model_version_id: int
    config_sha256: str
    decode_params: dict[str, Any]
    repeats: int
    concurrency: int
    case_timeout_s: float
    request_timeout_s: float
    run_timeout_s: float
    drain_timeout_s: float
    max_retries: int


@dataclass(frozen=True)
class ExecutionResult:
    case_id: int
    external_id: str
    repeat_idx: int
    outcome: FailureOutcome
    attempts: int
    cached: bool
    duration_ms: float | None
    # False when shutdown skipped persist; do not count as completed work.
    persisted: bool = True


@dataclass(frozen=True)
class AttemptOutcome:
    """Result of a cache hit or of the provider retry loop for one case."""

    response: GenerationResponse | None
    attempt_log: list[dict[str, Any]]
    harness_error: bool
    harness_timeout: bool
    cached: bool
