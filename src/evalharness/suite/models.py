"""Typed contracts for benchmark suite manifests and artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# JSON view models are intentionally open at the serialization boundary. Pydantic
# validates all contract-bearing inputs before values reach this alias.
type JsonValue = Any


class StrictModel(BaseModel):
    """Base model for versioned inputs that reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class MemberRole(StrEnum):
    """Role a run plays in a benchmark suite."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"
    REFERENCE = "reference"


class MemberRun(StrictModel):
    """One published run artifact declared by a suite."""

    path: str = Field(min_length=1)
    role: MemberRole
    label: str = Field(min_length=1)
    model: str | None = None
    prompt: str | None = None
    dataset: str | None = None
    domain: str | None = None
    task: str | None = None


class ArtifactReference(StrictModel):
    """Path to a versioned published artifact."""

    path: str = Field(min_length=1)


class PrimaryMetric(StrictModel):
    """Metric that ranks members for one dataset."""

    dataset: str = Field(min_length=1)
    metric: str = Field(min_length=1)


class SuiteManifest(StrictModel):
    """Input contract for suite.yaml schema 0.1."""

    schema_version: str
    name: str = Field(min_length=1)
    description: str | None = None
    member_runs: list[MemberRun] = Field(min_length=1)
    compares: list[ArtifactReference] = Field(default_factory=list)
    primary_metrics: list[PrimaryMetric] = Field(min_length=1)
    calibrations: list[ArtifactReference] = Field(default_factory=list)
    judge_artifacts: list[ArtifactReference] = Field(default_factory=list)
    rag_artifacts: list[ArtifactReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_declarations(self) -> SuiteManifest:
        """Reject ambiguous duplicate artifact and metric declarations."""
        member_paths = [member.path for member in self.member_runs]
        if len(member_paths) != len(set(member_paths)):
            raise ValueError("member run paths must be unique")
        compare_paths = [comparison.path for comparison in self.compares]
        if len(compare_paths) != len(set(compare_paths)):
            raise ValueError("compare paths must be unique")
        for name, references in (
            ("calibration", self.calibrations),
            ("judge artifact", self.judge_artifacts),
            ("RAG artifact", self.rag_artifacts),
        ):
            paths = [reference.path for reference in references]
            if len(paths) != len(set(paths)):
                raise ValueError(f"{name} paths must be unique")
        datasets = [primary.dataset for primary in self.primary_metrics]
        if len(datasets) != len(set(datasets)):
            raise ValueError("primary metric datasets must be unique")
        return self


class MetricAggregate(StrictModel):
    """Metric aggregate read from a run report."""

    metric: str
    version: str
    config_sha256: str
    slice: str
    n: int = Field(ge=0)
    value: float
    ci_low: float | None
    ci_high: float | None
    method: str


class RunArtifact(BaseModel):
    """Required projection of a run report 2.2."""

    model_config = ConfigDict(extra="allow", strict=True)

    schema_version: str
    run_id: str
    run_status: str
    model_digest: str
    dataset_sha256: str
    model: dict[str, JsonValue]
    dataset: dict[str, JsonValue]
    prompt_template: dict[str, JsonValue]
    coverage: float
    planned_generations: int
    written_generations: int
    coverage_floor: float
    publishable: bool
    primary_metric: str
    metric_aggregates: list[MetricAggregate]
    case_examples: list[dict[str, JsonValue]] = Field(default_factory=list)
    outcome_histogram: dict[str, int] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    finish_reasons: dict[str, int] = Field(default_factory=dict)
    harness_failures: int = Field(default=0, ge=0)
    cost_usd_total: float = 0.0
    cost_unpriced_generations: int = Field(ge=0)
    cost_per_correct: float | None = None
    retries: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_rate: float = 0.0


class ComparisonResult(StrictModel):
    """Statistical result in a compare 1.0 artifact."""

    metric: str
    n: int
    baseline: float
    candidate: float
    absolute_delta: float
    relative_delta: float | None
    cohens_h: float
    ci_low: float
    ci_high: float
    p_value: float
    significant_bh: bool


class CompareArtifact(StrictModel):
    """Required projection of a runs compare artifact 1.0."""

    schema_version: str
    baseline_run_id: str
    candidate_run_id: str
    excluded_flaky_cases: list[str] = Field(default_factory=list)
    result: ComparisonResult


class LoadedMember(BaseModel):
    """Validated member plus provenance needed during assembly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    declaration: MemberRun
    resolved_path: str
    digest: str
    report: RunArtifact


class LoadedCompare(BaseModel):
    """Validated comparison plus provenance needed during assembly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    declared_path: str
    resolved_path: str
    digest: str
    artifact: CompareArtifact


class LoadedSupplement(BaseModel):
    """Validated optional judge or RAG artifact plus deterministic provenance."""

    declared_path: str
    resolved_path: str
    digest: str
    payload: dict[str, JsonValue]


class LoadedSuite(BaseModel):
    """Fully validated suite and all referenced local artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest_path: str
    manifest: SuiteManifest
    members: list[LoadedMember]
    compares: list[LoadedCompare]
    calibrations: list[LoadedSupplement]
    judge_artifacts: list[LoadedSupplement]
    rag_artifacts: list[LoadedSupplement]


class SuiteReport(BaseModel):
    """Deterministic suite.json schema 0.1 view model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    name: str
    description: str | None
    suite_digest: str
    members: list[dict[str, JsonValue]]
    exclusions: list[dict[str, JsonValue]]
    coverage_matrix: list[dict[str, JsonValue]]
    quality_tables: list[dict[str, JsonValue]]
    leaderboards: list[dict[str, JsonValue]]
    slices: list[dict[str, JsonValue]]
    comparisons: list[dict[str, JsonValue]]
    failure_gallery: list[dict[str, JsonValue]]
    ops: dict[str, JsonValue]
    calibrations: list[dict[str, JsonValue]] | None = None
    judge_artifacts: list[dict[str, JsonValue]] | None = None
    rag_artifacts: list[dict[str, JsonValue]] | None = None
