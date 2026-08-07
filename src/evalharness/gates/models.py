"""Typed contracts for gates.yaml and evaluation results."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalharness.artifacts.calibration import CalibrationArtifact
from evalharness.suite.models import CompareArtifact, RunArtifact, StrictModel

type JsonValue = Any


class GateSeverity(StrEnum):
    BLOCKING = "blocking"
    INFORMATIONAL = "informational"


class GateArtifacts(StrictModel):
    """Relative paths to artifacts, resolved from the gates.yaml directory.

    Unknown keys are rejected (``StrictModel`` / ``extra='forbid'``). The former
    validate-only ``suite`` binding was removed; declare only run/compare/calibration.
    """

    run_report: str | None = None
    compare: str | None = None
    calibration: str | None = None


class CoverageGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["coverage"] = "coverage"
    severity: GateSeverity
    require_completed: bool = True
    min_coverage: float | None = Field(default=None, ge=0.0, le=1.0)


class HarnessFailureRateGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["harness_failure_rate"] = "harness_failure_rate"
    severity: GateSeverity
    max_rate: float = Field(ge=0.0, le=1.0)


class PairedRegressionGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["paired_regression"] = "paired_regression"
    severity: GateSeverity
    min_abs_effect: float = Field(ge=0.0)
    min_cohens_h: float | None = Field(default=None, ge=0.0)
    max_p_value: float | None = Field(default=None, ge=0.0, le=1.0)


class QualityFloorGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["quality_floor"] = "quality_floor"
    severity: GateSeverity
    metric: str = Field(min_length=1)
    min_value: float


class CalibratedJudgeGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["calibrated_judge"] = "calibrated_judge"
    severity: GateSeverity
    min_agreement: float | None = Field(default=None, ge=0.0, le=1.0)


class LatencyGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["latency"] = "latency"
    severity: GateSeverity
    max_p95_ms: float = Field(ge=0.0)


class CostGate(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["cost"] = "cost"
    severity: GateSeverity
    max_usd: float = Field(ge=0.0)


GateSpec = Annotated[
    CoverageGate
    | HarnessFailureRateGate
    | PairedRegressionGate
    | QualityFloorGate
    | CalibratedJudgeGate
    | LatencyGate
    | CostGate,
    Field(discriminator="kind"),
]


class GatesManifest(StrictModel):
    """Input contract for gates.yaml schema 0.1."""

    schema_version: str
    name: str = Field(min_length=1)
    artifacts: GateArtifacts = Field(default_factory=GateArtifacts)
    gates: list[GateSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_names_and_bindings(self) -> GatesManifest:
        names = [gate.name for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValueError("gate names must be unique")
        kinds_needing_run = {
            "coverage",
            "harness_failure_rate",
            "quality_floor",
            "latency",
            "cost",
        }
        for gate in self.gates:
            if gate.kind in kinds_needing_run and not self.artifacts.run_report:
                raise ValueError(f"gate {gate.name!r} requires artifacts.run_report")
            if gate.kind == "paired_regression" and not self.artifacts.compare:
                raise ValueError(f"gate {gate.name!r} requires artifacts.compare")
            if gate.kind == "calibrated_judge" and not self.artifacts.calibration:
                raise ValueError(f"gate {gate.name!r} requires artifacts.calibration")
        return self


class GateResult(StrictModel):
    """One evaluated gate with evidence for operators and CI."""

    name: str
    kind: str
    severity: GateSeverity
    passed: bool
    blocking_failure: bool
    reason: str
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class GatesEvaluation(BaseModel):
    """JSON-serializable gates check result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    gates_name: str
    results: list[GateResult]
    blocking_failed: bool
    informational_failed: bool


class LoadedGates(BaseModel):
    """Validated gates manifest plus optional loaded artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest_path: str
    manifest: GatesManifest
    run_report: RunArtifact | None = None
    run_report_path: str | None = None
    compare: CompareArtifact | None = None
    compare_path: str | None = None
    calibration: CalibrationArtifact | None = None
    calibration_path: str | None = None


class ArtifactOverrides(StrictModel):
    """CLI path overrides for bound artifacts (absolute or relative to cwd)."""

    run_report: str | None = None
    compare: str | None = None
    calibration: str | None = None

    @field_validator("run_report", "compare", "calibration", mode="before")
    @classmethod
    def empty_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value
