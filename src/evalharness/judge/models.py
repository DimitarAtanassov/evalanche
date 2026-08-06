"""Typed contracts for rubrics, labels, judgments, and calibration artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The aliased names are re-exported: the judge subsystem produces calibration
# artifacts, so it keeps publishing their contract under its own namespace.
from evalharness.artifacts.calibration import AgreementMetric as AgreementMetric
from evalharness.artifacts.calibration import CalibrationArtifact as CalibrationArtifact
from evalharness.artifacts.calibration import SplitCalibration as SplitCalibration
from evalharness.artifacts.calibration import StrictModel


class JudgeMode(StrEnum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


class LabelShape(StrEnum):
    ORDINAL_SCORE = "ordinal_score"
    PREFERENCE = "preference"
    NOMINAL = "nominal"


class LabelSplit(StrEnum):
    DEV = "dev"
    HOLDOUT = "holdout"


class ScaleSpec(StrictModel):
    min: int
    max: int
    anchors: dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bounds(self) -> ScaleSpec:
        if self.min >= self.max:
            raise ValueError("scale.min must be less than scale.max")
        return self


class RubricCalibration(StrictModel):
    agreement_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    agreement_metric: AgreementMetric = AgreementMetric.COHEN_KAPPA
    min_holdout_n: int = Field(default=150, ge=1)
    min_dev_n: int = Field(default=50, ge=1)


class Rubric(StrictModel):
    schema_version: Literal["0.1"]
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    mode: JudgeMode
    scale: ScaleSpec
    instructions: str = Field(min_length=1)
    require_reasoning_before_score: bool = True
    calibration: RubricCalibration = Field(default_factory=RubricCalibration)
    forbidden_candidate_families: list[str] = Field(default_factory=list)


class HumanLabel(StrictModel):
    schema_version: Literal["0.1"]
    rubric_name: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    label_shape: LabelShape
    value: int | str | float
    split: LabelSplit
    label_set_id: str = Field(min_length=1)


class PointwiseCandidate(StrictModel):
    case_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    candidate_text: str = Field(min_length=1)
    prompt: str | None = None
    reference: str | None = None


class PairwisePair(StrictModel):
    case_id: str = Field(min_length=1)
    a_generation_id: str = Field(min_length=1)
    b_generation_id: str = Field(min_length=1)
    a_model_label: str = Field(min_length=1)
    b_model_label: str = Field(min_length=1)
    a_text: str = Field(min_length=1)
    b_text: str = Field(min_length=1)
    prompt: str | None = None

    @model_validator(mode="after")
    def reject_self_pair(self) -> PairwisePair:
        if self.a_model_label == self.b_model_label:
            raise ValueError("SELF_PAIR")
        return self


class MockPointwiseResponse(StrictModel):
    generation_id: str = Field(min_length=1)
    score: int
    reasoning: str = ""


class MockPairwiseResponse(StrictModel):
    case_id: str = Field(min_length=1)
    swap_position: Literal[0, 1]
    preference: Literal["A", "B", "tie"]
    reasoning: str = ""


class JudgeModelIdentity(StrictModel):
    provider: str
    model: str
    resolved_version: str


class PairwiseOrdering(StrictModel):
    swap_position: Literal[0, 1]
    preference: Literal["A", "B", "tie"]
    reasoning: str


class PointwiseItem(StrictModel):
    case_id: str
    generation_id: str
    score: int | None
    reasoning: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    outcome: str | None = None


class PairwiseItem(StrictModel):
    case_id: str
    a_generation_id: str
    b_generation_id: str
    a_model_label: str
    b_model_label: str
    orderings: list[PairwiseOrdering]
    consistent: bool
    final_preference: Literal["A", "B", "tie"]


class BradleyTerryRefused(StrictModel):
    status: Literal["refused"] = "refused"
    reason: Literal["DISCONNECTED_PAIRWISE_GRAPH"] = "DISCONNECTED_PAIRWISE_GRAPH"
    n_models: int
    n_edges: int
    component_sizes: list[int]
    isolated_models: list[str]


class PairwiseWinRates(StrictModel):
    """Connected pairwise graph summarised by raw win rates.

    The harness does not fit Bradley-Terry strengths, so ``status`` and ``win_rates``
    name what the numbers actually are. A consumer that needs identifiable BT
    strengths must not read these as such.
    """

    status: Literal["win_rates_only"] = "win_rates_only"
    n_models: int
    n_edges: int
    win_rates: dict[str, float]
    component_sizes: list[int] = Field(default_factory=list)
    isolated_models: list[str] = Field(default_factory=list)


class PairwiseSummary(StrictModel):
    n_pairs: int
    swap_consistency: float | None
    position_bias: float | None
    bradley_terry: PairwiseWinRates | BradleyTerryRefused | None


class LatencySummary(StrictModel):
    p50: float
    p95: float


class JudgmentArtifact(BaseModel):
    """Judgment run artifact. Open at the boundary for forward-compatible fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    mode: JudgeMode
    rubric_name: str
    rubric_version: str
    judge_model: JudgeModelIdentity
    candidate_model_family: str
    judge_model_family: str
    gating_allowed: bool = False
    gating_block_reason: str
    calibration_digest: str | None = None
    cost_usd_total: float = 0.0
    latency_ms: LatencySummary
    items: list[dict[str, Any]]
    pairwise_summary: PairwiseSummary | None = None
    seed: int | None = None

    @field_validator("candidate_model_family", "judge_model_family")
    @classmethod
    def nonempty_family(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model family must be non-empty")
        return value

    @model_validator(mode="after")
    def require_calibration_for_gating(self) -> JudgmentArtifact:
        """Reject a gate bit that is not linked to a calibration artifact."""
        if self.gating_allowed and self.calibration_digest is None:
            raise ValueError("gating_allowed requires calibration_digest")
        return self
