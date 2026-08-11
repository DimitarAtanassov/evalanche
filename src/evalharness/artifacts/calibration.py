"""Calibration artifact contract, owned by neither the producer nor the consumer.

The judge subsystem writes these artifacts and the suite loader validates them, so the
model lives here rather than in either package to keep the dependency one-directional.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field

from evalharness.domain.artifacts import StrictModel


class AgreementMetric(StrEnum):
    COHEN_KAPPA = "cohen_kappa"
    SPEARMAN = "spearman"
    KRIPPENDORFF_ALPHA = "krippendorff_alpha"


class SplitCalibration(StrictModel):
    label_set_id: str
    n: int
    agreement_metric: AgreementMetric
    agreement: float | None
    agreement_ci: tuple[float, float] | None = None


class CalibrationArtifact(StrictModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    calibration_digest: str
    judgment_digest: str
    rubric_name: str
    rubric_version: str
    holdout: SplitCalibration
    dev: SplitCalibration | None
    threshold: float
    min_holdout_n: int
    min_dev_n: int
    family_separation_ok: bool
    gating_allowed: bool
    plain_language: str
    block_reasons: list[str] = Field(default_factory=list)
