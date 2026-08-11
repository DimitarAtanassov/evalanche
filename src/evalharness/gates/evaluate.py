"""Evaluate gates.yaml policies against loaded run/compare/calibration artifacts."""

from __future__ import annotations

from evalharness.artifacts.calibration import CalibrationArtifact
from evalharness.domain.constants import GATES_SCHEMA_VERSION, OVERALL_SLICE
from evalharness.gates.errors import GatesValidationError
from evalharness.gates.models import (
    CalibratedJudgeGate,
    CostGate,
    CoverageGate,
    GateResult,
    GatesEvaluation,
    GateSeverity,
    GateSpec,
    HarnessFailureRateGate,
    LatencyGate,
    LoadedGates,
    PairedRegressionGate,
    QualityFloorGate,
)
from evalharness.suite.models import CompareArtifact, RunArtifact


def _result(
    gate: GateSpec,
    *,
    passed: bool,
    reason: str,
    evidence: dict[str, object] | None = None,
) -> GateResult:
    blocking_failure = gate.severity == GateSeverity.BLOCKING and not passed
    return GateResult(
        name=gate.name,
        kind=gate.kind,
        severity=gate.severity,
        passed=passed,
        blocking_failure=blocking_failure,
        reason=reason,
        evidence=evidence or {},
    )


def _require_run(loaded: LoadedGates, gate: GateSpec) -> RunArtifact:
    if loaded.run_report is None:
        raise GatesValidationError(
            "MISSING_ARTIFACT",
            f"gate {gate.name!r} requires a run_report artifact",
        )
    return loaded.run_report


def _require_compare(loaded: LoadedGates, gate: GateSpec) -> CompareArtifact:
    if loaded.compare is None:
        raise GatesValidationError(
            "MISSING_ARTIFACT",
            f"gate {gate.name!r} requires a compare artifact",
        )
    return loaded.compare


def _require_calibration(loaded: LoadedGates, gate: GateSpec) -> CalibrationArtifact:
    if loaded.calibration is None:
        raise GatesValidationError(
            "MISSING_ARTIFACT",
            f"gate {gate.name!r} requires a calibration artifact",
        )
    return loaded.calibration


def _eval_coverage(gate: CoverageGate, report: RunArtifact) -> GateResult:
    floor = gate.min_coverage if gate.min_coverage is not None else report.coverage_floor
    evidence: dict[str, object] = {
        "run_status": report.run_status,
        "coverage": report.coverage,
        "coverage_floor": report.coverage_floor,
        "min_coverage": floor,
        "require_completed": gate.require_completed,
    }
    if gate.require_completed and report.run_status != "completed":
        return _result(
            gate,
            passed=False,
            reason=f"run_status is {report.run_status!r}, expected 'completed'",
            evidence=evidence,
        )
    if report.coverage < floor:
        return _result(
            gate,
            passed=False,
            reason=f"coverage {report.coverage} is below min_coverage {floor}",
            evidence=evidence,
        )
    return _result(gate, passed=True, reason="coverage ok", evidence=evidence)


def _eval_harness_failure_rate(gate: HarnessFailureRateGate, report: RunArtifact) -> GateResult:
    planned = report.planned_generations
    rate = (report.harness_failures / planned) if planned else None
    evidence: dict[str, object] = {
        "harness_failures": report.harness_failures,
        "planned_generations": planned,
        "rate": rate,
        "max_rate": gate.max_rate,
    }
    if planned == 0:
        return _result(
            gate,
            passed=False,
            reason="planned_generations is 0",
            evidence=evidence,
        )
    assert rate is not None
    if rate > gate.max_rate:
        return _result(
            gate,
            passed=False,
            reason=f"harness failure rate {rate} exceeds max_rate {gate.max_rate}",
            evidence=evidence,
        )
    return _result(gate, passed=True, reason="harness failure rate ok", evidence=evidence)


def _is_significant(gate: PairedRegressionGate, compare: CompareArtifact) -> bool:
    if gate.max_p_value is not None:
        return compare.result.p_value <= gate.max_p_value
    return compare.result.significant_bh


def _has_practical_effect(gate: PairedRegressionGate, compare: CompareArtifact) -> bool:
    delta = abs(compare.result.absolute_delta)
    if delta < gate.min_abs_effect:
        return False
    if gate.min_cohens_h is not None and abs(compare.result.cohens_h) < gate.min_cohens_h:
        return False
    return True


def _eval_paired_regression(gate: PairedRegressionGate, compare: CompareArtifact) -> GateResult:
    result = compare.result
    significant = _is_significant(gate, compare)
    practical = _has_practical_effect(gate, compare)
    evidence: dict[str, object] = {
        "metric": result.metric,
        "absolute_delta": result.absolute_delta,
        "cohens_h": result.cohens_h,
        "p_value": result.p_value,
        "significant_bh": result.significant_bh,
        "significant": significant,
        "practical_effect": practical,
        "min_abs_effect": gate.min_abs_effect,
        "min_cohens_h": gate.min_cohens_h,
        "max_p_value": gate.max_p_value,
    }
    # Positive delta means candidate better; never a paired-regression failure.
    if result.absolute_delta >= 0:
        return _result(
            gate,
            passed=True,
            reason="candidate is not worse than baseline (absolute_delta >= 0)",
            evidence=evidence,
        )
    if significant and practical:
        return _result(
            gate,
            passed=False,
            reason=(
                "significant regression with practical effect "
                f"(absolute_delta={result.absolute_delta})"
            ),
            evidence=evidence,
        )
    return _result(
        gate,
        passed=True,
        reason="no significant practical regression",
        evidence=evidence,
    )


def _eval_quality_floor(gate: QualityFloorGate, report: RunArtifact) -> GateResult:
    row = next(
        (
            item
            for item in report.metric_aggregates
            if item.metric == gate.metric and item.slice == OVERALL_SLICE
        ),
        None,
    )
    evidence: dict[str, object] = {
        "metric": gate.metric,
        "slice": OVERALL_SLICE,
        "min_value": gate.min_value,
        "value": row.value if row is not None else None,
    }
    if row is None:
        return _result(
            gate,
            passed=False,
            reason=f"metric {gate.metric!r} missing at slice {OVERALL_SLICE}",
            evidence=evidence,
        )
    if row.value < gate.min_value:
        return _result(
            gate,
            passed=False,
            reason=f"metric {gate.metric!r} value {row.value} is below min_value {gate.min_value}",
            evidence=evidence,
        )
    return _result(gate, passed=True, reason="quality floor met", evidence=evidence)


def _eval_calibrated_judge(
    gate: CalibratedJudgeGate, calibration: CalibrationArtifact
) -> GateResult:
    evidence: dict[str, object] = {
        "gating_allowed": calibration.gating_allowed,
        "holdout_agreement": calibration.holdout.agreement,
        "min_agreement": gate.min_agreement,
        "block_reasons": list(calibration.block_reasons),
    }
    if not calibration.gating_allowed:
        return _result(
            gate,
            passed=False,
            reason="calibration does not allow gating (gating_allowed=false)",
            evidence=evidence,
        )
    if gate.min_agreement is not None:
        agreement = calibration.holdout.agreement
        if agreement is None or agreement < gate.min_agreement:
            return _result(
                gate,
                passed=False,
                reason=(
                    f"holdout agreement {agreement!r} is below min_agreement {gate.min_agreement}"
                ),
                evidence=evidence,
            )
    return _result(gate, passed=True, reason="calibrated judge ok", evidence=evidence)


def _eval_latency(gate: LatencyGate, report: RunArtifact) -> GateResult:
    p95 = report.latency_ms.get("p95")
    evidence: dict[str, object] = {
        "p95_ms": p95,
        "max_p95_ms": gate.max_p95_ms,
        "latency_ms": dict(report.latency_ms),
    }
    if p95 is None:
        return _result(
            gate,
            passed=False,
            reason="latency_ms.p95 is missing from the run report",
            evidence=evidence,
        )
    if p95 > gate.max_p95_ms:
        return _result(
            gate,
            passed=False,
            reason=f"p95 latency {p95} ms exceeds max_p95_ms {gate.max_p95_ms}",
            evidence=evidence,
        )
    return _result(gate, passed=True, reason="latency ok", evidence=evidence)


def _eval_cost(gate: CostGate, report: RunArtifact) -> GateResult:
    evidence: dict[str, object] = {
        "cost_usd_total": report.cost_usd_total,
        "cost_unpriced_generations": report.cost_unpriced_generations,
        "max_usd": gate.max_usd,
    }
    if report.cost_unpriced_generations > 0:
        return _result(
            gate,
            passed=False,
            reason=(
                f"cost_unpriced_generations {report.cost_unpriced_generations} > 0 "
                "(missing cost fails closed for CostGate)"
            ),
            evidence=evidence,
        )
    if report.cost_usd_total > gate.max_usd:
        return _result(
            gate,
            passed=False,
            reason=(f"cost_usd_total {report.cost_usd_total} exceeds max_usd {gate.max_usd}"),
            evidence=evidence,
        )
    return _result(gate, passed=True, reason="cost ok", evidence=evidence)


def _eval_one(gate: GateSpec, loaded: LoadedGates) -> GateResult:
    if isinstance(gate, CoverageGate):
        return _eval_coverage(gate, _require_run(loaded, gate))
    if isinstance(gate, HarnessFailureRateGate):
        return _eval_harness_failure_rate(gate, _require_run(loaded, gate))
    if isinstance(gate, PairedRegressionGate):
        return _eval_paired_regression(gate, _require_compare(loaded, gate))
    if isinstance(gate, QualityFloorGate):
        return _eval_quality_floor(gate, _require_run(loaded, gate))
    if isinstance(gate, CalibratedJudgeGate):
        return _eval_calibrated_judge(gate, _require_calibration(loaded, gate))
    if isinstance(gate, LatencyGate):
        return _eval_latency(gate, _require_run(loaded, gate))
    if isinstance(gate, CostGate):
        return _eval_cost(gate, _require_run(loaded, gate))
    raise GatesValidationError("INVALID_MANIFEST", f"unsupported gate kind: {gate.kind}")


def evaluate_gates(loaded: LoadedGates) -> GatesEvaluation:
    """Evaluate every gate in a loaded manifest; never raises on a failed gate."""
    results = [_eval_one(gate, loaded) for gate in loaded.manifest.gates]
    blocking_failed = any(item.blocking_failure for item in results)
    informational_failed = any(
        item.severity == GateSeverity.INFORMATIONAL and not item.passed for item in results
    )
    return GatesEvaluation(
        schema_version=GATES_SCHEMA_VERSION,
        gates_name=loaded.manifest.name,
        results=results,
        blocking_failed=blocking_failed,
        informational_failed=informational_failed,
    )
