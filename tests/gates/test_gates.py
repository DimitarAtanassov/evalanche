"""Gates validate/check contracts and fail-closed behavior."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalharness.cli import app
from evalharness.gates import (
    GatesValidationError,
    evaluate_gates,
    load_gates,
)
from evalharness.gates.models import ArtifactOverrides
from evalharness.suite.models import CompareArtifact, ComparisonResult

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "fixtures" / "gates" / "golden"
CLI = CliRunner()


@pytest.fixture
def mutable_gates(tmp_path: Path) -> Path:
    dest = tmp_path / "gates"
    shutil.copytree(GOLDEN, dest)
    return dest / "gates.yaml"


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _manifest(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_compare(path: Path, **result_overrides: object) -> None:
    result = {
        "metric": "exact_match",
        "n": 9,
        "baseline": 0.8,
        "candidate": 0.6,
        "absolute_delta": -0.2,
        "relative_delta": -0.25,
        "cohens_h": -0.4,
        "ci_low": -0.4,
        "ci_high": -0.05,
        "p_value": 0.01,
        "significant_bh": True,
    }
    result.update(result_overrides)
    payload = {
        "schema_version": "1.0",
        "baseline_run_id": "10000000-0000-4000-8000-000000000001",
        "candidate_run_id": "10000000-0000-4000-8000-000000000002",
        "excluded_flaky_cases": [],
        "result": result,
    }
    CompareArtifact.model_validate(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _mutate_run_report(path: Path, **overrides: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    for key, value in overrides.items():
        if key == "latency_ms" and isinstance(value, dict):
            existing = payload.get("latency_ms")
            assert isinstance(existing, dict)
            merged = dict(existing)
            merged.update(value)
            payload["latency_ms"] = merged
        else:
            payload[key] = value
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _keep_only_gate_kind(manifest_path: Path, kind: str) -> None:
    manifest = _manifest(manifest_path)
    gates = manifest["gates"]
    assert isinstance(gates, list)
    kept = [gate for gate in gates if isinstance(gate, dict) and gate.get("kind") == kind]
    assert kept, f"no gate of kind {kind!r} in golden manifest"
    manifest["gates"] = kept
    _write_manifest(manifest_path, manifest)


def test_golden_gates_validate_and_check_pass() -> None:
    loaded = load_gates(GOLDEN / "gates.yaml")
    assert loaded.manifest.name == "gates-golden"
    kinds = {gate.kind for gate in loaded.manifest.gates}
    assert kinds == {
        "coverage",
        "harness_failure_rate",
        "paired_regression",
        "quality_floor",
        "latency",
        "cost",
        "calibrated_judge",
    }
    result = evaluate_gates(loaded)
    assert result.blocking_failed is False
    assert result.informational_failed is False
    assert all(item.passed for item in result.results)

    validate = CLI.invoke(app, ["gates", "validate", str(GOLDEN / "gates.yaml")])
    assert validate.exit_code == 0, validate.output
    check = CLI.invoke(app, ["gates", "check", "--gates", str(GOLDEN / "gates.yaml")])
    assert check.exit_code == 0, check.output
    payload = json.loads(check.stdout)
    assert payload["blocking_failed"] is False


def test_schema_error_on_unknown_kind(mutable_gates: Path) -> None:
    manifest = _manifest(mutable_gates)
    manifest["gates"] = [
        {"name": "bad", "kind": "not_a_kind", "severity": "blocking"},
    ]
    _write_manifest(mutable_gates, manifest)
    with pytest.raises(GatesValidationError, match="INVALID_MANIFEST"):
        load_gates(mutable_gates)


def test_artifacts_suite_key_is_rejected(mutable_gates: Path) -> None:
    """Former validate-only suite binding is gone; StrictModel forbids extras."""
    manifest = _manifest(mutable_gates)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["suite"] = "suite.json"
    _write_manifest(mutable_gates, manifest)
    with pytest.raises(GatesValidationError) as exc:
        load_gates(mutable_gates)
    assert exc.value.code == "INVALID_MANIFEST"
    assert "suite" in str(exc.value)


def test_unsupported_schema_version_raises(mutable_gates: Path) -> None:
    manifest = _manifest(mutable_gates)
    manifest["schema_version"] = "99.0"
    _write_manifest(mutable_gates, manifest)
    with pytest.raises(GatesValidationError) as exc:
        load_gates(mutable_gates)
    assert exc.value.code == "UNSUPPORTED_SCHEMA"


def test_missing_run_report_binding_fails(mutable_gates: Path) -> None:
    manifest = _manifest(mutable_gates)
    manifest["artifacts"] = {"compare": "qa-compare.json"}
    manifest["gates"] = [
        {"name": "coverage-ok", "kind": "coverage", "severity": "blocking"},
    ]
    _write_manifest(mutable_gates, manifest)
    with pytest.raises(GatesValidationError) as exc:
        load_gates(mutable_gates)
    assert exc.value.code == "INVALID_MANIFEST"


def test_coverage_below_floor_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "coverage")
    _mutate_run_report(mutable_gates.parent / "qa-baseline.json", coverage=0.5)
    result = evaluate_gates(load_gates(mutable_gates))
    assert len(result.results) == 1
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_coverage_incomplete_run_status_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "coverage")
    _mutate_run_report(mutable_gates.parent / "qa-baseline.json", run_status="failed")
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_harness_failure_rate_above_max_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "harness_failure_rate")
    _mutate_run_report(
        mutable_gates.parent / "qa-baseline.json",
        harness_failures=2,
        planned_generations=10,
    )
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_harness_failure_rate_zero_planned_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "harness_failure_rate")
    _mutate_run_report(
        mutable_gates.parent / "qa-baseline.json",
        planned_generations=0,
        harness_failures=0,
    )
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_latency_p95_above_max_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "latency")
    _mutate_run_report(
        mutable_gates.parent / "qa-baseline.json",
        latency_ms={"p95": 250.0},
    )
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_latency_missing_p95_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "latency")
    report_path = mutable_gates.parent / "qa-baseline.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    latency_ms = dict(payload["latency_ms"])
    del latency_ms["p95"]
    payload["latency_ms"] = latency_ms
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_cost_above_max_usd_fails(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "cost")
    _mutate_run_report(mutable_gates.parent / "qa-baseline.json", cost_usd_total=5.0)
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_cost_unpriced_generations_fails_blocking(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "cost")
    _mutate_run_report(
        mutable_gates.parent / "qa-baseline.json",
        cost_usd_total=0.0,
        cost_unpriced_generations=3,
    )
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True
    assert result.results[0].evidence["cost_unpriced_generations"] == 3


def test_cost_explicit_zero_with_no_unpriced_passes(mutable_gates: Path) -> None:
    _keep_only_gate_kind(mutable_gates, "cost")
    _mutate_run_report(
        mutable_gates.parent / "qa-baseline.json",
        cost_usd_total=0.0,
        cost_unpriced_generations=0,
    )
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is True
    assert result.blocking_failed is False
    assert result.results[0].evidence["cost_usd_total"] == 0.0
    assert result.results[0].evidence["cost_unpriced_generations"] == 0


def test_cost_unpriced_field_absent_fails_closed_at_load(mutable_gates: Path) -> None:
    """Pre-2.2 omit-field reports must not load as fully priced at $0."""
    _keep_only_gate_kind(mutable_gates, "cost")
    report_path = mutable_gates.parent / "qa-baseline.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["cost_usd_total"] = 0.0
    payload.pop("cost_unpriced_generations", None)
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(GatesValidationError) as exc:
        load_gates(mutable_gates)
    assert exc.value.code == "INVALID_ARTIFACT"
    assert "cost_unpriced_generations" in str(exc.value)


def test_paired_regression_significant_effect_fails(mutable_gates: Path) -> None:
    _write_compare(mutable_gates.parent / "qa-compare.json")
    loaded = load_gates(mutable_gates)
    # Keep only the paired gate for a sharp assertion.
    loaded.manifest.gates = [
        gate for gate in loaded.manifest.gates if gate.kind == "paired_regression"
    ]
    result = evaluate_gates(loaded)
    assert len(result.results) == 1
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_paired_regression_non_significant_does_not_fail(mutable_gates: Path) -> None:
    _write_compare(
        mutable_gates.parent / "qa-compare.json",
        absolute_delta=-0.2,
        cohens_h=-0.4,
        significant_bh=False,
        p_value=0.4,
    )
    loaded = load_gates(mutable_gates)
    loaded.manifest.gates = [
        gate for gate in loaded.manifest.gates if gate.kind == "paired_regression"
    ]
    result = evaluate_gates(loaded)
    assert result.results[0].passed is True
    assert result.blocking_failed is False


def test_paired_regression_significant_below_min_abs_effect_does_not_fail(
    mutable_gates: Path,
) -> None:
    _write_compare(
        mutable_gates.parent / "qa-compare.json",
        absolute_delta=-0.01,
        relative_delta=-0.0125,
        cohens_h=-0.02,
        significant_bh=True,
        p_value=0.01,
    )
    loaded = load_gates(mutable_gates)
    loaded.manifest.gates = [
        gate for gate in loaded.manifest.gates if gate.kind == "paired_regression"
    ]
    result = evaluate_gates(loaded)
    assert result.results[0].passed is True
    assert result.blocking_failed is False


def test_paired_regression_improvement_does_not_fail(mutable_gates: Path) -> None:
    _write_compare(
        mutable_gates.parent / "qa-compare.json",
        baseline=0.6,
        candidate=0.8,
        absolute_delta=0.2,
        relative_delta=0.333,
        cohens_h=0.4,
        significant_bh=True,
        p_value=0.01,
    )
    loaded = load_gates(mutable_gates)
    loaded.manifest.gates = [
        gate for gate in loaded.manifest.gates if gate.kind == "paired_regression"
    ]
    result = evaluate_gates(loaded)
    assert result.results[0].passed is True


def test_calibrated_judge_blocking_without_gating_allowed_fails(mutable_gates: Path) -> None:
    manifest = _manifest(mutable_gates)
    manifest["artifacts"]["calibration"] = "calibration-blocked.json"
    manifest["gates"] = [
        {
            "name": "judge",
            "kind": "calibrated_judge",
            "severity": "blocking",
        }
    ]
    _write_manifest(mutable_gates, manifest)
    loaded = load_gates(mutable_gates)
    result = evaluate_gates(loaded)
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_calibrated_judge_below_min_agreement_fails(mutable_gates: Path) -> None:
    manifest = _manifest(mutable_gates)
    manifest["gates"] = [
        {
            "name": "judge",
            "kind": "calibrated_judge",
            "severity": "blocking",
            "min_agreement": 0.99,
        }
    ]
    _write_manifest(mutable_gates, manifest)
    result = evaluate_gates(load_gates(mutable_gates))
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.blocking_failed is True


def test_calibrated_judge_informational_without_gating_allowed_not_blocking(
    mutable_gates: Path,
) -> None:
    manifest = _manifest(mutable_gates)
    manifest["artifacts"]["calibration"] = "calibration-blocked.json"
    manifest["gates"] = [
        {
            "name": "judge",
            "kind": "calibrated_judge",
            "severity": "informational",
        }
    ]
    _write_manifest(mutable_gates, manifest)
    loaded = load_gates(mutable_gates)
    result = evaluate_gates(loaded)
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is False
    assert result.blocking_failed is False
    assert result.informational_failed is True


def test_check_exit_code_blocking_vs_informational(mutable_gates: Path) -> None:
    # Force a quality floor failure.
    manifest = _manifest(mutable_gates)
    for gate in manifest["gates"]:
        assert isinstance(gate, dict)
        if gate.get("kind") == "quality_floor":
            gate["min_value"] = 0.99
            gate["severity"] = "blocking"
    _write_manifest(mutable_gates, manifest)
    blocking = CLI.invoke(app, ["gates", "check", "--gates", str(mutable_gates)])
    assert blocking.exit_code == 1, blocking.output
    payload = json.loads(blocking.stdout)
    assert payload["blocking_failed"] is True

    for gate in manifest["gates"]:
        assert isinstance(gate, dict)
        if gate.get("kind") == "quality_floor":
            gate["severity"] = "informational"
    _write_manifest(mutable_gates, manifest)
    informational = CLI.invoke(app, ["gates", "check", "--gates", str(mutable_gates)])
    assert informational.exit_code == 0, informational.output
    info_payload = json.loads(informational.stdout)
    assert info_payload["blocking_failed"] is False
    assert info_payload["informational_failed"] is True


def test_cli_overrides_run_report(mutable_gates: Path, tmp_path: Path) -> None:
    other = tmp_path / "other-report.json"
    shutil.copyfile(mutable_gates.parent / "qa-baseline.json", other)
    loaded = load_gates(
        mutable_gates,
        overrides=ArtifactOverrides(run_report=str(other)),
    )
    assert loaded.run_report_path == str(other.resolve())


def test_cli_override_skips_path_jail_for_outside_tree_report(
    mutable_gates: Path, tmp_path: Path
) -> None:
    """CLI overrides intentionally bypass the manifest tree jail."""
    outside = tmp_path / "outside" / "report.json"
    outside.parent.mkdir(parents=True)
    shutil.copyfile(mutable_gates.parent / "qa-baseline.json", outside)
    # Manifest path still points inside; override is absolute outside the gates dir.
    loaded = load_gates(
        mutable_gates,
        overrides=ArtifactOverrides(run_report=str(outside)),
    )
    assert loaded.run_report_path == str(outside.resolve())
    assert Path(loaded.run_report_path).is_relative_to(mutable_gates.parent) is False


def test_comparison_result_contract_matches_fixture() -> None:
    """Guard the absolute_delta sign convention used by paired_regression."""
    payload = json.loads((GOLDEN / "qa-compare.json").read_text(encoding="utf-8"))
    artifact = CompareArtifact.model_validate(payload)
    assert isinstance(artifact.result, ComparisonResult)
    assert artifact.result.candidate > artifact.result.baseline
    assert artifact.result.absolute_delta > 0


@pytest.mark.parametrize(
    "declared",
    ["../outside-run.json", "nested/../../outside-run.json"],
)
def test_gates_refuses_artifacts_outside_the_manifest_directory(
    mutable_gates: Path,
    declared: str,
) -> None:
    outside = mutable_gates.parent.parent / "outside-run.json"
    shutil.copyfile(mutable_gates.parent / "qa-baseline.json", outside)
    manifest = _manifest(mutable_gates)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["run_report"] = declared
    _write_manifest(mutable_gates, manifest)

    with pytest.raises(GatesValidationError) as exc:
        load_gates(mutable_gates)

    assert exc.value.code == "ARTIFACT_OUTSIDE_GATES"
