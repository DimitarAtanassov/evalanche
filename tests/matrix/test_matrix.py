"""Matrix and baseline pin validation and promotion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalharness.cli import app
from evalharness.hashing import sha256_hex
from evalharness.matrix import (
    MatrixValidationError,
    load_baseline,
    load_matrix,
    promote_baseline,
)

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "fixtures" / "matrix" / "golden"
CLI = CliRunner()


@pytest.fixture
def mutable_matrix(tmp_path: Path) -> Path:
    dest = tmp_path / "matrix"
    shutil.copytree(GOLDEN, dest)
    return dest / "matrix.yaml"


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_golden_matrix_validate() -> None:
    loaded = load_matrix(GOLDEN / "matrix.yaml")
    assert loaded.manifest.name == "matrix-golden"
    assert loaded.matrix_digest == (
        "cac5586f43f159d31a23f37fb87eb3b752600d6dcde707eb6580b3f642dc679d"
    )
    result = CLI.invoke(app, ["matrix", "validate", str(GOLDEN / "matrix.yaml")])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["matrix_digest"] == loaded.matrix_digest


def test_matrix_digest_command() -> None:
    loaded = load_matrix(GOLDEN / "matrix.yaml")
    result = CLI.invoke(app, ["matrix", "digest", str(GOLDEN / "matrix.yaml")])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["matrix_digest"] == loaded.matrix_digest
    assert payload["name"] == "matrix-golden"


def test_matrix_rejects_latest_revision(mutable_matrix: Path) -> None:
    manifest = _read_yaml(mutable_matrix)
    models = manifest["models"]
    assert isinstance(models, list)
    assert isinstance(models[0], dict)
    models[0]["revision"] = "latest"
    _write_yaml(mutable_matrix, manifest)
    with pytest.raises(MatrixValidationError, match="INVALID_MANIFEST"):
        load_matrix(mutable_matrix)


def test_matrix_rejects_latest_model_name(mutable_matrix: Path) -> None:
    manifest = _read_yaml(mutable_matrix)
    models = manifest["models"]
    assert isinstance(models, list)
    assert isinstance(models[0], dict)
    models[0]["model"] = "Latest"
    _write_yaml(mutable_matrix, manifest)
    with pytest.raises(MatrixValidationError, match="INVALID_MANIFEST"):
        load_matrix(mutable_matrix)


def test_matrix_digest_mismatch_fails(mutable_matrix: Path) -> None:
    manifest = _read_yaml(mutable_matrix)
    prompts = manifest["prompts"]
    assert isinstance(prompts, list)
    assert isinstance(prompts[0], dict)
    prompts[0]["digest"] = "0" * 64
    _write_yaml(mutable_matrix, manifest)
    with pytest.raises(MatrixValidationError, match="DIGEST_MISMATCH"):
        load_matrix(mutable_matrix)


def test_matrix_unsupported_schema_version_raises(mutable_matrix: Path) -> None:
    manifest = _read_yaml(mutable_matrix)
    manifest["schema_version"] = "99.0"
    _write_yaml(mutable_matrix, manifest)
    with pytest.raises(MatrixValidationError) as exc:
        load_matrix(mutable_matrix)
    assert exc.value.code == "UNSUPPORTED_SCHEMA"


def test_baseline_validate_golden() -> None:
    loaded_matrix = load_matrix(GOLDEN / "matrix.yaml")
    loaded = load_baseline(
        GOLDEN / "baseline.yaml",
        matrix=GOLDEN / "matrix.yaml",
    )
    assert loaded.manifest.matrix_name == "matrix-golden"
    assert loaded.manifest.matrix_digest == loaded_matrix.matrix_digest
    result = CLI.invoke(
        app,
        [
            "baseline",
            "validate",
            str(GOLDEN / "baseline.yaml"),
            "--matrix",
            str(GOLDEN / "matrix.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_baseline_validate_catches_digest_mismatch(mutable_matrix: Path) -> None:
    baseline_path = mutable_matrix.parent / "baseline.yaml"
    manifest = _read_yaml(baseline_path)
    cells = manifest["pinned_cells"]
    assert isinstance(cells, list)
    assert isinstance(cells[0], dict)
    cells[0]["run_report_digest"] = "0" * 64
    _write_yaml(baseline_path, manifest)
    with pytest.raises(MatrixValidationError, match="DIGEST_MISMATCH"):
        load_baseline(baseline_path)


def test_baseline_matrix_digest_mismatch_fails(mutable_matrix: Path) -> None:
    baseline_path = mutable_matrix.parent / "baseline.yaml"
    manifest = _read_yaml(baseline_path)
    manifest["matrix_digest"] = "0" * 64
    _write_yaml(baseline_path, manifest)
    with pytest.raises(MatrixValidationError) as exc:
        load_baseline(baseline_path, matrix=mutable_matrix)
    assert exc.value.code == "MATRIX_MISMATCH"


def test_baseline_matrix_name_mismatch_fails(mutable_matrix: Path) -> None:
    baseline_path = mutable_matrix.parent / "baseline.yaml"
    manifest = _read_yaml(baseline_path)
    manifest["matrix_name"] = "other-matrix"
    _write_yaml(baseline_path, manifest)
    with pytest.raises(MatrixValidationError) as exc:
        load_baseline(baseline_path, matrix=mutable_matrix)
    assert exc.value.code == "MATRIX_MISMATCH"


def test_baseline_unknown_cell_with_matrix_fails(mutable_matrix: Path) -> None:
    baseline_path = mutable_matrix.parent / "baseline.yaml"
    manifest = _read_yaml(baseline_path)
    cells = manifest["pinned_cells"]
    assert isinstance(cells, list)
    assert isinstance(cells[0], dict)
    cells[0]["cell_id"] = "not-a-real-cell"
    _write_yaml(baseline_path, manifest)
    with pytest.raises(MatrixValidationError) as exc:
        load_baseline(baseline_path, matrix=mutable_matrix)
    assert exc.value.code == "UNKNOWN_CELL"


def test_baseline_promote_validate_golden_roundtrip(tmp_path: Path) -> None:
    """CI golden path: fail-closed promote then validate against matrix."""
    output = tmp_path / "baseline.yaml"
    report = GOLDEN / "qa-baseline-match.json"
    promote = CLI.invoke(
        app,
        [
            "baseline",
            "promote",
            "--matrix",
            str(GOLDEN / "matrix.yaml"),
            "--cell",
            "mock-small-qa-smoke",
            "--run-report",
            str(report),
            "--output",
            str(output),
            "--name",
            "golden-promoted",
        ],
    )
    assert promote.exit_code == 0, promote.output
    validate = CLI.invoke(
        app,
        [
            "baseline",
            "validate",
            str(output),
            "--matrix",
            str(GOLDEN / "matrix.yaml"),
        ],
    )
    assert validate.exit_code == 0, validate.output
    payload = json.loads(validate.stdout)
    assert payload["valid"] is True
    assert payload["name"] == "golden-promoted"
    assert payload["matrix_name"] == "matrix-golden"
    loaded = load_baseline(output, matrix=GOLDEN / "matrix.yaml")
    assert loaded.manifest.pinned_cells[0].run_report_digest == sha256_hex(report.read_bytes())


def test_baseline_promote_pins_digests(mutable_matrix: Path, tmp_path: Path) -> None:
    report = mutable_matrix.parent / "qa-baseline-match.json"
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert isinstance(report_payload, dict)
    expected_config = report_payload["config_sha256"]
    loaded_matrix = load_matrix(mutable_matrix)
    output = tmp_path / "promoted.yaml"
    baseline = promote_baseline(
        matrix_path=mutable_matrix,
        cell_id="mock-small-qa-smoke",
        run_report_path=report,
        output_path=output,
        name="promoted",
    )
    assert baseline.pinned_cells[0].run_report_digest == sha256_hex(report.read_bytes())
    assert baseline.pinned_cells[0].config_sha256 == expected_config
    assert baseline.pinned_cells[0].model_digest == "mock-small-v1"
    assert baseline.matrix_digest == loaded_matrix.matrix_digest
    assert "latest" not in output.read_text(encoding="utf-8").casefold()

    result = CLI.invoke(
        app,
        [
            "baseline",
            "promote",
            "--matrix",
            str(mutable_matrix),
            "--cell",
            "mock-small-qa-smoke",
            "--run-report",
            str(report),
            "--output",
            str(output),
            "--name",
            "promoted-cli",
        ],
    )
    assert result.exit_code == 0, result.output
    reloaded = load_baseline(output, matrix=mutable_matrix)
    assert reloaded.manifest.name == "promoted-cli"
    assert reloaded.manifest.matrix_digest == loaded_matrix.matrix_digest
    assert reloaded.manifest.pinned_cells[0].config_sha256 == expected_config
    assert reloaded.manifest.pinned_cells[0].run_report_digest == sha256_hex(report.read_bytes())


def test_baseline_promote_rejects_cell_identity_mismatch(
    mutable_matrix: Path, tmp_path: Path
) -> None:
    """qa-baseline.json digests intentionally disagree with matrix.yaml pins."""
    report = mutable_matrix.parent / "qa-baseline.json"
    output = tmp_path / "promoted.yaml"
    with pytest.raises(MatrixValidationError) as exc:
        promote_baseline(
            matrix_path=mutable_matrix,
            cell_id="mock-small-qa-smoke",
            run_report_path=report,
            output_path=output,
        )
    assert exc.value.code == "CELL_IDENTITY_MISMATCH"


def test_baseline_promote_allow_mismatch_skips_identity(
    mutable_matrix: Path, tmp_path: Path
) -> None:
    report = mutable_matrix.parent / "qa-baseline.json"
    output = tmp_path / "promoted.yaml"
    baseline = promote_baseline(
        matrix_path=mutable_matrix,
        cell_id="mock-small-qa-smoke",
        run_report_path=report,
        output_path=output,
        name="mismatched",
        allow_mismatch=True,
    )
    assert baseline.pinned_cells[0].run_report_digest == sha256_hex(report.read_bytes())
    result = CLI.invoke(
        app,
        [
            "baseline",
            "promote",
            "--matrix",
            str(mutable_matrix),
            "--cell",
            "mock-small-qa-smoke",
            "--run-report",
            str(report),
            "--output",
            str(output),
            "--name",
            "mismatched-cli",
            "--allow-mismatch",
        ],
    )
    assert result.exit_code == 0, result.output
    reloaded = load_baseline(output, matrix=mutable_matrix)
    assert reloaded.manifest.name == "mismatched-cli"


def test_baseline_promote_provider_model_checked_when_model_digest_is_null(
    mutable_matrix: Path, tmp_path: Path
) -> None:
    """Matrix model.digest null skips digest-only check; provider/model still required."""
    report = mutable_matrix.parent / "qa-baseline-match.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    model = payload["model"]
    assert isinstance(model, dict)
    model["provider"] = "other-provider"
    report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    output = tmp_path / "promoted.yaml"
    with pytest.raises(MatrixValidationError) as exc:
        promote_baseline(
            matrix_path=mutable_matrix,
            cell_id="mock-small-qa-smoke",
            run_report_path=report,
            output_path=output,
        )
    assert exc.value.code == "CELL_IDENTITY_MISMATCH"
    assert "does not match matrix" in str(exc.value)


def test_baseline_promote_null_model_digest_skips_digest_only_check(
    mutable_matrix: Path, tmp_path: Path
) -> None:
    """With matrix digest null, a non-matching report model_digest still promotes."""
    report = mutable_matrix.parent / "qa-baseline-match.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["model_digest"] = "totally-different-digest"
    model = payload["model"]
    assert isinstance(model, dict)
    model["resolved_version"] = "totally-different-digest"
    report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    matrix = _read_yaml(mutable_matrix)
    models = matrix["models"]
    assert isinstance(models, list)
    assert isinstance(models[0], dict)
    assert models[0].get("digest") is None
    output = tmp_path / "promoted.yaml"
    baseline = promote_baseline(
        matrix_path=mutable_matrix,
        cell_id="mock-small-qa-smoke",
        run_report_path=report,
        output_path=output,
        name="null-digest-ok",
    )
    assert baseline.pinned_cells[0].model_digest == "totally-different-digest"


def test_baseline_promote_absolute_run_report_round_trip(
    mutable_matrix: Path,
    tmp_path: Path,
) -> None:
    """Promote may pin an absolute report path when it lives outside the output dir."""
    report = tmp_path / "external" / "qa-baseline-match.json"
    report.parent.mkdir(parents=True)
    shutil.copyfile(mutable_matrix.parent / "qa-baseline-match.json", report)
    output = tmp_path / "baselines" / "promoted.yaml"
    loaded_matrix = load_matrix(mutable_matrix)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert isinstance(report_payload, dict)

    baseline = promote_baseline(
        matrix_path=mutable_matrix,
        cell_id="mock-small-qa-smoke",
        run_report_path=report,
        output_path=output,
        name="abs-promoted",
    )
    assert baseline.pinned_cells[0].run_report_path == str(report.resolve())
    assert baseline.pinned_cells[0].config_sha256 == report_payload["config_sha256"]
    assert baseline.matrix_digest == loaded_matrix.matrix_digest

    result = CLI.invoke(
        app,
        [
            "baseline",
            "promote",
            "--matrix",
            str(mutable_matrix),
            "--cell",
            "mock-small-qa-smoke",
            "--run-report",
            str(report),
            "--output",
            str(output),
            "--name",
            "abs-promoted-cli",
        ],
    )
    assert result.exit_code == 0, result.output
    reloaded = load_baseline(output)
    assert reloaded.manifest.name == "abs-promoted-cli"
    assert reloaded.manifest.matrix_digest == loaded_matrix.matrix_digest
    assert reloaded.manifest.pinned_cells[0].run_report_path == str(report.resolve())
    assert reloaded.manifest.pinned_cells[0].run_report_digest == sha256_hex(report.read_bytes())
    assert reloaded.manifest.pinned_cells[0].config_sha256 == report_payload["config_sha256"]


@pytest.mark.parametrize(
    "declared",
    ["../outside-prompt.txt", "nested/../../outside-prompt.txt"],
)
def test_matrix_refuses_artifacts_outside_the_manifest_directory(
    mutable_matrix: Path,
    declared: str,
) -> None:
    outside = mutable_matrix.parent.parent / "outside-prompt.txt"
    shutil.copyfile(mutable_matrix.parent / "prompt.txt", outside)
    manifest = _read_yaml(mutable_matrix)
    prompts = manifest["prompts"]
    assert isinstance(prompts, list)
    assert isinstance(prompts[0], dict)
    prompts[0]["path"] = declared
    prompts[0]["digest"] = sha256_hex(outside.read_bytes())
    _write_yaml(mutable_matrix, manifest)

    with pytest.raises(MatrixValidationError) as exc:
        load_matrix(mutable_matrix)

    assert exc.value.code == "ARTIFACT_OUTSIDE_MATRIX"


@pytest.mark.parametrize(
    "declared",
    ["../outside-run.json", "nested/../../outside-run.json"],
)
def test_baseline_refuses_artifacts_outside_the_manifest_directory(
    mutable_matrix: Path,
    declared: str,
) -> None:
    outside = mutable_matrix.parent.parent / "outside-run.json"
    shutil.copyfile(mutable_matrix.parent / "qa-baseline.json", outside)
    baseline_path = mutable_matrix.parent / "baseline.yaml"
    manifest = _read_yaml(baseline_path)
    cells = manifest["pinned_cells"]
    assert isinstance(cells, list)
    assert isinstance(cells[0], dict)
    cells[0]["run_report_path"] = declared
    cells[0]["run_report_digest"] = sha256_hex(outside.read_bytes())
    _write_yaml(baseline_path, manifest)

    with pytest.raises(MatrixValidationError) as exc:
        load_baseline(baseline_path)

    assert exc.value.code == "ARTIFACT_OUTSIDE_BASELINE"


def test_baseline_promote_cross_directory_absolute_pin_round_trip(
    mutable_matrix: Path, tmp_path: Path
) -> None:
    """Promote to a directory outside the report tree must still validate."""
    report = mutable_matrix.parent / "qa-baseline-match.json"
    output = tmp_path / "outside" / "baseline.yaml"
    promote_baseline(
        matrix_path=mutable_matrix,
        cell_id="mock-small-qa-smoke",
        run_report_path=report,
        output_path=output,
        name="cross-dir",
    )
    pinned = _read_yaml(output)["pinned_cells"]
    assert isinstance(pinned, list)
    assert isinstance(pinned[0], dict)
    assert Path(str(pinned[0]["run_report_path"])).is_absolute()

    loaded = load_baseline(output, matrix=mutable_matrix)
    assert loaded.manifest.name == "cross-dir"
    assert loaded.manifest.pinned_cells[0].run_report_digest == sha256_hex(report.read_bytes())

    result = CLI.invoke(
        app,
        ["baseline", "validate", str(output), "--matrix", str(mutable_matrix)],
    )
    assert result.exit_code == 0, result.output


def test_baseline_rejects_relative_report_outside_baseline_dir(
    mutable_matrix: Path, tmp_path: Path
) -> None:
    report = mutable_matrix.parent / "qa-baseline-match.json"
    output = tmp_path / "baseline.yaml"
    promote_baseline(
        matrix_path=mutable_matrix,
        cell_id="mock-small-qa-smoke",
        run_report_path=report,
        output_path=output,
    )
    manifest = _read_yaml(output)
    cells = manifest["pinned_cells"]
    assert isinstance(cells, list)
    assert isinstance(cells[0], dict)
    cells[0]["run_report_path"] = "../nope.json"
    _write_yaml(output, manifest)
    with pytest.raises(MatrixValidationError, match="ARTIFACT_OUTSIDE_BASELINE"):
        load_baseline(output)
