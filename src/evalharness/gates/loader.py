"""Strict, local-only loading for gates.yaml and bound artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from evalharness.artifacts.calibration import CalibrationArtifact
from evalharness.domain.constants import (
    COMPARE_SCHEMA_VERSION,
    GATES_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    SUPPLEMENT_SCHEMA_VERSION,
)
from evalharness.gates.errors import GatesValidationError
from evalharness.gates.models import (
    ArtifactOverrides,
    GatesManifest,
    JsonValue,
    LoadedGates,
)
from evalharness.hashing import calibration_body_digest
from evalharness.path_jail import resolve_jailed_path
from evalharness.suite.models import CompareArtifact, RunArtifact

_KINDS_NEEDING_RUN = frozenset(
    {"coverage", "harness_failure_rate", "quality_floor", "latency", "cost"}
)


def _read_json(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        raise GatesValidationError("MISSING_ARTIFACT", str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatesValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GatesValidationError("INVALID_ARTIFACT", f"{path}: expected a JSON object")
    return cast(dict[str, JsonValue], value)


def _require_schema(payload: dict[str, JsonValue], expected: str, path: Path) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise GatesValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{path}: expected {expected}, got {actual!r}",
        )


def _artifact_path(base: Path, declared: str) -> Path:
    """Resolve a declared path, refusing anything outside the gates.yaml tree."""
    try:
        return resolve_jailed_path(base, declared)
    except ValueError:
        raise GatesValidationError(
            "ARTIFACT_OUTSIDE_GATES",
            f"{declared!r} resolves outside the gates directory {base}",
        ) from None


def _override_path(declared: str) -> Path:
    """CLI overrides may be absolute or cwd-relative; they skip the tree check."""
    path = Path(declared).expanduser()
    path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if not path.is_file():
        raise GatesValidationError("MISSING_ARTIFACT", str(path))
    return path


def _load_manifest(path: Path) -> GatesManifest:
    if not path.is_file():
        raise GatesValidationError("MISSING_ARTIFACT", str(path))
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GatesValidationError("INVALID_MANIFEST", f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GatesValidationError("INVALID_MANIFEST", "gates.yaml must contain a mapping")
    if payload.get("schema_version") != GATES_SCHEMA_VERSION:
        raise GatesValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{path}: expected {GATES_SCHEMA_VERSION}, got {payload.get('schema_version')!r}",
        )
    try:
        return GatesManifest.model_validate(payload)
    except ValidationError as exc:
        raise GatesValidationError("INVALID_MANIFEST", str(exc)) from exc


def _load_run_report(path: Path) -> RunArtifact:
    payload = _read_json(path)
    _require_schema(payload, REPORT_SCHEMA_VERSION, path)
    try:
        return RunArtifact.model_validate(payload)
    except ValidationError as exc:
        raise GatesValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc


def _load_compare(path: Path) -> CompareArtifact:
    payload = _read_json(path)
    _require_schema(payload, COMPARE_SCHEMA_VERSION, path)
    try:
        return CompareArtifact.model_validate(payload)
    except ValidationError as exc:
        raise GatesValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc


def _load_calibration(path: Path) -> CalibrationArtifact:
    payload = _read_json(path)
    _require_schema(payload, SUPPLEMENT_SCHEMA_VERSION, path)
    try:
        artifact = CalibrationArtifact.model_validate(payload)
    except ValidationError as exc:
        raise GatesValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if payload["calibration_digest"] != calibration_body_digest(payload):
        raise GatesValidationError(
            "INVALID_ARTIFACT",
            f"{path}: calibration_digest does not match artifact body",
        )
    return artifact


def _resolve_bound(
    *,
    declared: str | None,
    override: str | None,
    base: Path,
) -> Path | None:
    if override is not None:
        return _override_path(override)
    if declared is None:
        return None
    return _artifact_path(base, declared)


def _assert_bindings(
    manifest: GatesManifest,
    *,
    run_report: Path | None,
    compare: Path | None,
    calibration: Path | None,
) -> None:
    for gate in manifest.gates:
        if gate.kind in _KINDS_NEEDING_RUN and run_report is None:
            raise GatesValidationError(
                "MISSING_ARTIFACT",
                f"gate {gate.name!r} requires a run_report artifact",
            )
        if gate.kind == "paired_regression" and compare is None:
            raise GatesValidationError(
                "MISSING_ARTIFACT",
                f"gate {gate.name!r} requires a compare artifact",
            )
        if gate.kind == "calibrated_judge" and calibration is None:
            raise GatesValidationError(
                "MISSING_ARTIFACT",
                f"gate {gate.name!r} requires a calibration artifact",
            )


def load_gates(
    path: Path,
    *,
    overrides: ArtifactOverrides | None = None,
) -> LoadedGates:
    """Load and strictly validate a gates manifest and its bound artifacts."""
    manifest_path = path.resolve()
    manifest = _load_manifest(manifest_path)
    base = manifest_path.parent
    overrides = overrides or ArtifactOverrides()

    run_path = _resolve_bound(
        declared=manifest.artifacts.run_report,
        override=overrides.run_report,
        base=base,
    )
    compare_path = _resolve_bound(
        declared=manifest.artifacts.compare,
        override=overrides.compare,
        base=base,
    )
    calibration_path = _resolve_bound(
        declared=manifest.artifacts.calibration,
        override=overrides.calibration,
        base=base,
    )
    _assert_bindings(
        manifest,
        run_report=run_path,
        compare=compare_path,
        calibration=calibration_path,
    )

    return LoadedGates(
        manifest_path=str(manifest_path),
        manifest=manifest,
        run_report=_load_run_report(run_path) if run_path else None,
        run_report_path=str(run_path) if run_path else None,
        compare=_load_compare(compare_path) if compare_path else None,
        compare_path=str(compare_path) if compare_path else None,
        calibration=_load_calibration(calibration_path) if calibration_path else None,
        calibration_path=str(calibration_path) if calibration_path else None,
    )
