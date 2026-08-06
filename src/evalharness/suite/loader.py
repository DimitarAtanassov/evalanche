"""Strict, local-only loading for benchmark suite artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import ValidationError

from evalharness.artifacts.calibration import CalibrationArtifact
from evalharness.core.constants import (
    COMPARE_SCHEMA_VERSION,
    OVERALL_SLICE,
    REPORT_SCHEMA_VERSION,
    SUITE_SCHEMA_VERSION,
    SUPPLEMENT_SCHEMA_VERSION,
)
from evalharness.hashing import calibration_body_digest
from evalharness.suite.models import (
    ArtifactReference,
    CompareArtifact,
    JsonValue,
    LoadedCompare,
    LoadedMember,
    LoadedSuite,
    LoadedSupplement,
    RunArtifact,
    SuiteManifest,
)

RUN_SCHEMA_VERSION = REPORT_SCHEMA_VERSION


class SuiteValidationError(ValueError):
    """Caller-actionable failure while validating a suite artifact."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def canonical_json(value: JsonValue) -> str:
    """Serialize JSON deterministically for digests and golden artifacts."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        raise SuiteValidationError("MISSING_ARTIFACT", str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SuiteValidationError("INVALID_ARTIFACT", f"{path}: expected a JSON object")
    return cast(dict[str, JsonValue], value)


def _require_schema(payload: dict[str, JsonValue], expected: str, path: Path) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise SuiteValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{path}: expected {expected}, got {actual!r}",
        )


def _contains_key(value: JsonValue, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _load_manifest(path: Path) -> SuiteManifest:
    if not path.is_file():
        raise SuiteValidationError("MISSING_ARTIFACT", str(path))
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteValidationError("INVALID_MANIFEST", f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SuiteValidationError("INVALID_MANIFEST", "suite.yaml must contain a mapping")
    if payload.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise SuiteValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{path}: expected {SUITE_SCHEMA_VERSION}, got {payload.get('schema_version')!r}",
        )
    try:
        return SuiteManifest.model_validate(payload)
    except ValidationError as exc:
        raise SuiteValidationError("INVALID_MANIFEST", str(exc)) from exc


def _artifact_path(base: Path, declared: str) -> Path:
    """Resolve a declared artifact path, refusing anything outside the manifest tree.

    A manifest is untrusted input, so ``../`` traversal or an absolute path would
    otherwise let it hash and publish files from anywhere on the host.
    """
    resolved = (base / declared).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise SuiteValidationError(
            "ARTIFACT_OUTSIDE_SUITE",
            f"{declared!r} resolves outside the suite directory {base}",
        ) from None
    return resolved


def _require_fields(
    payload: dict[str, JsonValue],
    fields: tuple[str, ...],
    path: Path,
) -> None:
    missing = sorted(field for field in fields if field not in payload)
    if missing:
        raise SuiteValidationError(
            "INVALID_ARTIFACT",
            f"{path}: missing required fields {', '.join(missing)}",
        )


def _validate_calibration(payload: dict[str, JsonValue], path: Path) -> None:
    """Hold suite calibrations to the same contract ``attach_calibration`` enforces.

    Field-presence checks let a self-consistent payload without ``judgment_digest``
    publish a passing gate the attach path would refuse, so the model is the gate.
    """
    try:
        CalibrationArtifact.model_validate(payload)
    except ValidationError as exc:
        raise SuiteValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if payload["calibration_digest"] != calibration_body_digest(payload):
        raise SuiteValidationError(
            "INVALID_ARTIFACT",
            f"{path}: calibration_digest does not match artifact body",
        )


def _validate_supplement(
    payload: dict[str, JsonValue],
    path: Path,
    kind: Literal["calibration", "judge", "rag"],
) -> None:
    _require_schema(payload, SUPPLEMENT_SCHEMA_VERSION, path)
    if kind == "calibration":
        _validate_calibration(payload, path)
        return
    required = {
        "judge": (
            "mode",
            "rubric_name",
            "rubric_version",
            "judge_model",
            "gating_allowed",
            "calibration_digest",
        ),
        "rag": (
            "run_id",
            "model_digest",
            "retrieval",
            "faithfulness",
            "citations",
            "gating_allowed",
        ),
    }[kind]
    _require_fields(payload, required, path)
    if kind == "rag" and payload["gating_allowed"] is not False:
        raise SuiteValidationError(
            "INVALID_ARTIFACT",
            f"{path}: RAG evidence artifacts must remain informational",
        )


def _load_supplements(
    references: list[ArtifactReference],
    base: Path,
    kind: Literal["calibration", "judge", "rag"],
) -> list[LoadedSupplement]:
    loaded: list[LoadedSupplement] = []
    for reference in references:
        artifact_path = _artifact_path(base, reference.path)
        payload = _read_json(artifact_path)
        _validate_supplement(payload, artifact_path, kind)
        loaded.append(
            LoadedSupplement(
                declared_path=reference.path,
                resolved_path=str(artifact_path),
                digest=_digest(payload),
                payload=payload,
            )
        )
    return loaded


def _member_dataset(member: LoadedMember) -> str:
    return member.declaration.dataset or str(member.report.dataset.get("name") or "")


def _validate_manifest_references(manifest: SuiteManifest, members: list[LoadedMember]) -> None:
    paths = [member.declaration.path for member in members]
    if len(paths) != len(set(paths)):
        raise SuiteValidationError("INVALID_MANIFEST", "member run paths must be unique")

    datasets = {_member_dataset(member) for member in members}
    # Assembly ranks every member by its dataset primary metric, so an undeclared
    # dataset must fail here rather than as a KeyError during build.
    undeclared = sorted(datasets - {primary.dataset for primary in manifest.primary_metrics})
    if undeclared:
        raise SuiteValidationError(
            "PRIMARY_METRIC_UNKNOWN",
            f"no primary metric declared for member datasets {', '.join(undeclared)}",
        )
    for primary in manifest.primary_metrics:
        if primary.dataset not in datasets:
            raise SuiteValidationError(
                "PRIMARY_METRIC_UNKNOWN",
                f"dataset {primary.dataset!r} is not declared by a member",
            )
        matching = [member for member in members if _member_dataset(member) == primary.dataset]
        missing = [
            member.report.run_id
            for member in matching
            if not any(
                row.metric == primary.metric and row.slice == OVERALL_SLICE
                for row in member.report.metric_aggregates
            )
        ]
        if missing:
            raise SuiteValidationError(
                "PRIMARY_METRIC_UNKNOWN",
                f"metric {primary.metric!r} missing for runs {', '.join(sorted(missing))}",
            )


def load_suite(path: Path) -> LoadedSuite:
    """Load and strictly validate a suite manifest and its referenced artifacts."""
    manifest_path = path.resolve()
    manifest = _load_manifest(manifest_path)
    base = manifest_path.parent
    members: list[LoadedMember] = []
    for declaration in manifest.member_runs:
        artifact_path = _artifact_path(base, declaration.path)
        payload = _read_json(artifact_path)
        _require_schema(payload, RUN_SCHEMA_VERSION, artifact_path)
        if _contains_key(payload.get("case_examples"), "raw_response"):
            raise SuiteValidationError(
                "INVALID_ARTIFACT",
                f"{artifact_path}: case_examples must not contain raw_response",
            )
        try:
            report = RunArtifact.model_validate(payload)
        except ValidationError as exc:
            raise SuiteValidationError("INVALID_ARTIFACT", f"{artifact_path}: {exc}") from exc
        members.append(
            LoadedMember(
                declaration=declaration,
                resolved_path=str(artifact_path),
                digest=_digest(payload),
                report=report,
            )
        )

    compares: list[LoadedCompare] = []
    run_ids = {member.report.run_id for member in members}
    if len(run_ids) != len(members):
        raise SuiteValidationError("INVALID_MANIFEST", "member run ids must be unique")
    for reference in manifest.compares:
        artifact_path = _artifact_path(base, reference.path)
        payload = _read_json(artifact_path)
        _require_schema(payload, COMPARE_SCHEMA_VERSION, artifact_path)
        try:
            artifact = CompareArtifact.model_validate(payload)
        except ValidationError as exc:
            raise SuiteValidationError("INVALID_ARTIFACT", f"{artifact_path}: {exc}") from exc
        if artifact.baseline_run_id not in run_ids or artifact.candidate_run_id not in run_ids:
            raise SuiteValidationError(
                "INVALID_ARTIFACT",
                f"{artifact_path}: comparison run ids must be declared suite members",
            )
        compares.append(
            LoadedCompare(
                declared_path=reference.path,
                resolved_path=str(artifact_path),
                digest=_digest(payload),
                artifact=artifact,
            )
        )

    _validate_manifest_references(manifest, members)
    return LoadedSuite(
        manifest_path=str(manifest_path),
        manifest=manifest,
        members=members,
        compares=compares,
        calibrations=_load_supplements(manifest.calibrations, base, "calibration"),
        judge_artifacts=_load_supplements(manifest.judge_artifacts, base, "judge"),
        rag_artifacts=_load_supplements(manifest.rag_artifacts, base, "rag"),
    )
