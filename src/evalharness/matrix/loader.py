"""Strict loading and digesting for matrix.yaml and baseline.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from evalharness.core.constants import (
    BASELINE_SCHEMA_VERSION,
    MATRIX_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
)
from evalharness.hashing import canonical_json, sha256_hex
from evalharness.matrix.errors import MatrixValidationError
from evalharness.matrix.models import (
    BaselineManifest,
    JsonValue,
    LoadedBaseline,
    LoadedMatrix,
    MatrixManifest,
    PinnedCell,
)
from evalharness.path_jail import resolve_jailed_path
from evalharness.suite.models import RunArtifact


def _read_yaml(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        raise MatrixValidationError("MISSING_ARTIFACT", str(path))
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MatrixValidationError("INVALID_MANIFEST", f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatrixValidationError("INVALID_MANIFEST", f"{path}: expected a mapping")
    return cast(dict[str, JsonValue], payload)


def _read_json(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        raise MatrixValidationError("MISSING_ARTIFACT", str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixValidationError("INVALID_ARTIFACT", f"{path}: expected a JSON object")
    return cast(dict[str, JsonValue], value)


def _artifact_path(base: Path, declared: str) -> Path:
    """Resolve a matrix-declared path, refusing anything outside the matrix tree."""
    try:
        return resolve_jailed_path(base, declared)
    except ValueError:
        raise MatrixValidationError(
            "ARTIFACT_OUTSIDE_MATRIX",
            f"{declared!r} resolves outside the matrix directory {base}",
        ) from None


def _baseline_report_path(base: Path, declared: str) -> Path:
    """Resolve a baseline run_report_path.

    Absolute paths are trusted explicit operator pins (promote may write them when
    the report is outside the baseline.yaml directory). Relative paths stay jailed
    under the baseline directory, same contract as suite manifests.
    """
    path = Path(declared)
    if path.is_absolute():
        return path.resolve()
    try:
        return resolve_jailed_path(base, declared)
    except ValueError:
        raise MatrixValidationError(
            "ARTIFACT_OUTSIDE_BASELINE",
            f"{declared!r} resolves outside the baseline directory {base}",
        ) from None


def _file_digest(path: Path) -> str:
    try:
        return sha256_hex(path.read_bytes())
    except OSError as exc:
        raise MatrixValidationError("INVALID_ARTIFACT", f"{path}: {exc}") from exc


def matrix_digest(manifest: MatrixManifest) -> str:
    """Digest the validated manifest body (declared relative paths, not resolved)."""
    body = manifest.model_dump(mode="json")
    return sha256_hex(canonical_json(body))


def _verify_path_digest(path: Path, expected: str | None, label: str) -> None:
    if expected is None:
        return
    actual = _file_digest(path)
    if actual != expected:
        raise MatrixValidationError(
            "DIGEST_MISMATCH",
            f"{label} {path}: expected digest {expected}, got {actual}",
        )


def _verify_cell_identity(
    *,
    loaded_matrix: LoadedMatrix,
    cell_id: str,
    report: RunArtifact,
) -> None:
    """Fail closed when the report does not match the matrix cell's pinned identity."""
    cell = next(item for item in loaded_matrix.manifest.cells if item.id == cell_id)
    model = next(item for item in loaded_matrix.manifest.models if item.id == cell.model)
    prompt = next(item for item in loaded_matrix.manifest.prompts if item.id == cell.prompt)
    dataset = next(item for item in loaded_matrix.manifest.datasets if item.id == cell.dataset)

    report_provider = report.model.get("provider")
    report_model = report.model.get("model")
    if report_provider != model.provider or report_model != model.model:
        raise MatrixValidationError(
            "CELL_IDENTITY_MISMATCH",
            f"cell {cell_id!r}: report model {report_provider!r}/{report_model!r} "
            f"does not match matrix {model.provider!r}/{model.model!r}",
        )
    if model.digest is not None and report.model_digest != model.digest:
        raise MatrixValidationError(
            "CELL_IDENTITY_MISMATCH",
            f"cell {cell_id!r}: report model_digest {report.model_digest} "
            f"does not match matrix model digest {model.digest}",
        )
    if dataset.digest is not None and report.dataset_sha256 != dataset.digest:
        raise MatrixValidationError(
            "CELL_IDENTITY_MISMATCH",
            f"cell {cell_id!r}: report dataset_sha256 {report.dataset_sha256} "
            f"does not match matrix dataset digest {dataset.digest}",
        )
    prompt_sha = report.prompt_template.get("content_sha256")
    if prompt.digest is not None and prompt_sha != prompt.digest:
        raise MatrixValidationError(
            "CELL_IDENTITY_MISMATCH",
            f"cell {cell_id!r}: report prompt_template.content_sha256 {prompt_sha!r} "
            f"does not match matrix prompt digest {prompt.digest}",
        )


def load_matrix(path: Path) -> LoadedMatrix:
    """Load and strictly validate a matrix manifest and referenced local files."""
    manifest_path = path.resolve()
    payload = _read_yaml(manifest_path)
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise MatrixValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{manifest_path}: expected {MATRIX_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}",
        )
    try:
        manifest = MatrixManifest.model_validate(payload)
    except ValidationError as exc:
        raise MatrixValidationError("INVALID_MANIFEST", str(exc)) from exc

    base = manifest_path.parent
    for prompt in manifest.prompts:
        prompt_path = _artifact_path(base, prompt.path)
        if not prompt_path.is_file():
            raise MatrixValidationError("MISSING_ARTIFACT", str(prompt_path))
        _verify_path_digest(prompt_path, prompt.digest, f"prompt {prompt.id!r}")
    for dataset in manifest.datasets:
        dataset_path = _artifact_path(base, dataset.path)
        if not dataset_path.is_file() and not dataset_path.is_dir():
            raise MatrixValidationError("MISSING_ARTIFACT", str(dataset_path))
        if dataset.digest is not None:
            if dataset_path.is_dir():
                raise MatrixValidationError(
                    "DIGEST_MISMATCH",
                    f"dataset {dataset.id!r}: digest pins require a file path, not a directory",
                )
            _verify_path_digest(dataset_path, dataset.digest, f"dataset {dataset.id!r}")

    return LoadedMatrix(
        manifest_path=str(manifest_path),
        manifest=manifest,
        matrix_digest=matrix_digest(manifest),
    )


def load_baseline(path: Path, *, matrix: Path | None = None) -> LoadedBaseline:
    """Load a baseline.yaml; optionally verify against a matrix manifest."""
    manifest_path = path.resolve()
    payload = _read_yaml(manifest_path)
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise MatrixValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{manifest_path}: expected {BASELINE_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}",
        )
    try:
        manifest = BaselineManifest.model_validate(payload)
    except ValidationError as exc:
        raise MatrixValidationError("INVALID_MANIFEST", str(exc)) from exc

    base = manifest_path.parent
    for cell in manifest.pinned_cells:
        report_path = _baseline_report_path(base, cell.run_report_path)
        if not report_path.is_file():
            raise MatrixValidationError("MISSING_ARTIFACT", str(report_path))
        actual = _file_digest(report_path)
        if actual != cell.run_report_digest:
            raise MatrixValidationError(
                "DIGEST_MISMATCH",
                f"cell {cell.cell_id!r}: run report digest mismatch "
                f"(expected {cell.run_report_digest}, got {actual})",
            )

    if matrix is not None:
        loaded_matrix = load_matrix(matrix)
        if manifest.matrix_name != loaded_matrix.manifest.name:
            raise MatrixValidationError(
                "MATRIX_MISMATCH",
                f"baseline matrix_name {manifest.matrix_name!r} does not match "
                f"matrix name {loaded_matrix.manifest.name!r}",
            )
        if manifest.matrix_digest != loaded_matrix.matrix_digest:
            raise MatrixValidationError(
                "MATRIX_MISMATCH",
                f"baseline matrix_digest {manifest.matrix_digest} does not match "
                f"matrix digest {loaded_matrix.matrix_digest}",
            )
        known_cells = {cell.id for cell in loaded_matrix.manifest.cells}
        unknown = sorted(
            cell.cell_id for cell in manifest.pinned_cells if cell.cell_id not in known_cells
        )
        if unknown:
            raise MatrixValidationError(
                "UNKNOWN_CELL",
                f"baseline pins unknown cell ids: {', '.join(unknown)}",
            )

    return LoadedBaseline(manifest_path=str(manifest_path), manifest=manifest)


def promote_baseline(
    *,
    matrix_path: Path,
    cell_id: str,
    run_report_path: Path,
    output_path: Path,
    name: str | None = None,
    allow_mismatch: bool = False,
) -> BaselineManifest:
    """Pin a run report into a baseline.yaml cell; never invents a latest selector."""
    loaded_matrix = load_matrix(matrix_path)
    known = {cell.id for cell in loaded_matrix.manifest.cells}
    if cell_id not in known:
        raise MatrixValidationError(
            "UNKNOWN_CELL",
            f"cell_id {cell_id!r} is not declared in matrix {loaded_matrix.manifest.name!r}",
        )

    report_path = run_report_path.resolve()
    if not report_path.is_file():
        raise MatrixValidationError("MISSING_ARTIFACT", str(report_path))
    payload = _read_json(report_path)
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise MatrixValidationError(
            "UNSUPPORTED_SCHEMA",
            f"{report_path}: expected {REPORT_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}",
        )
    try:
        report = RunArtifact.model_validate(payload)
    except ValidationError as exc:
        raise MatrixValidationError("INVALID_ARTIFACT", f"{report_path}: {exc}") from exc
    if not allow_mismatch:
        _verify_cell_identity(
            loaded_matrix=loaded_matrix,
            cell_id=cell_id,
            report=report,
        )
    config_sha256 = payload.get("config_sha256")
    if not isinstance(config_sha256, str) or not config_sha256:
        raise MatrixValidationError(
            "INVALID_ARTIFACT",
            f"{report_path}: missing config_sha256",
        )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_report = str(report_path.relative_to(output_path.parent))
    except ValueError:
        relative_report = str(report_path)

    pinned = PinnedCell(
        cell_id=cell_id,
        run_id=report.run_id,
        run_report_path=relative_report,
        run_report_digest=_file_digest(report_path),
        config_sha256=config_sha256,
        model_digest=report.model_digest,
    )

    if output_path.is_file():
        existing = load_baseline(output_path).manifest
        cells = [cell for cell in existing.pinned_cells if cell.cell_id != cell_id]
        cells.append(pinned)
        baseline = BaselineManifest(
            schema_version=BASELINE_SCHEMA_VERSION,
            name=name or existing.name,
            matrix_name=loaded_matrix.manifest.name,
            matrix_digest=loaded_matrix.matrix_digest,
            pinned_cells=sorted(cells, key=lambda item: item.cell_id),
        )
    else:
        baseline = BaselineManifest(
            schema_version=BASELINE_SCHEMA_VERSION,
            name=name or f"{loaded_matrix.manifest.name}-baseline",
            matrix_name=loaded_matrix.manifest.name,
            matrix_digest=loaded_matrix.matrix_digest,
            pinned_cells=[pinned],
        )

    dumped = baseline.model_dump(mode="json")
    output_path.write_text(
        yaml.safe_dump(dumped, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return baseline
