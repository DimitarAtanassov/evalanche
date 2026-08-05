"""Deterministic, offline materialization for pinned dataset snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from evalharness.datasets import DatasetTier, load_dataset, validate_dataset
from evalharness.datasets.validator import (
    ALLOWED_SMOKE_LICENSES,
    FIXTURES_DIR,
    TIER_SIZE_BOUNDS,
)
from tools.datasets.adapters import ADAPTERS, AdapterSpec, CaseRecord, fits_field_bounds


class MaterializationError(ValueError):
    """A stable, caller-actionable materialization failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class SourcePin:
    """Operator-reviewed identity for an external local snapshot."""

    revision: str
    revision_digest: str
    canonical_url: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_external_pin(source: Path, spec: AdapterSpec) -> SourcePin:
    pin_path = source.with_name(f"{source.name}.pin.yaml")
    if not pin_path.is_file():
        raise MaterializationError(
            "SOURCE_PIN_REQUIRED",
            f"external adapter requires {pin_path.name}",
        )
    raw_value = yaml.safe_load(pin_path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise MaterializationError("SOURCE_PIN_INVALID", "pin root must be a mapping")
    raw = cast(dict[str, object], raw_value)
    expected = {"revision", "revision_digest", "canonical_url"}
    if set(raw) != expected or any(not isinstance(raw[key], str) for key in expected):
        raise MaterializationError(
            "SOURCE_PIN_INVALID",
            "pin requires only revision, revision_digest, and canonical_url strings",
        )
    revision = cast(str, raw["revision"])
    if spec.source_revision != "operator-pinned" and revision != spec.source_revision:
        raise MaterializationError(
            "SOURCE_REVISION_MISMATCH",
            f"expected {spec.source_revision}, got {revision}",
        )
    canonical_url = cast(str, raw["canonical_url"])
    if spec.canonical_url is not None and canonical_url != spec.canonical_url:
        raise MaterializationError(
            "SOURCE_URL_MISMATCH",
            f"expected {spec.canonical_url}, got {canonical_url}",
        )
    return SourcePin(
        revision=revision,
        revision_digest=cast(str, raw["revision_digest"]),
        canonical_url=canonical_url,
    )


def _source_pin(source: Path, spec: AdapterSpec) -> SourcePin:
    source_digest = f"sha256:{_sha256_bytes(source.read_bytes())}"
    if not spec.requires_external_pin:
        return SourcePin(
            revision=spec.source_revision,
            revision_digest=source_digest,
            canonical_url=f"repo://tools/datasets/sources/{source.name}",
        )
    pin = _load_external_pin(source, spec)
    if pin.revision_digest != source_digest:
        raise MaterializationError(
            "SOURCE_DIGEST_MISMATCH",
            f"expected {pin.revision_digest}, got {source_digest}",
        )
    if "://" not in pin.canonical_url:
        raise MaterializationError("SOURCE_PIN_INVALID", "canonical_url must be absolute")
    return pin


def _committed_fixture_root(output: Path) -> Path | None:
    """Root above the ``fixtures/`` tree an output would land in, if any.

    Keyed off the path itself rather than ``.git`` discovery: a bare checkout, an
    unpacked tarball, or a vendored copy carries the same redistribution obligations as
    a git working tree.
    """
    parts = output.resolve().parts
    for index, part in enumerate(parts):
        if part == FIXTURES_DIR:
            return Path(*parts[:index])
    return None


def _enforce_license(spec: AdapterSpec, output: Path) -> None:
    tree_root = _committed_fixture_root(output)
    if tree_root is None:
        return
    if not spec.redistributable_smoke or spec.license not in ALLOWED_SMOKE_LICENSES:
        raise MaterializationError(
            "LICENSE_BLOCK",
            f"{spec.source_id} may only be materialized outside tracked fixtures",
        )
    datasets_doc = tree_root / "docs" / "datasets.md"
    if not datasets_doc.is_file() or spec.source_id not in datasets_doc.read_text(encoding="utf-8"):
        raise MaterializationError(
            "LICENSE_BLOCK",
            f"docs/datasets.md has no card for {spec.source_id}",
        )


def _validate_requested_size(tier: DatasetTier, size: int) -> None:
    minimum, maximum = TIER_SIZE_BOUNDS[tier]
    if size < minimum or (maximum is not None and size > maximum):
        upper = str(maximum) if maximum is not None else "unbounded"
        raise MaterializationError(
            "TIER_SIZE_INVALID",
            f"{tier.value} requires size {minimum}..{upper}; got {size}",
        )


def _select_cases(spec: AdapterSpec, source: Path, *, seed: int, size: int) -> list[CaseRecord]:
    """Parse, drop records that exceed the published field bounds, then sample."""
    try:
        parsed = spec.parser(source)
    except (csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise MaterializationError("SOURCE_FORMAT_INVALID", str(exc)) from exc
    publishable = [record for record in parsed if fits_field_bounds(record)]
    if len(publishable) < size:
        raise MaterializationError(
            "SOURCE_TOO_SMALL",
            f"requested {size} records; {len(publishable)} of {len(parsed)} source records "
            "fit the published field bounds",
        )
    return _sample(publishable, seed, size)


def _sample(records: list[CaseRecord], seed: int, size: int) -> list[CaseRecord]:
    by_id: dict[str, CaseRecord] = {}
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise MaterializationError("CASE_VALIDATION_FAILED", "every source record needs an id")
        if record_id in by_id:
            raise MaterializationError("CASE_VALIDATION_FAILED", f"duplicate source id {record_id}")
        by_id[record_id] = record
    if len(by_id) < size:
        raise MaterializationError(
            "SOURCE_TOO_SMALL",
            f"requested {size} records from source containing {len(by_id)}",
        )

    def sample_key(item: tuple[str, CaseRecord]) -> tuple[str, str]:
        record_id, _ = item
        digest = _sha256_bytes(f"{seed}:{record_id}".encode())
        return digest, record_id

    selected = sorted(by_id.items(), key=sample_key)[:size]
    return [record for _, record in sorted(selected)]


def _case_with_provenance(record: CaseRecord, spec: AdapterSpec, pin: SourcePin) -> CaseRecord:
    record_id = cast(str, record["id"])
    provenance_value = record.get("provenance", {})
    provenance = (
        dict(cast(dict[str, Any], provenance_value)) if isinstance(provenance_value, dict) else {}
    )
    provenance.update(
        {
            "source_id": spec.source_id,
            "source_record_id": record_id,
            "source_revision": pin.revision,
            "adapter_name": spec.name,
            "adapter_version": spec.version,
        }
    )
    return {**record, "provenance": provenance}


def _serialize_cases(records: list[CaseRecord]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _manifest(
    spec: AdapterSpec,
    pin: SourcePin,
    *,
    seed: int,
    size: int,
    tier: DatasetTier,
    content_sha256: str,
) -> dict[str, object]:
    split = "holdout" if tier is DatasetTier.RELEASE else "dev"
    return {
        "schema_version": "0.1",
        "name": spec.dataset_name,
        "version": spec.dataset_version,
        "split": split,
        "license": spec.license,
        "pii_scrubbed": spec.pii_scrubbed,
        "pii_scrub_procedure": spec.pii_scrub_procedure,
        "created_at": spec.created_at,
        "slices": list(spec.slices),
        "content_sha256": content_sha256,
        "tier": tier.value,
        "source": {
            "id": spec.source_id,
            "revision": pin.revision,
            "revision_digest": pin.revision_digest,
            "canonical_url": pin.canonical_url,
            "redistributable_smoke": spec.redistributable_smoke,
            "attribution": spec.attribution,
        },
        "adapter": {
            "name": spec.name,
            "version": spec.version,
            "sample_seed": seed,
            "sample_size": size,
        },
        "task_metrics": list(spec.task_metrics),
        "contamination_risk": spec.contamination_risk,
    }


def _write_bundle(
    output: Path,
    manifest: dict[str, object],
    cases_bytes: bytes,
    *,
    allow_holdout: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temp_path / "cases.jsonl").write_bytes(cases_bytes)
        (temp_path / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        report = validate_dataset(load_dataset(temp_path), allow_holdout=allow_holdout)
        if not report.valid:
            raise MaterializationError("CASE_VALIDATION_FAILED", "; ".join(report.errors))
        try:
            os.rename(temp_path, output)
        except FileExistsError as exc:
            raise MaterializationError("OUTPUT_EXISTS", str(output)) from exc
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


def materialize_dataset(
    *,
    adapter_name: str,
    source: Path,
    output: Path,
    seed: int,
    size: int,
    tier: DatasetTier,
    check_deterministic: bool = False,
) -> None:
    """Materialize one pinned local source without network or runtime downloads."""
    try:
        spec = ADAPTERS[adapter_name]
    except KeyError as exc:
        raise MaterializationError("UNKNOWN_ADAPTER", adapter_name) from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise MaterializationError("OUTPUT_EXISTS", str(output))
    _validate_requested_size(tier, size)
    _enforce_license(spec, output)
    pin = _source_pin(source, spec)
    selected = _select_cases(spec, source, seed=seed, size=size)
    cases = [_case_with_provenance(record, spec, pin) for record in selected]
    cases_bytes = _serialize_cases(cases)
    if check_deterministic:
        repeated = _select_cases(spec, source, seed=seed, size=size)
        repeated_bytes = _serialize_cases(
            [_case_with_provenance(record, spec, pin) for record in repeated]
        )
        if repeated_bytes != cases_bytes:
            raise MaterializationError(
                "NONDETERMINISTIC_ADAPTER",
                f"{adapter_name} emitted different bytes on repeated transform",
            )
    content_sha256 = _sha256_bytes(cases_bytes.rstrip(b"\n"))
    manifest = _manifest(
        spec,
        pin,
        seed=seed,
        size=size,
        tier=tier,
        content_sha256=content_sha256,
    )
    _write_bundle(
        output,
        manifest,
        cases_bytes,
        allow_holdout=tier is DatasetTier.RELEASE,
    )
