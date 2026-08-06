"""Dataset manifest and JSONL loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml

from evalharness.core.enums import TaskType
from evalharness.core.models import Case
from evalharness.hashing import sha256_hex


class DatasetTier(StrEnum):
    """Supported dataset materialization sizes."""

    SMOKE = "smoke"
    CI = "ci"
    NIGHTLY = "nightly"
    RELEASE = "release"


class ContaminationRisk(StrEnum):
    """Declared likelihood that benchmark content appears in model training data."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Pinned source identity for a materialized dataset."""

    id: str
    revision: str
    revision_digest: str
    canonical_url: str
    redistributable_smoke: bool
    attribution: str


@dataclass(frozen=True, slots=True)
class DatasetAdapter:
    """Adapter identity and deterministic sampling configuration."""

    name: str
    version: str
    sample_seed: int
    sample_size: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    version: str
    split: str
    license: str
    pii_scrubbed: bool
    created_at: str
    slices: list[str]
    content_sha256: str | None = None
    schema_version: str | None = None
    tier: DatasetTier | None = None
    source: DatasetSource | None = None
    adapter: DatasetAdapter | None = None
    task_metrics: list[str] | None = None
    contamination_risk: ContaminationRisk | None = None
    pii_scrub_procedure: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    manifest: DatasetManifest
    cases: list[Case]
    content_sha256: str
    source_path: Path
    manifest_warnings: tuple[str, ...] = ()


class DatasetUpsertFields(TypedDict):
    """Domain values a persistence seam needs to record a loaded dataset."""

    name: str
    version: str
    split: str
    content_sha256: str
    license: str
    pii_scrubbed: bool
    created_at: str
    slices: list[str]
    cases: list[Case]


def dataset_upsert_fields(bundle: DatasetBundle) -> DatasetUpsertFields:
    """Flatten a bundle for storage so the store layer never sees the loader type."""
    return {
        "name": bundle.manifest.name,
        "version": bundle.manifest.version,
        "split": bundle.manifest.split,
        "content_sha256": bundle.content_sha256,
        "license": bundle.manifest.license,
        "pii_scrubbed": bundle.manifest.pii_scrubbed,
        "created_at": bundle.manifest.created_at,
        "slices": bundle.manifest.slices,
        "cases": bundle.cases,
    }


class DatasetManifestError(ValueError):
    """Raised when a dataset manifest violates its typed boundary contract."""


class DatasetCaseError(ValueError):
    """Raised when a JSONL case violates the dataset boundary contract."""


LEGACY_MANIFEST_KEYS = frozenset(
    {
        "name",
        "version",
        "split",
        "license",
        "pii_scrubbed",
        "created_at",
        "slices",
        "content_sha256",
    }
)
VERSIONED_MANIFEST_KEYS = LEGACY_MANIFEST_KEYS | {
    "schema_version",
    "tier",
    "source",
    "adapter",
    "task_metrics",
    "contamination_risk",
    "pii_scrub_procedure",
}
REQUIRED_MANIFEST_KEYS = LEGACY_MANIFEST_KEYS - {"content_sha256"}
VERSIONED_REQUIRED_KEYS = VERSIONED_MANIFEST_KEYS - {"content_sha256"}


TASK_REQUIRED_FIELDS: dict[TaskType, list[str]] = {
    TaskType.GENERATION: [],
    TaskType.CLASSIFICATION: ["expected_label"],
    TaskType.EXTRACTION: ["expected_json"],
    TaskType.SUMMARIZATION: ["reference_answer"],
    TaskType.QA_SHORT: ["reference_answer"],
    TaskType.RETRIEVAL: ["qrels"],
    TaskType.RAG: ["reference_answer", "qrels"],
    TaskType.TOOL_USE: ["expected_json"],
    TaskType.AGENT_TRAJECTORY: [],
    TaskType.SAFETY: [],
    TaskType.PAIRWISE: [],
}


def _normalize_prompt(inputs: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(inputs.keys()):
        value = inputs[key]
        if isinstance(value, str):
            parts.append(f"{key}:{value.strip().lower()}")
        elif isinstance(value, list):
            parts.append(f"{key}:{'|'.join(str(v).strip().lower() for v in value)}")
        else:
            parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
    return " ".join(parts)


def _case_string(raw: dict[str, Any], key: str, *, required: bool = False) -> str | None:
    value = raw.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value):
        qualifier = "non-empty string" if required else "string or null"
        raise DatasetCaseError(f"Case field '{key}' must be a {qualifier}")
    return value


def _case_string_list(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetCaseError(f"Case field '{key}' must be a list of strings")
    return cast(list[str], value)


def _case_string_mapping(raw: dict[str, Any], key: str) -> dict[str, str]:
    value = raw.get(key, {})
    if not isinstance(value, dict) or any(
        not isinstance(item_key, str) or not isinstance(item_value, str)
        for item_key, item_value in value.items()
    ):
        raise DatasetCaseError(f"Case field '{key}' must map strings to strings")
    return cast(dict[str, str], value)


def _parse_case(raw_value: object) -> Case:
    if not isinstance(raw_value, dict) or any(not isinstance(key, str) for key in raw_value):
        raise DatasetCaseError("Case must be an object with string keys")
    raw = cast(dict[str, Any], raw_value)
    external_id = _case_string(raw, "id", required=True)
    task_type_value = _case_string(raw, "task_type", required=True)
    inputs_value = raw.get("inputs")
    if not isinstance(inputs_value, dict) or any(not isinstance(key, str) for key in inputs_value):
        raise DatasetCaseError("Case field 'inputs' must be an object with string keys")
    expected_json_value = raw.get("expected_json")
    if expected_json_value is not None and not isinstance(expected_json_value, dict):
        raise DatasetCaseError("Case field 'expected_json' must be an object or null")
    qrels_value = raw.get("qrels")
    if qrels_value is not None and (
        not isinstance(qrels_value, dict)
        or any(
            not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int)
            for key, value in qrels_value.items()
        )
    ):
        raise DatasetCaseError("Case field 'qrels' must map document ids to integer relevance")
    provenance_value = raw.get("provenance", {})
    if not isinstance(provenance_value, dict):
        raise DatasetCaseError("Case field 'provenance' must be an object")
    weight_value = raw.get("weight", 1.0)
    if isinstance(weight_value, bool) or not isinstance(weight_value, (int, float)):
        raise DatasetCaseError("Case field 'weight' must be numeric")
    if weight_value <= 0:
        raise DatasetCaseError("Case field 'weight' must be greater than zero")
    try:
        task_type = TaskType(cast(str, task_type_value))
    except ValueError as exc:
        raise DatasetCaseError(f"Unknown task_type: {task_type_value}") from exc
    return Case(
        external_id=cast(str, external_id),
        task_type=task_type,
        inputs=cast(dict[str, Any], inputs_value),
        reference_answer=_case_string(raw, "reference_answer"),
        references=_case_string_list(raw, "references"),
        expected_label=_case_string(raw, "expected_label"),
        expected_json=cast(dict[str, Any] | None, expected_json_value),
        qrels=cast(dict[str, int] | None, qrels_value),
        slices=_case_string_mapping(raw, "slices"),
        must_contain=_case_string_list(raw, "must_contain"),
        must_not_contain=_case_string_list(raw, "must_not_contain"),
        canary=_case_string(raw, "canary"),
        weight=float(weight_value),
        provenance=cast(dict[str, Any], provenance_value),
        normalized_prompt=_normalize_prompt(cast(dict[str, Any], inputs_value)),
    )


def _required(raw: dict[str, object], key: str) -> object:
    try:
        return raw[key]
    except KeyError as exc:
        raise DatasetManifestError(f"Missing required manifest key: {key}") from exc


def _require_type(value: object, expected: type[object], field_name: str) -> object:
    if not isinstance(value, expected):
        raise DatasetManifestError(f"Manifest field '{field_name}' must be {expected.__name__}")
    return value


def _parse_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetManifestError(f"Manifest field '{field_name}' must be a list of strings")
    return cast(list[str], value)


def _parse_source(value: object) -> DatasetSource:
    source = _require_type(value, dict, "source")
    source_raw = cast(dict[str, object], source)
    expected = {
        "id",
        "revision",
        "revision_digest",
        "canonical_url",
        "redistributable_smoke",
        "attribution",
    }
    unknown = set(source_raw) - expected
    missing = expected - set(source_raw)
    if unknown:
        raise DatasetManifestError(f"Unknown source keys: {', '.join(sorted(unknown))}")
    if missing:
        raise DatasetManifestError(f"Missing source keys: {', '.join(sorted(missing))}")
    return DatasetSource(
        id=cast(str, _require_type(source_raw["id"], str, "source.id")),
        revision=cast(str, _require_type(source_raw["revision"], str, "source.revision")),
        revision_digest=cast(
            str,
            _require_type(source_raw["revision_digest"], str, "source.revision_digest"),
        ),
        canonical_url=cast(
            str,
            _require_type(source_raw["canonical_url"], str, "source.canonical_url"),
        ),
        redistributable_smoke=cast(
            bool,
            _require_type(
                source_raw["redistributable_smoke"],
                bool,
                "source.redistributable_smoke",
            ),
        ),
        attribution=cast(
            str,
            _require_type(source_raw["attribution"], str, "source.attribution"),
        ),
    )


def _parse_adapter(value: object) -> DatasetAdapter:
    adapter = _require_type(value, dict, "adapter")
    adapter_raw = cast(dict[str, object], adapter)
    expected = {"name", "version", "sample_seed", "sample_size"}
    unknown = set(adapter_raw) - expected
    missing = expected - set(adapter_raw)
    if unknown:
        raise DatasetManifestError(f"Unknown adapter keys: {', '.join(sorted(unknown))}")
    if missing:
        raise DatasetManifestError(f"Missing adapter keys: {', '.join(sorted(missing))}")
    seed = adapter_raw["sample_seed"]
    size = adapter_raw["sample_size"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetManifestError("Manifest field 'adapter.sample_seed' must be int")
    if isinstance(size, bool) or not isinstance(size, int):
        raise DatasetManifestError("Manifest field 'adapter.sample_size' must be int")
    return DatasetAdapter(
        name=cast(str, _require_type(adapter_raw["name"], str, "adapter.name")),
        version=cast(str, _require_type(adapter_raw["version"], str, "adapter.version")),
        sample_seed=seed,
        sample_size=size,
    )


def _parse_manifest(raw_value: object) -> tuple[DatasetManifest, tuple[str, ...]]:
    if not isinstance(raw_value, dict) or any(not isinstance(key, str) for key in raw_value):
        raise DatasetManifestError("Manifest root must be a mapping with string keys")
    raw = cast(dict[str, object], raw_value)
    schema_version_value = raw.get("schema_version")
    schema_version = (
        cast(str, _require_type(schema_version_value, str, "schema_version"))
        if schema_version_value is not None
        else None
    )
    warnings: list[str] = []
    if schema_version is None:
        missing = REQUIRED_MANIFEST_KEYS - set(raw)
        if missing:
            raise DatasetManifestError(
                f"Missing required manifest keys: {', '.join(sorted(missing))}"
            )
        unknown = set(raw) - LEGACY_MANIFEST_KEYS
        if unknown:
            warnings.append(f"Unknown legacy manifest keys ignored: {', '.join(sorted(unknown))}")
    else:
        if schema_version != "0.1":
            raise DatasetManifestError(f"Unsupported dataset schema_version: {schema_version}")
        unknown = set(raw) - VERSIONED_MANIFEST_KEYS
        missing = VERSIONED_REQUIRED_KEYS - set(raw)
        if unknown:
            raise DatasetManifestError(f"UNKNOWN_MANIFEST_KEY: {', '.join(sorted(unknown))}")
        if missing:
            raise DatasetManifestError(
                f"Missing manifest keys for schema_version 0.1: {', '.join(sorted(missing))}"
            )

    tier_value = raw.get("tier")
    risk_value = raw.get("contamination_risk")
    try:
        tier = DatasetTier(cast(str, tier_value)) if tier_value is not None else None
        risk = ContaminationRisk(cast(str, risk_value)) if risk_value is not None else None
    except ValueError as exc:
        raise DatasetManifestError(str(exc)) from exc

    return (
        DatasetManifest(
            name=cast(str, _require_type(_required(raw, "name"), str, "name")),
            version=cast(str, _require_type(_required(raw, "version"), str, "version")),
            split=cast(str, _require_type(_required(raw, "split"), str, "split")),
            license=cast(str, _require_type(_required(raw, "license"), str, "license")),
            pii_scrubbed=cast(
                bool,
                _require_type(_required(raw, "pii_scrubbed"), bool, "pii_scrubbed"),
            ),
            created_at=cast(
                str,
                _require_type(_required(raw, "created_at"), str, "created_at"),
            ),
            slices=_parse_string_list(_required(raw, "slices"), "slices"),
            content_sha256=cast(str | None, raw.get("content_sha256")),
            schema_version=schema_version,
            tier=tier,
            source=_parse_source(raw["source"]) if schema_version is not None else None,
            adapter=_parse_adapter(raw["adapter"]) if schema_version is not None else None,
            task_metrics=(
                _parse_string_list(raw["task_metrics"], "task_metrics")
                if schema_version is not None
                else None
            ),
            contamination_risk=risk,
            pii_scrub_procedure=(
                cast(
                    str,
                    _require_type(
                        raw["pii_scrub_procedure"],
                        str,
                        "pii_scrub_procedure",
                    ),
                )
                if schema_version is not None
                else None
            ),
        ),
        tuple(warnings),
    )


def load_dataset(dataset_dir: Path) -> DatasetBundle:
    manifest_path = dataset_dir / "manifest.yaml"
    cases_path = dataset_dir / "cases.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not cases_path.exists():
        raise FileNotFoundError(f"Missing cases: {cases_path}")

    manifest, manifest_warnings = _parse_manifest(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )

    cases: list[Case] = []
    content_lines: list[str] = []
    with cases_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            content_lines.append(line)
            cases.append(_parse_case(json.loads(line)))

    content_sha256 = sha256_hex("\n".join(content_lines).encode("utf-8"))
    return DatasetBundle(
        manifest=manifest,
        cases=cases,
        content_sha256=content_sha256,
        source_path=dataset_dir,
        manifest_warnings=manifest_warnings,
    )
