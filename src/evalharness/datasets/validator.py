"""Dataset validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalharness.datasets.loader import TASK_REQUIRED_FIELDS, DatasetBundle, DatasetTier
from evalharness.domain.enums import TaskType

ALLOWED_SMOKE_LICENSES = frozenset(
    {
        "CC0-1.0",
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
    }
)
ATTRIBUTION_LICENSES = frozenset({"CC-BY-4.0", "CC-BY-SA-4.0"})
FIXTURES_DIR = "fixtures"
TIER_SIZE_BOUNDS: dict[DatasetTier, tuple[int, int | None]] = {
    DatasetTier.SMOKE: (5, 20),
    DatasetTier.CI: (50, 200),
    DatasetTier.NIGHTLY: (500, 2_000),
    DatasetTier.RELEASE: (1, None),
}
TASK_METRICS: dict[TaskType, frozenset[str]] = {
    TaskType.QA_SHORT: frozenset({"squad_f1", "exact_match", "numeric_assertion"}),
    TaskType.CLASSIFICATION: frozenset({"classification"}),
    TaskType.EXTRACTION: frozenset({"json_validity", "json_field_f1"}),
    TaskType.SUMMARIZATION: frozenset({"rouge_l", "chrf_pp", "meteor"}),
    TaskType.RETRIEVAL: frozenset(
        {"retrieval_ndcg_10", "retrieval_precision_at_k", "retrieval_mrr", "retrieval_map"}
    ),
}
INPUT_TEXT_LIMIT = 2_000
REFERENCE_TEXT_LIMIT = 500
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIRED_CASE_PROVENANCE = frozenset(
    {"source_id", "source_record_id", "source_revision", "adapter_name", "adapter_version"}
)


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_prompts: list[tuple[str, str]] = field(default_factory=list)


def _is_committed_pack(source_path: Path) -> bool:
    """True when the pack sits inside a ``fixtures/`` tree, so its text ships with the repo."""
    return FIXTURES_DIR in source_path.resolve().parts


def _field_present(case: object, field_name: str) -> bool:
    value = getattr(case, field_name, None)
    if value is None:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


def _validate_license_policy(bundle: DatasetBundle, report: ValidationReport) -> None:
    """Allow-list when text ships with the repo or claims redistributable smoke.

    Location decides redistribution, not only the operator's claim: a pack under
    ``fixtures/`` ships with the repository whether or not it sets the flag, and
    whether or not it carries a versioned ``schema_version``.
    """
    manifest = bundle.manifest
    source = manifest.source
    committed = _is_committed_pack(bundle.source_path)
    redistributable = source is not None and source.redistributable_smoke
    if not (committed or redistributable):
        return
    if manifest.license not in ALLOWED_SMOKE_LICENSES:
        report.errors.append(
            f"LICENSE_BLOCK: {manifest.license} is not allowed for committed smoke text"
        )
    if manifest.license in ATTRIBUTION_LICENSES:
        attribution = source.attribution if source is not None else ""
        if not attribution.strip():
            report.errors.append(f"{manifest.license} requires source.attribution")


def _validate_versioned_manifest(bundle: DatasetBundle, report: ValidationReport) -> None:
    manifest = bundle.manifest
    if manifest.schema_version is None:
        report.warnings.extend(bundle.manifest_warnings)
        return
    if (
        manifest.tier is None
        or manifest.source is None
        or manifest.adapter is None
        or manifest.task_metrics is None
        or manifest.contamination_risk is None
        or manifest.pii_scrub_procedure is None
    ):
        report.errors.append("schema_version 0.1 manifest fields must all be present")
        return

    source = manifest.source
    if not SHA256_PATTERN.fullmatch(source.revision_digest):
        report.errors.append("source.revision_digest must be sha256:<64 lowercase hex>")
    if "://" not in source.canonical_url:
        report.errors.append("source.canonical_url must be an absolute URL")
    if manifest.pii_scrubbed and not manifest.pii_scrub_procedure.strip():
        report.errors.append("pii_scrubbed=true requires pii_scrub_procedure")

    minimum, maximum = TIER_SIZE_BOUNDS[manifest.tier]
    case_count = len(bundle.cases)
    if case_count < minimum or (maximum is not None and case_count > maximum):
        upper = str(maximum) if maximum is not None else "unbounded"
        report.errors.append(
            f"Tier {manifest.tier.value} requires {minimum}..{upper} cases; got {case_count}"
        )
    if manifest.adapter.sample_size != case_count:
        report.errors.append(
            f"adapter.sample_size={manifest.adapter.sample_size} does not match {case_count} cases"
        )
    if manifest.tier is DatasetTier.RELEASE and manifest.split != "holdout":
        report.errors.append("Release tier must use split: holdout")
    if manifest.tier is not DatasetTier.RELEASE and manifest.split == "holdout":
        report.errors.append("Only release tier may use split: holdout")


def _validate_text_bounds(case_id: str, value: Any, path: str, report: ValidationReport) -> None:
    if isinstance(value, str):
        if len(value) > INPUT_TEXT_LIMIT:
            report.errors.append(
                f"FIELD_LENGTH_EXCEEDED: case {case_id} {path} exceeds {INPUT_TEXT_LIMIT} chars"
            )
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_text_bounds(case_id, child, f"{path}.{key}", report)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_text_bounds(case_id, child, f"{path}[{index}]", report)


def _validate_case_contract(bundle: DatasetBundle, report: ValidationReport) -> None:
    manifest = bundle.manifest
    for case in bundle.cases:
        _validate_text_bounds(case.external_id, case.inputs, "inputs", report)
        references = [case.reference_answer, case.expected_label, *case.references]
        if any(value is not None and len(value) > REFERENCE_TEXT_LIMIT for value in references):
            report.errors.append(
                f"FIELD_LENGTH_EXCEEDED: case {case.external_id} reference exceeds "
                f"{REFERENCE_TEXT_LIMIT} chars"
            )
        if manifest.schema_version is not None:
            missing = REQUIRED_CASE_PROVENANCE - set(case.provenance)
            if missing:
                report.errors.append(
                    f"Case {case.external_id}: missing provenance keys {', '.join(sorted(missing))}"
                )
            invalid = [
                key
                for key in REQUIRED_CASE_PROVENANCE
                if key in case.provenance
                and (not isinstance(case.provenance[key], str) or not case.provenance[key].strip())
            ]
            if invalid:
                report.errors.append(
                    f"Case {case.external_id}: provenance values must be non-empty strings: "
                    f"{', '.join(sorted(invalid))}"
                )
            if manifest.task_metrics is not None:
                allowed = TASK_METRICS.get(case.task_type)
                if allowed is not None and not (set(manifest.task_metrics) & allowed):
                    report.errors.append(
                        f"Case {case.external_id}: no task-fit metric for {case.task_type.value}"
                    )


def validate_dataset(
    bundle: DatasetBundle,
    *,
    allow_holdout: bool = False,
) -> ValidationReport:
    report = ValidationReport(valid=True)
    manifest = bundle.manifest
    _validate_license_policy(bundle, report)
    _validate_versioned_manifest(bundle, report)

    if manifest.split == "holdout" and not allow_holdout:
        report.errors.append(
            "Holdout split requires --i-am-doing-a-final-eval flag to prevent overfitting."
        )

    seen_ids: set[str] = set()
    prompt_to_id: dict[str, str] = {}

    for case in bundle.cases:
        if case.external_id in seen_ids:
            report.errors.append(f"Duplicate case id: {case.external_id}")
        seen_ids.add(case.external_id)

        required = TASK_REQUIRED_FIELDS[case.task_type]
        for req_field in required:
            if not _field_present(case, req_field):
                report.errors.append(
                    f"Case {case.external_id}: task_type={case.task_type.value} "
                    f"requires field '{req_field}'"
                )

        for slice_key in manifest.slices:
            if slice_key not in case.slices:
                report.errors.append(
                    f"Case {case.external_id}: missing required slice key '{slice_key}'"
                )

        if case.normalized_prompt:
            if case.normalized_prompt in prompt_to_id:
                report.duplicate_prompts.append(
                    (case.external_id, prompt_to_id[case.normalized_prompt])
                )
            else:
                prompt_to_id[case.normalized_prompt] = case.external_id

    _validate_case_contract(bundle, report)

    if manifest.content_sha256 and manifest.content_sha256 != bundle.content_sha256:
        report.errors.append(
            f"Manifest content_sha256 mismatch: expected {manifest.content_sha256}, "
            f"got {bundle.content_sha256}"
        )

    if report.duplicate_prompts:
        report.warnings.append(
            f"Found {len(report.duplicate_prompts)} duplicate normalized prompts"
        )

    report.valid = len(report.errors) == 0
    return report
