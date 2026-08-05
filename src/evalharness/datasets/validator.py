"""Dataset validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalharness.datasets.loader import TASK_REQUIRED_FIELDS, DatasetBundle


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_prompts: list[tuple[str, str]] = field(default_factory=list)


def _field_present(case: object, field_name: str) -> bool:
    value = getattr(case, field_name, None)
    if value is None:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


def validate_dataset(
    bundle: DatasetBundle,
    *,
    allow_holdout: bool = False,
) -> ValidationReport:
    report = ValidationReport(valid=True)
    manifest = bundle.manifest

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
