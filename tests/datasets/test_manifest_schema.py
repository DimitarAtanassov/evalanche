"""schema_version strictness and legacy compatibility for dataset manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evalharness.datasets import DatasetManifestError, load_dataset, validate_dataset
from evalharness.hashing import sha256_hex
from tests.datasets._helpers import (
    DELETE,
    SMOKE_ROOT,
    copy_dataset,
    load_case_dicts,
    rewrite_cases,
    rewrite_manifest,
    rewrite_source,
)


def test_legacy_manifest_loads_without_schema_version_fields() -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))

    assert bundle.manifest.schema_version is None
    assert bundle.manifest.source is None
    assert bundle.manifest.adapter is None
    assert validate_dataset(bundle).valid


def test_versioned_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "bad-version")
    rewrite_manifest(dataset, schema_version="9.9")

    with pytest.raises(DatasetManifestError, match="Unsupported dataset schema_version"):
        load_dataset(dataset)


def test_versioned_manifest_rejects_missing_source(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "missing-source")
    rewrite_manifest(dataset, source=DELETE)

    with pytest.raises(DatasetManifestError, match="Missing Phase 4 manifest keys: source"):
        load_dataset(dataset)


def test_versioned_manifest_rejects_missing_adapter(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "missing-adapter")
    rewrite_manifest(dataset, adapter=DELETE)

    with pytest.raises(DatasetManifestError, match="Missing Phase 4 manifest keys: adapter"):
        load_dataset(dataset)


def test_pii_scrubbed_true_requires_nonempty_procedure(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "empty-pii")
    rewrite_manifest(dataset, pii_scrubbed=True, pii_scrub_procedure="   ")

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert "pii_scrubbed=true requires pii_scrub_procedure" in report.errors


def test_source_revision_digest_must_be_sha256_hex(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "bad-digest")
    rewrite_source(dataset, revision_digest="not-a-digest")

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert "source.revision_digest must be sha256:<64 lowercase hex>" in report.errors


def test_canonical_url_must_be_absolute(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "relative-url")
    rewrite_source(dataset, canonical_url="tools/datasets/sources/synthetic_qa.jsonl")

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert "source.canonical_url must be an absolute URL" in report.errors


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "dup-ids")
    cases = load_case_dicts(dataset)
    cases[1]["id"] = cases[0]["id"]
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert f"Duplicate case id: {cases[0]['id']}" in report.errors


def test_duplicate_normalized_prompts_warn(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "dup-prompts")
    cases = load_case_dicts(dataset)
    cases[1]["inputs"] = dict(cases[0]["inputs"])
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert report.valid
    assert any("duplicate normalized prompts" in warning for warning in report.warnings)


def test_content_sha256_matches_joined_jsonl_lines() -> None:
    bundle = load_dataset(SMOKE_ROOT / "synthetic-qa-smoke")
    lines = [
        line.strip()
        for line in (SMOKE_ROOT / "synthetic-qa-smoke" / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert bundle.content_sha256 == sha256_hex("\n".join(lines).encode("utf-8"))
    assert bundle.manifest.content_sha256 == bundle.content_sha256


def test_classification_case_missing_expected_label_fails(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-news-smoke", tmp_path / "missing-label")
    cases = load_case_dicts(dataset)
    del cases[0]["expected_label"]
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any("requires field 'expected_label'" in error for error in report.errors)


def test_smoke_tier_cannot_use_holdout_split(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "smoke-holdout")
    rewrite_manifest(dataset, split="holdout")

    report = validate_dataset(load_dataset(dataset), allow_holdout=True)

    assert not report.valid
    assert "Only release tier may use split: holdout" in report.errors


def test_release_holdout_valid_with_allow_flag(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "release-holdout")
    rewrite_manifest(dataset, tier="release", split="holdout")

    denied = validate_dataset(load_dataset(dataset), allow_holdout=False)
    allowed = validate_dataset(load_dataset(dataset), allow_holdout=True)

    assert not denied.valid
    assert (
        "Holdout split requires --i-am-doing-a-final-eval flag to prevent overfitting."
        in denied.errors
    )
    assert allowed.valid
    assert allowed.errors == []


def test_release_tier_without_holdout_is_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "release-dev")
    rewrite_manifest(dataset, tier="release", split="dev")

    report = validate_dataset(load_dataset(dataset), allow_holdout=True)

    assert not report.valid
    assert "Release tier must use split: holdout" in report.errors


def test_task_metrics_must_intersect_case_task_type(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "wrong-metrics")
    rewrite_manifest(dataset, task_metrics=["classification"])

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any("no task-fit metric for qa_short" in error for error in report.errors)


def test_adapter_sample_size_must_match_case_count(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "sample-mismatch")
    manifest_path = dataset / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter"]["sample_size"] = 99
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(error.startswith("adapter.sample_size=99 does not match") for error in report.errors)


def test_missing_case_provenance_keys_are_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "missing-prov")
    cases = load_case_dicts(dataset)
    cases[0]["provenance"] = {"source_id": "synthetic_qa"}
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any("missing provenance keys" in error for error in report.errors)


def test_empty_case_provenance_values_are_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "empty-prov")
    cases = load_case_dicts(dataset)
    cases[0]["provenance"] = {
        "source_id": "synthetic_qa",
        "source_record_id": "qa-001",
        "source_revision": "synthetic-v1",
        "adapter_name": "synthetic_qa",
        "adapter_version": "   ",
    }
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any("provenance values must be non-empty strings" in error for error in report.errors)


def test_smoke_tier_case_count_bounds_are_enforced(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "too-few-cases")
    cases = load_case_dicts(dataset)[:4]
    rewrite_cases(dataset, cases)
    manifest_path = dataset / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter"]["sample_size"] = 4
    manifest["content_sha256"] = None
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(
        error.startswith("Tier smoke requires 5..20 cases; got 4") for error in report.errors
    )
