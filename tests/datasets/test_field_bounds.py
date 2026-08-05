"""Published text bounds and PII field-limit fail-closed checks."""

from __future__ import annotations

from pathlib import Path

from evalharness.datasets import load_dataset, validate_dataset
from evalharness.datasets.validator import INPUT_TEXT_LIMIT, REFERENCE_TEXT_LIMIT
from tests.datasets._helpers import (
    SMOKE_ROOT,
    copy_dataset,
    load_case_dicts,
    rewrite_cases,
    rewrite_manifest,
)


def test_input_at_exact_limit_is_accepted(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "exact-input")
    cases = load_case_dicts(dataset)
    cases[0]["inputs"]["question"] = "x" * INPUT_TEXT_LIMIT
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert report.valid
    assert report.errors == []


def test_oversize_input_text_is_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "long-input")
    cases = load_case_dicts(dataset)
    cases[0]["inputs"]["question"] = "x" * (INPUT_TEXT_LIMIT + 1)
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(error.startswith("FIELD_LENGTH_EXCEEDED:") for error in report.errors)
    assert any("inputs.question" in error for error in report.errors)


def test_reference_at_exact_limit_is_accepted(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "exact-ref")
    cases = load_case_dicts(dataset)
    cases[0]["reference_answer"] = "y" * REFERENCE_TEXT_LIMIT
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert report.valid
    assert report.errors == []


def test_oversize_reference_answer_is_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "long-ref")
    cases = load_case_dicts(dataset)
    cases[0]["reference_answer"] = "y" * (REFERENCE_TEXT_LIMIT + 1)
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(
        error.startswith("FIELD_LENGTH_EXCEEDED:") and "reference exceeds" in error
        for error in report.errors
    )


def test_oversize_expected_label_is_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-news-smoke", tmp_path / "long-label")
    cases = load_case_dicts(dataset)
    cases[0]["expected_label"] = "z" * (REFERENCE_TEXT_LIMIT + 1)
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(error.startswith("FIELD_LENGTH_EXCEEDED:") for error in report.errors)


def test_retrieval_candidate_body_over_limit_is_rejected(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-retrieval-smoke", tmp_path / "long-body")
    cases = load_case_dicts(dataset)
    cases[0]["inputs"]["candidates"][0]["text"] = "b" * (INPUT_TEXT_LIMIT + 1)
    rewrite_cases(dataset, cases)
    rewrite_manifest(dataset, content_sha256=None)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any("FIELD_LENGTH_EXCEEDED" in error for error in report.errors)
