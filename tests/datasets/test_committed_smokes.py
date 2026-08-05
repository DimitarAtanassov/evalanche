"""Committed smoke fixture policy and presence checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalharness.datasets import load_dataset, validate_dataset
from evalharness.datasets.validator import ALLOWED_SMOKE_LICENSES, ATTRIBUTION_LICENSES
from tests.datasets._helpers import SMOKE_ROOT, smoke_paths

BANNED_FIXTURE_NAMES = (
    "financial-phrasebank",
    "phrasebank",
    "cnn-dailymail",
    "cnn_dailymail",
    "xsum",
    "ag-news",
    "ag_news",
)


@pytest.mark.parametrize("dataset_path", smoke_paths(), ids=lambda path: path.name)
def test_committed_smoke_meets_redistribution_policy(dataset_path: Path) -> None:
    bundle = load_dataset(dataset_path)
    report = validate_dataset(bundle)
    manifest = bundle.manifest

    assert report.errors == []
    assert report.valid
    assert manifest.schema_version == "0.1"
    assert manifest.tier is not None and manifest.tier.value == "smoke"
    assert manifest.split == "dev"
    assert manifest.license in ALLOWED_SMOKE_LICENSES
    assert manifest.pii_scrubbed is True
    assert manifest.pii_scrub_procedure is not None
    assert manifest.pii_scrub_procedure.strip()
    assert manifest.source is not None
    assert manifest.source.redistributable_smoke is True
    assert manifest.task_metrics
    if manifest.license in ATTRIBUTION_LICENSES:
        assert manifest.source.attribution.strip()


def test_banned_external_corpora_are_absent_from_committed_fixtures() -> None:
    names = {path.name.lower() for path in SMOKE_ROOT.iterdir() if path.is_dir()}
    for banned in BANNED_FIXTURE_NAMES:
        assert not any(banned in name for name in names), names


def test_eight_license_safe_smokes_are_present() -> None:
    expected = {
        "synthetic-qa-smoke",
        "synthetic-news-smoke",
        "synthetic-healthcare-smoke",
        "synthetic-finance-smoke",
        "synthetic-summarization-smoke",
        "synthetic-extraction-smoke",
        "synthetic-retrieval-smoke",
        "synthetic-math-smoke",
    }
    assert {path.name for path in smoke_paths()} == expected
