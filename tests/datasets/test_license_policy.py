"""License and redistributable-smoke fail-closed policy."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evalharness.datasets import DatasetTier, load_dataset, validate_dataset
from tests.datasets._helpers import (
    DELETE,
    SMOKE_ROOT,
    SOURCE_ROOT,
    copy_dataset,
    rewrite_manifest,
    rewrite_source,
)
from tools.datasets import MaterializationError, materialize_dataset
from tools.datasets.adapters import ADAPTERS

CACHE_ONLY_ADAPTERS = (
    "financial_phrasebank",
    "cnn_dailymail",
    "xsum",
    "ag_news",
)


@pytest.mark.parametrize("adapter_name", CACHE_ONLY_ADAPTERS)
def test_cache_only_adapter_cannot_write_under_fixtures(adapter_name: str) -> None:
    output = SMOKE_ROOT / f"forbidden-{adapter_name}"
    assert not output.exists()

    with pytest.raises(MaterializationError, match="LICENSE_BLOCK"):
        materialize_dataset(
            adapter_name=adapter_name,
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert not output.exists()


@pytest.mark.parametrize("license_id", ["unknown", "CC-BY-NC-SA-3.0", "DUA"])
def test_disallowed_license_with_redistributable_smoke_is_blocked(
    tmp_path: Path, license_id: str
) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "bad-license")
    rewrite_manifest(dataset, license=license_id)
    rewrite_source(dataset, redistributable_smoke=True)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(error.startswith("LICENSE_BLOCK:") for error in report.errors)


@pytest.mark.parametrize("license_id", ["CC-BY-4.0", "CC-BY-SA-4.0"])
def test_attribution_license_requires_nonempty_attribution(tmp_path: Path, license_id: str) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "missing-attr")
    rewrite_manifest(dataset, license=license_id)
    rewrite_source(dataset, attribution="   ", redistributable_smoke=True)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert f"{license_id} requires source.attribution" in report.errors


@pytest.mark.parametrize("adapter_name", CACHE_ONLY_ADAPTERS)
def test_cache_only_adapter_cannot_write_under_fixtures_without_a_git_root(
    tmp_path: Path, adapter_name: str
) -> None:
    """A bare tree or unpacked tarball carries the same redistribution obligations."""
    output = tmp_path / "unpacked" / "fixtures" / "datasets" / f"forbidden-{adapter_name}"

    with pytest.raises(MaterializationError, match="LICENSE_BLOCK"):
        materialize_dataset(
            adapter_name=adapter_name,
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert not output.exists()


def test_unlisted_license_under_fixtures_is_rejected_despite_false_flag(tmp_path: Path) -> None:
    """Committed location decides redistribution, not the redistributable_smoke boolean."""
    dataset = copy_dataset(
        SMOKE_ROOT / "synthetic-qa-smoke",
        tmp_path / "fixtures" / "datasets" / "nc-license",
    )
    rewrite_manifest(dataset, license="CC-BY-NC-SA-3.0")
    rewrite_source(dataset, redistributable_smoke=False)

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(error.startswith("LICENSE_BLOCK:") for error in report.errors)


def test_legacy_unlisted_license_under_fixtures_is_rejected(tmp_path: Path) -> None:
    """Allow-list applies under fixtures/ even when schema_version is absent."""
    dataset = copy_dataset(
        SMOKE_ROOT / "synthetic-qa-smoke",
        tmp_path / "fixtures" / "datasets" / "legacy-nc-license",
    )
    rewrite_manifest(
        dataset,
        license="CC-BY-NC-SA-3.0",
        schema_version=DELETE,
        tier=DELETE,
        source=DELETE,
        adapter=DELETE,
        task_metrics=DELETE,
        contamination_risk=DELETE,
        pii_scrub_procedure=DELETE,
    )

    report = validate_dataset(load_dataset(dataset))

    assert not report.valid
    assert any(error.startswith("LICENSE_BLOCK:") for error in report.errors)


def test_unlisted_license_outside_fixtures_stays_valid_for_cache_use(tmp_path: Path) -> None:
    dataset = copy_dataset(SMOKE_ROOT / "synthetic-qa-smoke", tmp_path / "cache" / "nc-license")
    rewrite_manifest(dataset, license="CC-BY-NC-SA-3.0")
    rewrite_source(dataset, redistributable_smoke=False)

    report = validate_dataset(load_dataset(dataset))

    assert report.valid


def test_dataset_card_registry_without_the_adapter_blocks_fixture_write(tmp_path: Path) -> None:
    """Mentioning a source id without meaningful card fields still fails closed."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "datasets.md").write_text(
        "The synthetic_qa adapter exists, but this is not a dataset card.\n",
        encoding="utf-8",
    )
    output = repo / "fixtures" / "datasets" / "uncarded-smoke"

    with pytest.raises(MaterializationError, match="LICENSE_BLOCK"):
        materialize_dataset(
            adapter_name="synthetic_qa",
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert not output.exists()


def test_complete_dataset_card_allows_redistributable_fixture_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "datasets.md").write_text(
        """### synthetic_qa

- License: `CC0-1.0`.
- Redistribution: repository-authored text may be committed.
- Attribution: repository-authored synthetic fixture.
- Source revision: `synthetic-v1`.
- Task/metrics: short QA; exact match.
- Privacy: fictional bounded text.
""",
        encoding="utf-8",
    )
    output = repo / "fixtures" / "datasets" / "synthetic-qa-smoke"

    materialize_dataset(
        adapter_name="synthetic_qa",
        source=SOURCE_ROOT / "synthetic_qa.jsonl",
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
    )

    assert validate_dataset(load_dataset(output)).valid


def test_by_materialization_requires_nonempty_adapter_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = ADAPTERS["synthetic_qa"]
    monkeypatch.setitem(
        ADAPTERS,
        "synthetic_qa",
        replace(base, license="CC-BY-4.0", attribution="   "),
    )
    output = tmp_path / "cache" / "missing-attribution"

    with pytest.raises(
        MaterializationError,
        match="CC-BY-4.0 requires nonempty attribution",
    ):
        materialize_dataset(
            adapter_name="synthetic_qa",
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert not output.exists()


def test_cache_only_materialize_outside_fixtures_is_allowed(tmp_path: Path) -> None:
    """Banned SPDX may still write outside tracked fixtures when source bytes pin."""
    source = tmp_path / "scifact-prejoined.jsonl"
    source.write_text(
        (
            '{"id":"c1","task_type":"retrieval","inputs":{"query":"q","candidates":[{"id":"d1"}]},'
            '"qrels":{"d1":1},"slices":{"domain":"science"}}\n'
            '{"id":"c2","task_type":"retrieval","inputs":{"query":"q2","candidates":[{"id":"d2"}]},'
            '"qrels":{"d2":1},"slices":{"domain":"science"}}\n'
            '{"id":"c3","task_type":"retrieval","inputs":{"query":"q3","candidates":[{"id":"d3"}]},'
            '"qrels":{"d3":1},"slices":{"domain":"science"}}\n'
            '{"id":"c4","task_type":"retrieval","inputs":{"query":"q4","candidates":[{"id":"d4"}]},'
            '"qrels":{"d4":1},"slices":{"domain":"science"}}\n'
            '{"id":"c5","task_type":"retrieval","inputs":{"query":"q5","candidates":[{"id":"d5"}]},'
            '"qrels":{"d5":1},"slices":{"domain":"science"}}\n'
        ),
        encoding="utf-8",
    )
    pin = {
        "revision": "operator-pinned",
        "revision_digest": f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}",
        "canonical_url": "https://example.invalid/scifact-prejoined.jsonl",
    }
    source.with_name(f"{source.name}.pin.yaml").write_text(
        yaml.safe_dump(pin),
        encoding="utf-8",
    )
    output = tmp_path / "cache" / "scifact-prejoined-smoke"

    materialize_dataset(
        adapter_name="scifact_prejoined",
        source=source,
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
    )

    bundle = load_dataset(output)
    report = validate_dataset(bundle)
    assert report.valid
    assert bundle.manifest.source is not None
    assert bundle.manifest.source.redistributable_smoke is False
