"""Dataset identities remain collision-free across fixtures and materializations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from evalharness.datasets import DatasetTier, load_dataset
from tests.datasets._helpers import smoke_paths
from tools.datasets.adapters import ADAPTERS
from tools.datasets.materialize import SourcePin, materialization_version


def test_all_committed_fixture_name_versions_are_unique() -> None:
    fixture_paths = (
        Path("fixtures/sample_dataset"),
        Path("fixtures/large_dataset"),
        *smoke_paths(),
    )

    identities = [
        (bundle.manifest.name, bundle.manifest.version)
        for bundle in (load_dataset(path) for path in fixture_paths)
    ]

    assert len(identities) == len(set(identities))
    assert identities[0] == ("synthetic-qa-sample", "1.0.0")
    assert all(name.startswith("phase4-") for name, _ in identities[2:])


def test_materialization_version_covers_every_identity_input() -> None:
    spec = ADAPTERS["synthetic_qa"]
    pin = SourcePin(
        revision="synthetic-v1",
        revision_digest=f"sha256:{'a' * 64}",
        canonical_url="repo://tools/datasets/sources/synthetic_qa.jsonl",
    )
    baseline = materialization_version(
        spec,
        pin,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
    )
    variants = {
        materialization_version(
            replace(spec, version="1.0.1"),
            pin,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        ),
        materialization_version(
            spec,
            replace(pin, revision="synthetic-v2"),
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        ),
        # Same revision label, different bytes: only the digest separates them.
        materialization_version(
            spec,
            replace(pin, revision_digest=f"sha256:{'b' * 64}"),
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        ),
        materialization_version(
            spec,
            pin,
            seed=43,
            size=5,
            tier=DatasetTier.SMOKE,
        ),
        materialization_version(
            spec,
            pin,
            seed=42,
            size=6,
            tier=DatasetTier.SMOKE,
        ),
        materialization_version(
            spec,
            pin,
            seed=42,
            size=5,
            tier=DatasetTier.RELEASE,
        ),
    }

    assert baseline == "1.0.0+materialized.29be8821a9541e60"
    assert baseline not in variants
    assert len(variants) == 6
