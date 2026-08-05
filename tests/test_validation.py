from pathlib import Path

from evalharness.datasets import load_dataset, validate_dataset
from evalharness.datasets.loader import DatasetBundle, DatasetManifest


def test_sample_dataset_valid() -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    report = validate_dataset(bundle)
    assert report.valid
    assert len(bundle.cases) == 5


def test_holdout_requires_flag() -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    holdout_bundle = DatasetBundle(
        manifest=DatasetManifest(
            name=bundle.manifest.name,
            version=bundle.manifest.version,
            split="holdout",
            license=bundle.manifest.license,
            pii_scrubbed=bundle.manifest.pii_scrubbed,
            created_at=bundle.manifest.created_at,
            slices=bundle.manifest.slices,
            content_sha256=bundle.content_sha256,
        ),
        cases=bundle.cases,
        content_sha256=bundle.content_sha256,
        source_path=bundle.source_path,
    )
    report = validate_dataset(holdout_bundle, allow_holdout=False)
    assert not report.valid
