"""Dataset use cases: load a bundle from disk and validate it."""

from __future__ import annotations

from pathlib import Path

from evalharness.datasets import DatasetBundle, ValidationReport
from evalharness.datasets import load_dataset as _load_dataset
from evalharness.datasets import validate_dataset as _validate_dataset


class DatasetService:
    """Load and validate dataset packs from disk."""

    def load_dataset(self, dataset_dir: Path) -> DatasetBundle:
        return _load_dataset(dataset_dir)

    def validate_dataset(
        self, bundle: DatasetBundle, *, allow_holdout: bool = False
    ) -> ValidationReport:
        return _validate_dataset(bundle, allow_holdout=allow_holdout)
