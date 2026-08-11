"""Suite use cases: validate a manifest and build its artifacts."""

from __future__ import annotations

from pathlib import Path

from evalharness.suite import LoadedSuite, SuiteReport
from evalharness.suite import load_suite as _load_suite
from evalharness.suite import write_suite_artifacts as _write_suite_artifacts


class SuiteService:
    """Multi-run suite validate and build."""

    def load_suite(self, path: Path) -> LoadedSuite:
        return _load_suite(path)

    def write_suite_artifacts(self, manifest_path: Path, output_dir: Path) -> SuiteReport:
        return _write_suite_artifacts(manifest_path, output_dir)
