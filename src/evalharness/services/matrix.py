"""Matrix and baseline use cases: validate manifests and promote a pinned cell."""

from __future__ import annotations

from pathlib import Path

from evalharness.matrix import BaselineManifest, LoadedBaseline, LoadedMatrix
from evalharness.matrix import load_baseline as _load_baseline
from evalharness.matrix import load_matrix as _load_matrix
from evalharness.matrix import promote_baseline as _promote_baseline


class MatrixService:
    """Matrix and baseline load / promote."""

    def load_matrix(self, path: Path) -> LoadedMatrix:
        return _load_matrix(path)

    def load_baseline(self, path: Path, *, matrix: Path | None = None) -> LoadedBaseline:
        return _load_baseline(path, matrix=matrix)

    def promote_baseline(
        self,
        *,
        matrix_path: Path,
        cell_id: str,
        run_report_path: Path,
        output_path: Path,
        name: str | None = None,
        allow_mismatch: bool = False,
    ) -> BaselineManifest:
        return _promote_baseline(
            matrix_path=matrix_path,
            cell_id=cell_id,
            run_report_path=run_report_path,
            output_path=output_path,
            name=name,
            allow_mismatch=allow_mismatch,
        )
