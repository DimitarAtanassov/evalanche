"""File-primary eval matrix and pinned baseline promotion."""

from evalharness.matrix.errors import MatrixValidationError
from evalharness.matrix.loader import load_baseline, load_matrix, matrix_digest, promote_baseline
from evalharness.matrix.models import (
    BaselineManifest,
    LoadedBaseline,
    LoadedMatrix,
    MatrixManifest,
)

__all__ = [
    "BaselineManifest",
    "LoadedBaseline",
    "LoadedMatrix",
    "MatrixManifest",
    "MatrixValidationError",
    "load_baseline",
    "load_matrix",
    "matrix_digest",
    "promote_baseline",
]
