"""Gate use cases: load a manifest with its bound artifacts and evaluate it."""

from __future__ import annotations

from pathlib import Path

from evalharness.gates import ArtifactOverrides, GatesEvaluation, LoadedGates
from evalharness.gates import evaluate_gates as _evaluate_gates
from evalharness.gates import load_gates as _load_gates


class GatesService:
    """Gate manifest load and evaluation."""

    def load_gates(self, path: Path, *, overrides: ArtifactOverrides | None = None) -> LoadedGates:
        return _load_gates(path, overrides=overrides)

    def evaluate_gates(self, loaded: LoadedGates) -> GatesEvaluation:
        return _evaluate_gates(loaded)
