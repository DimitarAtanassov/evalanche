"""Load and validate versioned judge rubrics."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from evalharness.judge.errors import JudgeError
from evalharness.judge.models import Rubric


def load_rubric(path: Path) -> Rubric:
    """Load ``rubric.yaml`` schema 0.1 from disk."""
    if not path.is_file():
        raise JudgeError("MISSING_ARTIFACT", str(path))
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise JudgeError("INVALID_RUBRIC", f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeError("INVALID_RUBRIC", f"{path}: expected a mapping")
    try:
        return Rubric.model_validate(payload)
    except ValidationError as exc:
        raise JudgeError("INVALID_RUBRIC", str(exc)) from exc
