"""JSONL / JSON helpers for judge artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from evalharness.judge.errors import JudgeError


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise JudgeError("MISSING_ARTIFACT", str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeError("INVALID_ARTIFACT", f"{path}: expected a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any] | BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_jsonl_models[T: BaseModel](path: Path, model_type: type[T], *, error_code: str) -> list[T]:
    if not path.is_file():
        raise JudgeError("MISSING_ARTIFACT", str(path))
    rows: list[T] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise JudgeError(error_code, f"{path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            rows.append(model_type.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise JudgeError(error_code, f"{path}:{index}: {exc}") from exc
    if not rows:
        raise JudgeError(error_code, f"{path}: no rows found")
    return rows
