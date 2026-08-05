"""Deterministic mock judge response fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evalharness.judge.errors import JudgeError
from evalharness.judge.models import MockPairwiseResponse, MockPointwiseResponse


def load_mock_responses(
    path: Path,
) -> tuple[dict[str, MockPointwiseResponse], dict[tuple[str, int], MockPairwiseResponse]]:
    """Load mock judge responses keyed for pointwise and pairwise lookup."""
    if not path.is_file():
        raise JudgeError("MISSING_ARTIFACT", str(path))
    pointwise: dict[str, MockPointwiseResponse] = {}
    pairwise: dict[tuple[str, int], MockPairwiseResponse] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise JudgeError("INVALID_MOCK_RESPONSES", f"{path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JudgeError("INVALID_MOCK_RESPONSES", f"{path}:{index}: {exc}") from exc
        if not isinstance(payload, dict):
            raise JudgeError("INVALID_MOCK_RESPONSES", f"{path}:{index}: expected object")
        try:
            if "generation_id" in payload:
                response = MockPointwiseResponse.model_validate(payload)
                pointwise[response.generation_id] = response
            elif "case_id" in payload and "swap_position" in payload:
                pair_response = MockPairwiseResponse.model_validate(payload)
                pairwise[(pair_response.case_id, int(pair_response.swap_position))] = pair_response
            else:
                raise JudgeError(
                    "INVALID_MOCK_RESPONSES",
                    f"{path}:{index}: missing generation_id or case_id/swap_position",
                )
        except ValidationError as exc:
            raise JudgeError("INVALID_MOCK_RESPONSES", f"{path}:{index}: {exc}") from exc
    return pointwise, pairwise
