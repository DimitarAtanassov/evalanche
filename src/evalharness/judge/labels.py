"""Human-label JSONL loading for judge calibration."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evalharness.judge.errors import JudgeError
from evalharness.judge.models import HumanLabel, LabelSplit


def load_labels(path: Path, *, expected_split: LabelSplit | None = None) -> list[HumanLabel]:
    """Load human labels from a JSONL file.

    When ``expected_split`` is set, every row must match that split. A holdout
    file containing ``split: dev`` is a hard error (``DEV_USED_FOR_GATE``). A
    case labeled twice is a hard error too: ``n`` counts distinct labeled cases,
    so repeats would pad the sample the gate is measured on.
    """
    seen_case_ids: set[str] = set()
    if not path.is_file():
        raise JudgeError("MISSING_ARTIFACT", str(path))
    labels: list[HumanLabel] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise JudgeError("INVALID_LABELS", f"{path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            label = HumanLabel.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise JudgeError("INVALID_LABELS", f"{path}:{index}: {exc}") from exc
        if expected_split is not None and label.split != expected_split:
            if expected_split is LabelSplit.HOLDOUT and label.split is LabelSplit.DEV:
                raise JudgeError(
                    "DEV_USED_FOR_GATE",
                    f"{path}:{index}: holdout labels file contains split=dev",
                )
            raise JudgeError(
                "HOLDOUT_REQUIRED",
                f"{path}:{index}: expected split={expected_split.value}, got {label.split.value}",
            )
        if label.case_id in seen_case_ids:
            raise JudgeError(
                "DUPLICATE_CASE_ID",
                f"{path}:{index}: case_id={label.case_id} is labeled more than once",
            )
        seen_case_ids.add(label.case_id)
        labels.append(label)
    if not labels:
        raise JudgeError("INVALID_LABELS", f"{path}: no labels found")
    label_set_ids = {label.label_set_id for label in labels}
    if len(label_set_ids) != 1:
        raise JudgeError(
            "INVALID_LABELS",
            f"{path}: expected a single label_set_id, got {sorted(label_set_ids)}",
        )
    return labels
