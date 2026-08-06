"""The evalctl `calibrate` command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evalharness.cli._common import _emit_json
from evalharness.scoring.calibration import calibrate_threshold


def calibrate(inputs: Path = typer.Argument(..., help="JSONL with label and score")) -> None:
    """Calibrate a threshold on development data."""
    rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines()]
    result = calibrate_threshold(
        [bool(row["label"]) for row in rows],
        [float(row["score"]) for row in rows],
    )
    _emit_json(result)
