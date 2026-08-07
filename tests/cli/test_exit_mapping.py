"""CLI exit-code mapping for score, calibrate, and runs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalharness.cli import app

CLI = CliRunner()


def test_score_happy_path_emits_json(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "c1", "output": "Paris", "reference": "Paris"}) + "\n",
        encoding="utf-8",
    )

    result = CLI.invoke(app, ["score", str(path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["id"] == "c1"
    assert payload["scores"][0]["metric"] == "exact_match"
    assert payload["scores"][0]["passed"] is True


def test_score_missing_file_exits_2(tmp_path: Path) -> None:
    result = CLI.invoke(app, ["score", str(tmp_path / "missing.jsonl")])

    assert result.exit_code == 2
    assert "IO_ERROR" in result.stdout


def test_score_invalid_json_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    result = CLI.invoke(app, ["score", str(path)])

    assert result.exit_code == 1
    assert "ERROR" in result.stdout


def test_score_unknown_metric_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "c1", "output": "x", "reference": "x"}) + "\n",
        encoding="utf-8",
    )

    result = CLI.invoke(app, ["score", str(path), "--metrics", "not_a_metric"])

    assert result.exit_code == 1
    assert "ERROR" in result.stdout
    assert "Unknown metric" in result.stdout


def test_calibrate_happy_path_emits_json(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    rows = [
        {"label": True, "score": 0.9},
        {"label": True, "score": 0.8},
        {"label": False, "score": 0.3},
        {"label": False, "score": 0.1},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = CLI.invoke(app, ["calibrate", str(path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "threshold" in payload
    assert "dev_f1" in payload


def test_calibrate_missing_file_exits_2(tmp_path: Path) -> None:
    result = CLI.invoke(app, ["calibrate", str(tmp_path / "missing.jsonl")])

    assert result.exit_code == 2
    assert "IO_ERROR" in result.stdout


def test_calibrate_one_class_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(
        "\n".join(json.dumps({"label": True, "score": 0.9}) for _ in range(3)) + "\n",
        encoding="utf-8",
    )

    result = CLI.invoke(app, ["calibrate", str(path)])

    assert result.exit_code == 1
    assert "ERROR" in result.stdout


def test_calibrate_missing_key_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps({"label": True}) + "\n", encoding="utf-8")

    result = CLI.invoke(app, ["calibrate", str(path)])

    assert result.exit_code == 1
    assert "ERROR" in result.stdout


def test_runs_rescore_invalid_uuid_exits_1() -> None:
    result = CLI.invoke(app, ["runs", "rescore", "not-a-uuid"])

    assert result.exit_code == 1
    assert "ERROR" in result.stdout


def test_runs_compare_invalid_uuid_exits_1() -> None:
    result = CLI.invoke(app, ["runs", "compare", "not-a-uuid", str(uuid.uuid4())])

    assert result.exit_code == 1
    assert "ERROR" in result.stdout


def test_runs_compare_write_failure_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_compare(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "baseline_run_id": "a",
            "candidate_run_id": "b",
            "excluded_flaky_cases": [],
            "result": {"metric": "exact_match", "n": 0},
        }

    monkeypatch.setattr("evalharness.cli.runs.compare_runs", fake_compare)
    output = tmp_path / "missing-parent" / "compare.json"

    result = CLI.invoke(
        app,
        [
            "runs",
            "compare",
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "IO_ERROR" in result.stdout
