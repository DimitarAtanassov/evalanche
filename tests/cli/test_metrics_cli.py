"""`evalctl metrics list` reports what this install can score."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from evalharness.cli import app

CLI = CliRunner()


def test_metrics_list_reports_enabled_metrics() -> None:
    result = CLI.invoke(app, ["metrics", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "exact_match" in payload["enabled"]
    assert "retrieval_ndcg_10" in payload["enabled"]


def test_metrics_list_reports_a_disabled_family_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METRIC_FAMILIES", "lexical")

    result = CLI.invoke(app, ["metrics", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    disabled = {row["name"]: row for row in payload["metrics"] if not row["enabled"]}
    assert "rouge_l" not in payload["enabled"]
    assert disabled["rouge_l"]["family"] == "overlap"
    assert "METRIC_FAMILIES" in disabled["rouge_l"]["reason"]
