"""Golden assertions for the committed evalanche PoC report."""

from __future__ import annotations

import json
from pathlib import Path

from evalharness.providers.mock import MOCK_DIGEST

POC_DIR = Path("fixtures/poc")


def test_poc_artifacts_exist() -> None:
    assert (POC_DIR / "report.json").is_file()
    assert (POC_DIR / "report.html").is_file()
    assert (POC_DIR / "meta.json").is_file()


def test_poc_report_is_publishable_and_complete() -> None:
    report = json.loads((POC_DIR / "report.json").read_text(encoding="utf-8"))
    meta = json.loads((POC_DIR / "meta.json").read_text(encoding="utf-8"))

    assert report["run_id"] == "00000000-0000-4000-8000-0000000000c1"
    assert report["model_digest"] == MOCK_DIGEST
    assert report["publishable"] is True
    assert report["coverage"] == 1.0
    assert report["pass_rate"] == 1.0
    assert (
        0.0
        <= report["pass_rate_ci"]["low"]
        <= report["pass_rate"]
        <= report["pass_rate_ci"]["high"]
    )
    assert set(report["latency_ms"]) >= {"p50", "p90", "p95", "p99", "max", "mean"}
    assert report["outcome_histogram"].get("passed", 0) == 5
    assert meta["provider"] == "mock"
    assert meta["config_sha256"] == report["config_sha256"]


def test_poc_report_rolls_up_slices() -> None:
    report = json.loads((POC_DIR / "report.json").read_text(encoding="utf-8"))
    slices = {row["slice"] for row in report["metric_aggregates"]}
    assert "__overall__" in slices
    # fixtures/sample_dataset tags every case with difficulty and lang.
    assert {"difficulty=easy", "difficulty=hard", "lang=en"} <= slices


def test_poc_html_contains_run_id() -> None:
    html = (POC_DIR / "report.html").read_text(encoding="utf-8")
    assert "00000000-0000-4000-8000-0000000000c1" in html
    assert "Pass rate" in html
    assert "What was evaluated, and on what?" in html
    assert "Sampled cases" in html


def test_poc_report_includes_evaluation_context() -> None:
    report = json.loads((POC_DIR / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "2.2"
    assert report["model"]["provider"] == "mock"
    assert report["model"]["resolved_version"] == MOCK_DIGEST
    assert report["dataset"]["name"]
    assert report["dataset"]["case_count"] == 5
    assert report["prompt_template"]["body"]
    assert isinstance(report["decode_params"], dict)
    assert len(report["case_examples"]) == 5
    example = report["case_examples"][0]
    assert {"case_id", "input", "reference", "output", "passed"} <= set(example)
    assert "raw_response" not in example


def test_poc_html_is_offline_self_contained() -> None:
    html = (POC_DIR / "report.html").read_text(encoding="utf-8")
    assert 'src="http' not in html
