"""Privacy and resilience guarantees for the shared observability layer."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

from evalharness.app.settings import get_settings
from evalharness.cli_progress import PipelineProgress
from evalharness.observability import (
    PipelineStage,
    ProgressEvent,
    emit_progress,
    get_logger,
    payload_summary,
    sanitize_text,
    setup_logging,
)


def test_sanitize_text_redacts_credentials_and_bounds_output() -> None:
    value = "Authorization: Bearer super-secret\npassword=hunter2 more text"
    sanitized = sanitize_text(value, max_chars=80)
    assert "super-secret" not in sanitized
    assert "hunter2" not in sanitized
    assert sanitized.count("[REDACTED]") == 2
    assert "\n" not in sanitized
    assert len(sanitized) <= 80


def test_sanitize_text_counts_the_ellipsis_inside_max_chars() -> None:
    sanitized = sanitize_text("x" * 500, max_chars=280)

    assert len(sanitized) == 280
    assert sanitized == "x" * 279 + "…"


def test_sanitize_text_returns_input_unchanged_at_exactly_max_chars() -> None:
    sanitized = sanitize_text("x" * 280, max_chars=280)

    assert sanitized == "x" * 280


def test_payload_summary_omits_content_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_PAYLOADS", "false")
    get_settings.cache_clear()
    summary = payload_summary("private customer content")
    assert summary["chars"] == 24
    assert "sha256" not in summary
    assert "preview" not in summary


def test_payload_hashing_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_PAYLOAD_HASHES", "true")
    get_settings.cache_clear()
    summary = payload_summary("1")
    assert len(summary["sha256"]) == 64
    assert "preview" not in summary


def test_payload_summary_preview_is_explicit_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_PAYLOADS", "true")
    monkeypatch.setenv("LOG_PAYLOAD_MAX_CHARS", "40")
    get_settings.cache_clear()
    summary = payload_summary("api_key=secret-value " + "x" * 100)
    assert "secret-value" not in summary["preview"]
    assert "[REDACTED]" in summary["preview"]
    assert len(summary["preview"]) <= 41


def test_progress_callback_failure_cannot_fail_a_run(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()

    def broken(_: ProgressEvent) -> None:
        raise RuntimeError("UI unavailable")

    emit_progress(
        broken,
        ProgressEvent(PipelineStage.GENERATING, 1, 2, "case-1"),
    )
    record = json.loads(capsys.readouterr().err)
    assert record["event"] == "progress_callback_failed"
    assert record["stage"] == "generating"


def test_rich_progress_adapter_handles_stage_transitions() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=True)
    with PipelineProgress(console) as progress:
        progress(ProgressEvent(PipelineStage.GENERATING, 2, 5, "case-2"))
        progress(
            ProgressEvent(
                PipelineStage.REPORTING,
                3,
                3,
                "Wrote junit",
                {"path": "reports/run.xml"},
            )
        )
    rendered = output.getvalue()
    assert "Reporting" in rendered
    assert "Wrote junit" in rendered


def test_json_logs_remain_one_object_per_stderr_line_during_narrow_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    get_settings.cache_clear()
    setup_logging()
    progress_output = StringIO()
    narrow_console = Console(
        file=progress_output,
        force_terminal=True,
        no_color=True,
        width=12,
    )
    log = get_logger("progress-json-regression")

    with PipelineProgress(narrow_console) as progress:
        progress(ProgressEvent(PipelineStage.GENERATING, 1, 2, "case-with-a-long-name"))
        log.info("first_event", case_id="case-with-a-long-name")
        log.info("second_event", case_id="another-case-with-a-long-name")

    stderr_lines = capsys.readouterr().err.splitlines()
    records = [json.loads(line) for line in stderr_lines]
    assert [record["event"] for record in records] == ["first_event", "second_event"]
    assert "first_event" not in progress_output.getvalue()
