"""Reporting builds through an injected RunStore without session.get for definitions."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from evalharness.core.enums import FailureOutcome, TaskType
from evalharness.core.models import Case
from evalharness.reporting import io as report_io
from evalharness.reporting.report import build_report

RUN_ID = uuid.UUID("00000000-0000-4000-8000-0000000000b1")


class _SessionGetForbidden:
    """Session stand-in: definition loads must not fall back to session.get."""

    async def get(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("build_report must load definitions via RunStore, not session.get")


class _FakeReportStore:
    """Minimal store implementing only the reads ``build_report`` uses."""

    def __init__(self, _session: object) -> None:
        self._case = Case(
            external_id="c1",
            task_type=TaskType.QA_SHORT,
            inputs={"question": "2+2?"},
            reference_answer="4",
        )
        self._run = SimpleNamespace(
            dataset_id=10,
            model_version_id=20,
            prompt_template_id=30,
            decode_params={"temperature": 0.0, "max_tokens": 32},
            status="completed",
            config_sha256="c" * 64,
        )
        self._dataset = SimpleNamespace(
            name="sample",
            version="1.0.0",
            split="test",
            content_sha256="d" * 64,
            manifest={"license": "test", "pii_scrubbed": True, "slices": ["lang"]},
        )
        self._model = SimpleNamespace(
            provider="mock",
            model="mock-model",
            resolved_version="mock-digest",
            quantization=None,
            params_b=None,
            context_window=None,
            capabilities={},
        )
        self._template = SimpleNamespace(
            name="default",
            version="1",
            content_sha256="e" * 64,
            body="Answer: {{ question }}",
        )
        self._generation = SimpleNamespace(
            id=1,
            case_id=1,
            repeat_idx=0,
            output="4",
            outcome=FailureOutcome.PASSED.value,
            total_ms=5.0,
            trace_id=None,
            finish_reason="stop",
            cost_usd=0.0,
            attempts=1,
            cached=False,
        )
        self._score = SimpleNamespace(
            generation_id=1,
            metric_name="exact_match",
            value=1.0,
            passed=True,
        )
        self._aggregate = SimpleNamespace(
            metric_name="exact_match",
            metric_version="1.0.0",
            metric_config_sha256="a" * 64,
            slice_key="__overall__",
            n=1,
            value=1.0,
            ci_low=0.0,
            ci_high=1.0,
            method="wilson",
        )

    async def get_run(self, run_id: uuid.UUID) -> SimpleNamespace | None:
        return self._run if run_id == RUN_ID else None

    async def get_dataset(self, dataset_id: int) -> SimpleNamespace | None:
        return self._dataset if dataset_id == 10 else None

    async def get_model_version(self, model_version_id: int) -> SimpleNamespace | None:
        return self._model if model_version_id == 20 else None

    async def get_prompt_template(self, prompt_template_id: int) -> SimpleNamespace | None:
        return self._template if prompt_template_id == 30 else None

    async def get_cases_for_dataset(self, dataset_id: int) -> list[tuple[int, Case]]:
        assert dataset_id == 10
        return [(1, self._case)]

    async def get_generations_for_run(self, run_id: uuid.UUID) -> list[SimpleNamespace]:
        assert run_id == RUN_ID
        return [self._generation]

    async def get_scores_for_run(self, run_id: uuid.UUID) -> list[SimpleNamespace]:
        assert run_id == RUN_ID
        return [self._score]

    async def get_metric_aggregates(self, run_id: uuid.UUID) -> list[SimpleNamespace]:
        assert run_id == RUN_ID
        return [self._aggregate]

    async def get_planned_generation_count(self, run_id: uuid.UUID) -> int:
        assert run_id == RUN_ID
        return 1


@pytest.mark.asyncio
async def test_build_report_uses_injected_run_store_for_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[_SessionGetForbidden]:
        yield _SessionGetForbidden()

    monkeypatch.setattr(report_io, "session_scope", fake_session_scope)

    report = await build_report(RUN_ID, coverage_floor=0.98, run_store=_FakeReportStore)

    assert report.run_id == str(RUN_ID)
    assert report.dataset_sha256 == "d" * 64
    assert report.model_digest == "mock-digest"
    assert report.dataset["name"] == "sample"
    assert report.model["provider"] == "mock"
    assert report.prompt_template["name"] == "default"
    assert report.publishable is True
    assert report.pass_rate == 1.0
    assert report.pass_rate_n == 1
