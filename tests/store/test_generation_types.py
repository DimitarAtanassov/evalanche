"""GenerationRow.tool_calls typing and domain mapping consistency."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from evalharness.core.enums import FailureOutcome, FinishReason
from evalharness.store.models import GenerationRow
from evalharness.store.repository import RunRepository


def test_generation_row_tool_calls_mapped_as_list() -> None:
    annotation = GenerationRow.__annotations__["tool_calls"]
    assert "list[dict[str, Any]]" in annotation
    assert annotation.startswith("Mapped[")


@pytest.mark.asyncio
async def test_generation_to_domain_preserves_tool_calls_list() -> None:
    tool_calls = [{"id": "call_1", "name": "lookup", "arguments": {"q": "x"}}]
    row = SimpleNamespace(
        id=42,
        run_id=uuid.uuid4(),
        repeat_idx=0,
        output="hello",
        tool_calls=tool_calls,
        finish_reason=FinishReason.STOP.value,
        outcome=FailureOutcome.PASSED.value,
        prompt_tokens=3,
        completion_tokens=2,
        cost_usd=None,
        ttft_ms=1.0,
        total_ms=5.0,
        queue_wait_ms=0.0,
        attempts=1,
        attempt_log=[],
        cached=False,
        raw_response={"ok": True},
        trace_id="trace-1",
        created_at=datetime.now(UTC),
    )
    repo = RunRepository(session=None)  # type: ignore[arg-type]

    generation = await repo.generation_to_domain(row, "case-ext-1")  # type: ignore[arg-type]

    assert generation.tool_calls == tool_calls
    assert isinstance(generation.tool_calls, list)


@pytest.mark.asyncio
async def test_generation_to_domain_non_list_tool_calls_becomes_empty() -> None:
    row = SimpleNamespace(
        id=43,
        run_id=uuid.uuid4(),
        repeat_idx=0,
        output=None,
        tool_calls={"legacy": "dict"},
        finish_reason=None,
        outcome=FailureOutcome.HARNESS_ERROR.value,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
        ttft_ms=None,
        total_ms=None,
        queue_wait_ms=None,
        attempts=1,
        attempt_log=None,
        cached=False,
        raw_response=None,
        trace_id=None,
    )
    repo = RunRepository(session=None)  # type: ignore[arg-type]

    generation = await repo.generation_to_domain(row, "case-ext-2")  # type: ignore[arg-type]

    assert generation.tool_calls == []
