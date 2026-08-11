"""Scoring use cases: rescore a stored run, or score supplied rows with no inference."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome, TaskType
from evalharness.domain.generation import Generation
from evalharness.domain.scoring import ScoreValue
from evalharness.observability import ProgressCallback
from evalharness.scoring.engine import ScoringEngineFactory

_SUPPLIED_RUN_ID = "supplied"
"""Marks a generation that was handed to us rather than produced by a stored run."""


@dataclass(frozen=True, slots=True)
class ScoredRow:
    """Scores produced for one supplied row, keyed by the id the caller gave it."""

    external_id: str
    scores: tuple[ScoreValue, ...]


class ScoringService:
    """Zero-inference scoring against a stored run or caller-supplied rows."""

    def __init__(self, *, scoring_engine: ScoringEngineFactory) -> None:
        self._scoring_engine = scoring_engine

    async def rescore_run(
        self,
        run_id: uuid.UUID,
        metric_names: Sequence[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> int:
        """Rescore stored generations with zero inference; returns the scores written."""
        return await self._scoring_engine().rescore_run(
            run_id, list(metric_names), progress=progress
        )

    def score_supplied_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        metric_names: Sequence[str],
    ) -> Iterator[ScoredRow]:
        """Score caller-supplied outputs: no provider call, nothing persisted.

        Lazy by design, so a caller streaming results still sees every row scored before
        the one that fails. Raises ``ValueError`` for an unknown metric, an unknown task
        type, or a case the metric cannot score.
        """
        engine = self._scoring_engine()
        names = list(metric_names)
        for index, row in enumerate(rows):
            case = _case_from_row(row, index)
            scores = engine.score_one(_generation_from_row(row, case.external_id), case, names)
            yield ScoredRow(external_id=case.external_id, scores=tuple(scores))


def _case_from_row(row: Mapping[str, Any], index: int) -> Case:
    return Case(
        external_id=str(row.get("id", index)),
        task_type=TaskType(row.get("task_type", "qa_short")),
        inputs=row.get("inputs", {}),
        reference_answer=row.get("reference"),
        references=row.get("references", []),
        expected_label=row.get("expected_label"),
        expected_json=row.get("expected_json"),
        qrels=row.get("qrels"),
    )


def _generation_from_row(row: Mapping[str, Any], external_id: str) -> Generation:
    return Generation(
        id=None,
        run_id=_SUPPLIED_RUN_ID,
        case_external_id=external_id,
        repeat_idx=0,
        output=row.get("output"),
        tool_calls=[],
        finish_reason=None,
        outcome=FailureOutcome.PASSED,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=0.0,
        ttft_ms=None,
        total_ms=None,
        queue_wait_ms=None,
        attempts=0,
        attempt_log=[],
        cached=False,
        raw_response=None,
        trace_id=None,
    )
