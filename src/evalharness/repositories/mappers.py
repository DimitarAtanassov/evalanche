"""ORM row to domain type mapping.

The only place that reads ORM attributes into domain objects, so no repository leaks a
row past the persistence boundary.
"""

from __future__ import annotations

from typing import Any

from evalharness.db.models import (
    CaseRow,
    DatasetRow,
    GenerationRow,
    MetricAggregateRow,
    ModelVersionRow,
    PromptTemplateRow,
    RunRow,
    ScoreRow,
)
from evalharness.domain.dataset import Case, DatasetRef
from evalharness.domain.enums import FailureOutcome, FinishReason, TaskType
from evalharness.domain.generation import StoredGeneration
from evalharness.domain.run import ModelVersionRef, PromptTemplateRef, RunRecord
from evalharness.domain.scoring import StoredAggregate, StoredScore


def _run_record(row: RunRow) -> RunRecord:
    return RunRecord(
        id=row.id,
        dataset_id=row.dataset_id,
        prompt_template_id=row.prompt_template_id,
        model_version_id=row.model_version_id,
        decode_params=dict(row.decode_params or {}),
        config_sha256=row.config_sha256,
        harness_version=row.harness_version,
        git_sha=row.git_sha,
        repeats=row.repeats,
        status=row.status,
        tenant_id=row.tenant_id,
        started_at=row.started_at,
        finished_at=row.finished_at,
        baseline_run_id=row.baseline_run_id,
    )


def _dataset_ref(row: DatasetRow) -> DatasetRef:
    return DatasetRef(
        id=row.id,
        name=row.name,
        version=row.version,
        content_sha256=row.content_sha256,
        split=row.split,
        manifest=dict(row.manifest or {}),
    )


def _model_version_ref(row: ModelVersionRow) -> ModelVersionRef:
    return ModelVersionRef(
        id=row.id,
        provider=row.provider,
        model=row.model,
        resolved_version=row.resolved_version,
        quantization=row.quantization,
        params_b=row.params_b,
        context_window=row.context_window,
        capabilities=dict(row.capabilities or {}),
    )


def _prompt_template_ref(row: PromptTemplateRow) -> PromptTemplateRef:
    return PromptTemplateRef(
        id=row.id,
        name=row.name,
        version=row.version,
        body=row.body,
        content_sha256=row.content_sha256,
    )


def _case(row: CaseRow) -> Case:
    reference = row.reference or {}
    return Case(
        external_id=row.external_id,
        task_type=TaskType(row.task_type),
        inputs=row.inputs,
        reference_answer=reference.get("reference_answer"),
        references=reference.get("references", []),
        expected_label=reference.get("expected_label"),
        expected_json=reference.get("expected_json"),
        qrels=row.qrels,
        slices=row.slices or {},
        must_contain=reference.get("must_contain", []),
        must_not_contain=reference.get("must_not_contain", []),
    )


def _case_reference(case: Case) -> dict[str, Any]:
    """The reference payload persisted alongside a case, mirroring ``_case``."""
    return {
        "reference_answer": case.reference_answer,
        "references": case.references,
        "expected_label": case.expected_label,
        "expected_json": case.expected_json,
        "must_contain": case.must_contain,
        "must_not_contain": case.must_not_contain,
    }


def _stored_generation(row: GenerationRow, case_external_id: str) -> StoredGeneration:
    return StoredGeneration(
        id=row.id,
        run_id=str(row.run_id),
        case_id=row.case_id,
        case_external_id=case_external_id,
        repeat_idx=row.repeat_idx,
        output=row.output,
        tool_calls=list(row.tool_calls) if isinstance(row.tool_calls, list) else [],
        finish_reason=FinishReason(row.finish_reason) if row.finish_reason else None,
        outcome=FailureOutcome(row.outcome),
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        cost_usd=float(row.cost_usd) if row.cost_usd is not None else None,
        ttft_ms=row.ttft_ms,
        total_ms=row.total_ms,
        queue_wait_ms=row.queue_wait_ms,
        attempts=row.attempts,
        attempt_log=list(row.attempt_log) if isinstance(row.attempt_log, list) else [],
        cached=row.cached,
        raw_response=row.raw_response,
        trace_id=row.trace_id,
    )


def _stored_score(row: ScoreRow) -> StoredScore:
    return StoredScore(
        id=row.id,
        generation_id=row.generation_id,
        metric_name=row.metric_name,
        metric_version=row.metric_version,
        metric_config_sha256=row.metric_config_sha256,
        value=row.value,
        passed=row.passed,
        detail=dict(row.detail) if isinstance(row.detail, dict) else row.detail,
    )


def _stored_aggregate(row: MetricAggregateRow) -> StoredAggregate:
    return StoredAggregate(
        id=row.id,
        run_id=str(row.run_id),
        metric_name=row.metric_name,
        metric_version=row.metric_version,
        metric_config_sha256=row.metric_config_sha256,
        slice_key=row.slice_key,
        n=row.n,
        value=float(row.value),
        ci_low=row.ci_low,
        ci_high=row.ci_high,
        stddev=row.stddev,
        method=row.method,
    )
