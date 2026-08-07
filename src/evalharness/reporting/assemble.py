"""Pure run-report assembly: RunReport, publishability, cost fields, helpers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from evalharness.core.constants import (
    OVERALL_SLICE,
    REPORT_SCHEMA_VERSION,
)
from evalharness.core.constants import (
    PRIMARY_METRIC as PRIMARY_METRIC,  # re-exported: callers import the default from here
)
from evalharness.core.enums import FailureOutcome
from evalharness.core.models import Case
from evalharness.scoring.registry import MetricRegistry
from evalharness.statistics import percentile, wilson_interval
from evalharness.store.models import DatasetRow, ModelVersionRow, PromptTemplateRow

SCHEMA_VERSION = REPORT_SCHEMA_VERSION
HARNESS_OUTCOMES = {FailureOutcome.HARNESS_ERROR.value, FailureOutcome.HARNESS_TIMEOUT.value}
EXAMPLE_LIMIT = 8
EXAMPLE_TEXT_LIMIT = 280
# Headline kinds: Bernoulli metrics publish a pass rate; continuous metrics
# (explicit threshold <= 0, e.g. retrieval_ndcg_10) publish the overall mean.
HEADLINE_PASS_RATE = "pass_rate"
HEADLINE_MEAN = "mean"


@dataclass
class RunReport:
    schema_version: str
    run_id: str
    run_status: str
    config_sha256: str
    model_digest: str
    dataset_sha256: str
    model: dict[str, Any]
    dataset: dict[str, Any]
    prompt_template: dict[str, Any]
    decode_params: dict[str, Any]
    coverage: float
    planned_generations: int
    written_generations: int
    coverage_floor: float
    publishable: bool
    primary_metric: str
    headline_kind: str
    pass_rate: float | None
    pass_rate_n: int
    pass_rate_ci: tuple[float | None, float | None]
    confidence_method: str
    flaky_cases_excluded: bool
    outcome_histogram: dict[str, int]
    harness_failures: int
    latency: dict[str, float]
    finish_reasons: dict[str, int]
    metric_aggregates: list[dict[str, Any]]
    case_examples: list[dict[str, Any]]
    cost_usd_total: float
    cost_unpriced_generations: int
    cost_per_correct: float | None
    retries: int
    cache_hits: int
    cache_rate: float
    trace_ids_sample: list[str]


def _truncate(value: str | None, limit: int = EXAMPLE_TEXT_LIMIT) -> str | None:
    """Single-line text at most ``limit`` chars, counting the ellipsis inside the cap.

    Matches ``observability.sanitize_text`` and the suite, judge, and RAG bounds so a
    published limit means the same number of characters everywhere.
    """
    if value is None:
        return None
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _format_inputs(inputs: dict[str, Any]) -> str:
    if not inputs:
        return ""
    if len(inputs) == 1:
        only = next(iter(inputs.values()))
        return _truncate(str(only)) or ""
    rendered = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _truncate(rendered) or ""


def _reference_text(case: Case) -> str | None:
    if case.reference_answer is not None:
        return _truncate(case.reference_answer)
    if case.expected_label is not None:
        return _truncate(case.expected_label)
    if case.expected_json is not None:
        return _truncate(json.dumps(case.expected_json, sort_keys=True, ensure_ascii=False))
    if case.references:
        return _truncate("; ".join(str(item) for item in case.references))
    return None


def _model_context(model: ModelVersionRow | None) -> dict[str, Any]:
    if model is None:
        return {
            "provider": "",
            "model": "",
            "resolved_version": "",
            "quantization": None,
            "params_b": None,
            "context_window": None,
            "capabilities": {},
        }
    return {
        "provider": model.provider,
        "model": model.model,
        "resolved_version": model.resolved_version,
        "quantization": model.quantization,
        "params_b": model.params_b,
        "context_window": model.context_window,
        "capabilities": dict(model.capabilities or {}),
    }


def _dataset_context(dataset: DatasetRow | None, case_count: int) -> dict[str, Any]:
    if dataset is None:
        return {
            "name": "",
            "version": "",
            "split": "",
            "content_sha256": "",
            "case_count": case_count,
            "license": None,
            "pii_scrubbed": None,
            "slice_dimensions": [],
        }
    manifest = dataset.manifest or {}
    slice_dimensions = manifest.get("slices") or []
    if not isinstance(slice_dimensions, list):
        slice_dimensions = list(slice_dimensions)
    return {
        "name": dataset.name,
        "version": dataset.version,
        "split": dataset.split,
        "content_sha256": dataset.content_sha256,
        "case_count": case_count,
        "license": manifest.get("license"),
        "pii_scrubbed": manifest.get("pii_scrubbed"),
        "slice_dimensions": [str(item) for item in slice_dimensions],
    }


def _prompt_context(template: PromptTemplateRow | None) -> dict[str, Any]:
    if template is None:
        return {"name": "", "version": "", "content_sha256": "", "body": ""}
    return {
        "name": template.name,
        "version": template.version,
        "content_sha256": template.content_sha256,
        "body": _truncate(template.body, limit=720) or "",
    }


def _stable_decode_params(decode_params: dict[str, Any] | None) -> dict[str, Any]:
    """Stable key order so JSON artifacts stay byte-reproducible."""
    if not decode_params:
        return {}
    return cast(dict[str, Any], json.loads(json.dumps(decode_params, sort_keys=True)))


def _case_examples(
    *,
    generations: list[Any],
    cases: dict[int, Case],
    scores: list[Any],
    primary_metric: str,
    limit: int = EXAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    """Bounded examples for the dashboard.

    Failures lead so a reader immediately sees what went wrong. Text is truncated and
    raw provider payloads are never included — lean reports stay lean.
    """
    primary_by_generation = {
        score.generation_id: score for score in scores if score.metric_name == primary_metric
    }
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for generation in generations:
        case = cases.get(generation.case_id)
        if case is None:
            continue
        score = primary_by_generation.get(generation.id)
        passed = None if score is None else score.passed
        if generation.outcome in HARNESS_OUTCOMES:
            priority = 0
        elif passed is False:
            priority = 1
        elif passed is True:
            priority = 3
        else:
            priority = 2
        candidates.append(
            (
                priority,
                generation.case_id,
                generation.repeat_idx,
                {
                    "case_id": case.external_id,
                    "repeat_idx": generation.repeat_idx,
                    "task_type": case.task_type.value,
                    "slices": dict(sorted((case.slices or {}).items())),
                    "input": _format_inputs(dict(case.inputs or {})),
                    "reference": _reference_text(case),
                    "output": _truncate(generation.output),
                    "outcome": generation.outcome,
                    "metric": primary_metric if score is not None else None,
                    "metric_value": score.value if score is not None else None,
                    "passed": passed,
                    "latency_ms": generation.total_ms,
                    "trace_id": generation.trace_id,
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in candidates[:limit]]


def _primary_uses_pass_rate(metric_name: str) -> bool:
    """True when ``passed`` is a meaningful quality gate for the headline.

    Metrics that set an explicit ``threshold <= 0`` (notably ``retrieval_ndcg_10``)
    mark every scored case as passed, so a Bernoulli pass rate is not a quality
    signal. Those primaries headline the overall aggregate mean instead.
    """
    try:
        metric = MetricRegistry.defaults().get(metric_name)
    except ValueError:
        return True
    config = getattr(metric, "config", None)
    if not isinstance(config, dict) or "threshold" not in config:
        return True
    return float(config["threshold"]) > 0.0


def _headline_quality(
    *,
    primary_metric: str,
    scores: list[Any],
    aggregates: list[Any],
) -> tuple[str, float | None, int, tuple[float | None, float | None], str, int]:
    """Return headline_kind, value, n, CI, confidence_method, and Bernoulli pass count."""
    primary = [
        score
        for score in scores
        if score.metric_name == primary_metric and score.passed is not None
    ]
    passed = sum(bool(score.passed) for score in primary)
    if _primary_uses_pass_rate(primary_metric):
        rate = passed / len(primary) if primary else 0.0
        return (
            HEADLINE_PASS_RATE,
            rate,
            len(primary),
            wilson_interval(passed, len(primary)),
            "Wilson 95%",
            passed,
        )
    overall = next(
        (
            row
            for row in aggregates
            if row.metric_name == primary_metric and row.slice_key == OVERALL_SLICE
        ),
        None,
    )
    if overall is None:
        return HEADLINE_MEAN, None, 0, (None, None), "mean (continuous primary)", 0
    return (
        HEADLINE_MEAN,
        float(overall.value),
        int(overall.n),
        (None, None),
        f"mean of {primary_metric}",
        0,
    )


def assemble_run_report(
    *,
    run_id: str,
    run_status: str,
    config_sha256: str,
    model_digest: str,
    dataset_sha256: str,
    model: dict[str, Any],
    dataset: dict[str, Any],
    prompt_template: dict[str, Any],
    decode_params: dict[str, Any],
    planned_generations: int,
    generations: list[Any],
    scores: list[Any],
    aggregates: list[Any],
    cases: dict[int, Case],
    coverage_floor: float = 0.98,
    primary_metric: str = PRIMARY_METRIC,
) -> RunReport:
    """Assemble a versioned run report from already-loaded run artifacts.

    Public pure seam for publishability, coverage, example truncation, and
    aggregate packaging. ``build_report`` loads rows then delegates here.
    Generation/score/aggregate objects are attribute-duck-typed (ORM rows or
    plain namespaces).

    The headline quality field is a Bernoulli pass rate when the primary metric
    has a positive threshold; continuous primaries (threshold ``<= 0``) publish
    the overall aggregate mean instead, with no invented pass-rate CI.
    """
    harness_failures = sum(row.outcome in HARNESS_OUTCOMES for row in generations)
    covered = max(0, len(generations) - harness_failures)
    coverage = covered / planned_generations if planned_generations else 0.0
    headline_kind, pass_rate, pass_rate_n, pass_rate_ci, confidence_method, passed = (
        _headline_quality(
            primary_metric=primary_metric,
            scores=scores,
            aggregates=aggregates,
        )
    )
    latencies = [float(row.total_ms) for row in generations if row.total_ms is not None]
    latency = {
        key: round(value, 2)
        for key, value in {
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
        }.items()
    }
    total_cost = sum(float(row.cost_usd) for row in generations if row.cost_usd is not None)
    unpriced = sum(1 for row in generations if row.cost_usd is None)
    retries = sum(max(0, row.attempts - 1) for row in generations)
    cached = sum(row.cached for row in generations)
    written = len(generations)
    case_examples = _case_examples(
        generations=generations,
        cases=cases,
        scores=scores,
        primary_metric=primary_metric,
    )
    return RunReport(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_status=run_status,
        config_sha256=config_sha256,
        model_digest=model_digest,
        dataset_sha256=dataset_sha256,
        model=model,
        dataset=dataset,
        prompt_template=prompt_template,
        decode_params=decode_params,
        coverage=coverage,
        planned_generations=planned_generations,
        written_generations=written,
        coverage_floor=coverage_floor,
        publishable=(
            run_status == "completed"
            and written == planned_generations
            and coverage >= coverage_floor
        ),
        primary_metric=primary_metric,
        headline_kind=headline_kind,
        pass_rate=pass_rate,
        pass_rate_n=pass_rate_n,
        pass_rate_ci=pass_rate_ci,
        confidence_method=confidence_method,
        flaky_cases_excluded=False,
        outcome_histogram=dict(sorted(Counter(row.outcome for row in generations).items())),
        harness_failures=harness_failures,
        latency=latency,
        finish_reasons=dict(
            sorted(Counter(row.finish_reason or "unknown" for row in generations).items())
        ),
        metric_aggregates=[
            {
                "metric": row.metric_name,
                "version": row.metric_version,
                "config_sha256": row.metric_config_sha256,
                "slice": row.slice_key,
                "n": row.n,
                "value": row.value,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "method": row.method,
            }
            for row in aggregates
        ],
        case_examples=case_examples,
        cost_usd_total=total_cost,
        cost_unpriced_generations=unpriced,
        cost_per_correct=(
            total_cost / passed if headline_kind == HEADLINE_PASS_RATE and passed else None
        ),
        retries=retries,
        cache_hits=cached,
        cache_rate=cached / len(generations) if generations else 0.0,
        trace_ids_sample=list(dict.fromkeys(row.trace_id for row in generations if row.trace_id))[
            :10
        ],
    )
