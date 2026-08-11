"""Pure assembly and publication of deterministic benchmark suites."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from evalharness.domain.constants import OVERALL_SLICE, SUITE_SCHEMA_VERSION
from evalharness.hashing import judgment_identity_digest
from evalharness.observability import sanitize_text
from evalharness.suite.loader import canonical_json, load_suite
from evalharness.suite.models import (
    JsonValue,
    LoadedMember,
    LoadedSuite,
    LoadedSupplement,
    SuiteReport,
)
from evalharness.suite.render import suite_to_html

EXAMPLE_TEXT_LIMIT = 280
EXAMPLE_LIMIT_PER_MEMBER = 8
EXAMPLE_LIMIT_TOTAL = 24
JUDGE_UNBOUND_BLOCK_REASON = (
    "CALIBRATION_JUDGMENT_MISMATCH: no passing calibration in this suite binds to "
    "this judgment body"
)


def _text(value: JsonValue, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _number(value: JsonValue) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _dataset(member: LoadedMember) -> str:
    return member.declaration.dataset or _text(member.report.dataset.get("name"))


def _model(member: LoadedMember) -> str:
    return member.declaration.model or _text(member.report.model.get("model"))


def _prompt(member: LoadedMember) -> str:
    return member.declaration.prompt or _text(member.report.prompt_template.get("name"))


def _task(member: LoadedMember) -> str:
    if member.declaration.task:
        return member.declaration.task
    if member.report.case_examples:
        return _text(member.report.case_examples[0].get("task_type"), "unspecified")
    return "unspecified"


def _headline(member: LoadedMember, metrics: dict[str, str]) -> dict[str, JsonValue]:
    metric = metrics[_dataset(member)]
    aggregate = next(
        row
        for row in member.report.metric_aggregates
        if row.metric == metric and row.slice == OVERALL_SLICE
    )
    return {
        "metric": aggregate.metric,
        "value": aggregate.value,
        "n": aggregate.n,
        "ci_low": aggregate.ci_low,
        "ci_high": aggregate.ci_high,
        "method": aggregate.method,
    }


def _suite_digest(suite: LoadedSuite) -> str:
    digest_input: dict[str, JsonValue] = {
        "member_digests": sorted(member.digest for member in suite.members),
        "compare_digests": sorted(compare.digest for compare in suite.compares),
        "primary_metrics": sorted(
            (
                {"dataset": primary.dataset, "metric": primary.metric}
                for primary in suite.manifest.primary_metrics
            ),
            key=lambda item: (str(item["dataset"]), str(item["metric"])),
        ),
    }
    if suite.calibrations:
        digest_input["calibration_digests"] = sorted(item.digest for item in suite.calibrations)
    if suite.judge_artifacts:
        digest_input["judge_artifact_digests"] = sorted(
            item.digest for item in suite.judge_artifacts
        )
    if suite.rag_artifacts:
        digest_input["rag_artifact_digests"] = sorted(item.digest for item in suite.rag_artifacts)
    return hashlib.sha256(canonical_json(digest_input).encode()).hexdigest()


def _exclusion_reason(member: LoadedMember) -> str:
    report = member.report
    reasons: list[str] = []
    if report.run_status != "completed":
        reasons.append(f"run status is {report.run_status}")
    if report.written_generations != report.planned_generations:
        reasons.append(
            f"written generations {report.written_generations}/{report.planned_generations}"
        )
    if report.coverage < report.coverage_floor:
        reasons.append(f"coverage {report.coverage:.4f} below floor {report.coverage_floor:.4f}")
    return "; ".join(reasons) or "report publishable flag is false"


def _members(
    suite: LoadedSuite,
    metrics: dict[str, str],
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    members: list[dict[str, JsonValue]] = []
    exclusions: list[dict[str, JsonValue]] = []
    for member in suite.members:
        report = member.report
        members.append(
            {
                "run_id": report.run_id,
                "path": member.declaration.path,
                "artifact_sha256": member.digest,
                "role": member.declaration.role.value,
                "label": member.declaration.label,
                "publishable": report.publishable,
                "coverage": report.coverage,
                "model": _model(member),
                "prompt": _prompt(member),
                "dataset": _dataset(member),
                "domain": member.declaration.domain or "unspecified",
                "task": _task(member),
                "model_digest": report.model_digest,
                "dataset_sha256": report.dataset_sha256,
                "primary_metric": _headline(member, metrics),
            }
        )
        if not report.publishable:
            exclusions.append(
                {
                    "run_id": report.run_id,
                    "reason": _exclusion_reason(member),
                }
            )
    return members, exclusions


def _quality_tables(suite: LoadedSuite) -> list[dict[str, JsonValue]]:
    groups: defaultdict[tuple[str, str], list[dict[str, JsonValue]]] = defaultdict(list)
    for member in suite.members:
        for aggregate in member.report.metric_aggregates:
            if aggregate.slice != OVERALL_SLICE:
                continue
            row: dict[str, JsonValue] = {
                "run_id": member.report.run_id,
                "label": member.declaration.label,
                "dataset": _dataset(member),
                "dataset_sha256": member.report.dataset_sha256,
                "metric": aggregate.metric,
                "value": aggregate.value,
                "n": aggregate.n,
                "ci_low": aggregate.ci_low,
                "ci_high": aggregate.ci_high,
                "method": aggregate.method,
                "model_digest": member.report.model_digest,
            }
            groups[("domain", member.declaration.domain or "unspecified")].append(row)
            groups[("task", _task(member))].append(row)
    return [
        {
            "dimension": dimension,
            "value": value,
            "rows": sorted(
                rows,
                key=lambda row: (str(row["dataset"]), str(row["metric"]), str(row["label"])),
            ),
        }
        for (dimension, value), rows in sorted(groups.items())
    ]


def _leaderboards(
    suite: LoadedSuite,
    metrics: dict[str, str],
) -> list[dict[str, JsonValue]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, JsonValue]]] = defaultdict(list)
    for member in suite.members:
        if not member.report.publishable:
            continue
        dataset = _dataset(member)
        metric = metrics[dataset]
        headline = _headline(member, metrics)
        groups[(dataset, _task(member), metric)].append(
            {
                "run_id": member.report.run_id,
                "label": member.declaration.label,
                "role": member.declaration.role.value,
                "model": _model(member),
                "prompt": _prompt(member),
                "value": headline["value"],
                "n": headline["n"],
                "ci_low": headline["ci_low"],
                "ci_high": headline["ci_high"],
                "model_digest": member.report.model_digest,
                "dataset_sha256": member.report.dataset_sha256,
            }
        )
    return [
        {
            "dataset": dataset,
            "task": task,
            "metric": metric,
            "entries": sorted(
                entries,
                key=lambda row: (-_number(row["value"]), str(row["label"])),
            ),
        }
        for (dataset, task, metric), entries in sorted(groups.items())
    ]


def _slices(
    suite: LoadedSuite,
    metrics: dict[str, str],
) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for member in suite.members:
        metric = metrics[_dataset(member)]
        for aggregate in member.report.metric_aggregates:
            if aggregate.metric != metric:
                continue
            rows.append(
                {
                    "run_id": member.report.run_id,
                    "label": member.declaration.label,
                    "dataset": _dataset(member),
                    "model_digest": member.report.model_digest,
                    "metric": aggregate.metric,
                    "slice": aggregate.slice,
                    "overall": aggregate.slice == OVERALL_SLICE,
                    "value": aggregate.value,
                    "n": aggregate.n,
                    "ci_low": aggregate.ci_low,
                    "ci_high": aggregate.ci_high,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["run_id"]),
            not bool(row["overall"]),
            _number(row["value"]) if not bool(row["overall"]) else -1.0,
            str(row["slice"]),
        ),
    )


def _bounded_text(value: str) -> str:
    return sanitize_text(value, max_chars=EXAMPLE_TEXT_LIMIT)


def _safe_example_value(value: JsonValue) -> JsonValue:
    if not isinstance(value, str):
        return value
    return _bounded_text(value)


def _failure_gallery(suite: LoadedSuite) -> list[dict[str, JsonValue]]:
    allowed = (
        "case_id",
        "repeat_idx",
        "task_type",
        "slices",
        "input",
        "reference",
        "output",
        "outcome",
        "metric",
        "metric_value",
        "passed",
        "latency_ms",
    )
    rows: list[dict[str, JsonValue]] = []
    for member in suite.members:
        failures = [
            example
            for example in member.report.case_examples
            if example.get("passed") is False
            or str(example.get("outcome", "")).startswith("harness_")
        ]
        failures.sort(
            key=lambda example: (str(example.get("case_id")), int(example.get("repeat_idx") or 0))
        )
        for example in failures[:EXAMPLE_LIMIT_PER_MEMBER]:
            row: dict[str, JsonValue] = {
                "run_id": member.report.run_id,
                "label": member.declaration.label,
            }
            row.update(
                {key: _safe_example_value(example.get(key)) for key in allowed if key in example}
            )
            rows.append(row)
    return rows[:EXAMPLE_LIMIT_TOTAL]


def _comparisons(suite: LoadedSuite) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for comparison in suite.compares:
        rows.append(
            {
                "path": comparison.declared_path,
                "artifact_sha256": comparison.digest,
                "baseline_run_id": comparison.artifact.baseline_run_id,
                "candidate_run_id": comparison.artifact.candidate_run_id,
                "excluded_flaky_cases": comparison.artifact.excluded_flaky_cases,
                **comparison.artifact.result.model_dump(mode="json"),
            }
        )
    return rows


def _ops(suite: LoadedSuite) -> dict[str, JsonValue]:
    members: list[JsonValue] = []
    for member in suite.members:
        report = member.report
        members.append(
            {
                "run_id": report.run_id,
                "label": member.declaration.label,
                "latency_ms": dict(sorted(report.latency_ms.items())),
                "cost_usd_total": report.cost_usd_total,
                "cost_per_correct": report.cost_per_correct,
                "retries": report.retries,
                "cache_hits": report.cache_hits,
                "cache_rate": report.cache_rate,
                "harness_failures": report.harness_failures,
                "outcome_histogram": dict(sorted(report.outcome_histogram.items())),
                "finish_reasons": dict(sorted(report.finish_reasons.items())),
            }
        )
    return {"members": members}


def _object(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _calibration_summaries(
    artifacts: list[LoadedSupplement],
) -> list[dict[str, JsonValue]]:
    summaries: list[dict[str, JsonValue]] = []
    for artifact in artifacts:
        payload = artifact.payload
        holdout = _object(payload.get("holdout"))
        summaries.append(
            {
                "path": artifact.declared_path,
                "artifact_sha256": artifact.digest,
                "calibration_digest": payload.get("calibration_digest"),
                "rubric_name": payload.get("rubric_name"),
                "rubric_version": payload.get("rubric_version"),
                "agreement_metric": holdout.get("agreement_metric"),
                "agreement_holdout": holdout.get("agreement"),
                "n_holdout": holdout.get("n"),
                "threshold": payload.get("threshold"),
                "family_separation_ok": payload.get("family_separation_ok"),
                "gating_allowed": payload.get("gating_allowed") is True,
                "plain_language": payload.get("plain_language"),
            }
        )
    return summaries


def _passing_calibrations(
    artifacts: list[LoadedSupplement],
) -> dict[str, dict[str, JsonValue]]:
    """Index passing calibrations by digest; the gate bit only ever comes from here."""
    passing: dict[str, dict[str, JsonValue]] = {}
    for artifact in artifacts:
        payload = artifact.payload
        digest = payload.get("calibration_digest")
        if payload.get("gating_allowed") is True and isinstance(digest, str) and digest:
            passing[digest] = payload
    return passing


def _judgment_is_bound(
    payload: dict[str, JsonValue],
    passing_calibrations: dict[str, dict[str, JsonValue]],
) -> bool:
    """Require the judgment and calibration to reference each other, both ways.

    Membership in ``passing_calibrations`` alone only proves the judgment quoted a
    digest that exists, which any file can copy. Re-deriving the judgment body
    digest is what makes a stolen digest useless.
    """
    if payload.get("gating_allowed") is not True:
        return False
    claimed = payload.get("calibration_digest")
    if not isinstance(claimed, str):
        return False
    calibration = passing_calibrations.get(claimed)
    if calibration is None:
        return False
    return calibration.get("judgment_digest") == judgment_identity_digest(payload)


def _judge_summaries(
    artifacts: list[LoadedSupplement],
    *,
    passing_calibrations: dict[str, dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    summaries: list[dict[str, JsonValue]] = []
    for artifact in artifacts:
        payload = artifact.payload
        bound = _judgment_is_bound(payload, passing_calibrations)
        summaries.append(
            {
                "path": artifact.declared_path,
                "artifact_sha256": artifact.digest,
                "mode": payload.get("mode"),
                "rubric_name": payload.get("rubric_name"),
                "rubric_version": payload.get("rubric_version"),
                "judge_model": payload.get("judge_model"),
                "calibration_digest": payload.get("calibration_digest"),
                "gating_allowed": bound,
                "gating_block_reason": payload.get("gating_block_reason")
                if bound
                else JUDGE_UNBOUND_BLOCK_REASON,
            }
        )
    return summaries


def _rag_summaries(artifacts: list[LoadedSupplement]) -> list[dict[str, JsonValue]]:
    summaries: list[dict[str, JsonValue]] = []
    for artifact in artifacts:
        payload = artifact.payload
        retrieval = _object(payload.get("retrieval"))
        faithfulness = _object(payload.get("faithfulness"))
        citations = _object(payload.get("citations"))
        summaries.append(
            {
                "path": artifact.declared_path,
                "artifact_sha256": artifact.digest,
                "run_id": payload.get("run_id"),
                "model_digest": payload.get("model_digest"),
                "retrieval_status": retrieval.get("status"),
                "retrieval_aggregate": retrieval.get("aggregate"),
                "faithfulness_status": faithfulness.get("status"),
                "faithfulness_aggregate": faithfulness.get("aggregate"),
                "citation_attribution": citations.get("attribution"),
                "gating_allowed": False,
            }
        )
    return summaries


def assemble_suite(suite: LoadedSuite) -> SuiteReport:
    """Assemble suite.json 0.1 from already validated published artifacts."""
    metrics = {primary.dataset: primary.metric for primary in suite.manifest.primary_metrics}
    members, exclusions = _members(suite, metrics)
    return SuiteReport(
        schema_version=SUITE_SCHEMA_VERSION,
        name=suite.manifest.name,
        description=suite.manifest.description,
        suite_digest=_suite_digest(suite),
        members=members,
        exclusions=exclusions,
        coverage_matrix=[
            {
                "run_id": member.report.run_id,
                "label": member.declaration.label,
                "dataset": _dataset(member),
                "publishable": member.report.publishable,
                "coverage": member.report.coverage,
                "planned_generations": member.report.planned_generations,
                "written_generations": member.report.written_generations,
                "coverage_floor": member.report.coverage_floor,
            }
            for member in suite.members
        ],
        quality_tables=_quality_tables(suite),
        leaderboards=_leaderboards(suite, metrics),
        slices=_slices(suite, metrics),
        comparisons=_comparisons(suite),
        failure_gallery=_failure_gallery(suite),
        ops=_ops(suite),
        calibrations=_calibration_summaries(suite.calibrations) or None,
        judge_artifacts=_judge_summaries(
            suite.judge_artifacts,
            passing_calibrations=_passing_calibrations(suite.calibrations),
        )
        or None,
        rag_artifacts=_rag_summaries(suite.rag_artifacts) or None,
    )


def suite_to_json(report: SuiteReport) -> str:
    """Serialize suite.json with stable key order and a trailing newline."""
    payload = report.model_dump(mode="json")
    for optional_section in ("calibrations", "judge_artifacts", "rag_artifacts"):
        if payload[optional_section] is None:
            del payload[optional_section]
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_suite(path: Path) -> SuiteReport:
    """Load, validate, and assemble a suite from local published artifacts."""
    return assemble_suite(load_suite(path))


def write_suite_artifacts(manifest_path: Path, output_dir: Path) -> SuiteReport:
    """Build and atomically publish canonical suite JSON and offline HTML."""
    report = build_suite(manifest_path)
    contents = {
        "suite.json": suite_to_json(report),
        "suite.html": suite_to_html(report),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path]] = []
    try:
        for name, content in contents.items():
            target = output_dir / name
            temporary = output_dir / f".{name}.tmp"
            temporary.write_text(content, encoding="utf-8")
            pending.append((temporary, target))
        for temporary, target in pending:
            os.replace(temporary, target)
    except OSError:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
        raise
    return report
