"""Report I/O: load via RunStore, assemble, and write JSON/HTML/JUnit artifacts."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from evalharness.domain.ports import RunStoreFactory
from evalharness.observability import (
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    StageTimer,
    emit_progress,
    get_logger,
    log_context,
)
from evalharness.reporting.assemble import (
    PRIMARY_METRIC,
    RunReport,
    _dataset_context,
    _model_context,
    _prompt_context,
    _stable_decode_params,
    assemble_run_report,
)
from evalharness.reporting.render import report_to_html, report_to_json, report_to_junit
from evalharness.db.session import session_scope
from evalharness.repositories import RunStoreUow

logger = get_logger(__name__)


async def build_report(
    run_id: uuid.UUID,
    coverage_floor: float = 0.98,
    primary_metric: str = PRIMARY_METRIC,
    run_store: RunStoreFactory | None = None,
) -> RunReport:
    """Read stored scores/aggregates; reporting never mutates evaluation state.

    ``primary_metric`` names the metric the headline quality number is computed from.
    Bernoulli primaries publish a pass rate; continuous primaries (threshold ``<= 0``)
    publish the overall aggregate mean. Callers that scored a task-fit metric list must
    pass its head, or the headline reports zero observations against a metric the run
    never scored.

    ``run_store`` defaults to ``RunStoreUow``, matching executor and scoring.
    """
    timer = StageTimer()
    store = run_store or RunStoreUow
    logger.info(
        "report_build_started",
        run_id=str(run_id),
        coverage_floor=coverage_floor,
        primary_metric=primary_metric,
    )
    async with session_scope() as session:
        repo = store(session)
        run = await repo.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        dataset = await repo.get_dataset(run.dataset_id)
        model = await repo.get_model_version(run.model_version_id)
        template = await repo.get_prompt_template(run.prompt_template_id)
        cases = {
            case_id: case for case_id, case in await repo.get_cases_for_dataset(run.dataset_id)
        }
        generations = await repo.get_generations_for_run(run_id)
        scores = await repo.get_scores_for_run(run_id)
        aggregates = await repo.get_metric_aggregates(run_id)
        planned = await repo.get_planned_generation_count(run_id)
        decode_params = _stable_decode_params(dict(run.decode_params or {}))
        run_status = run.status
        config_sha256 = run.config_sha256

    report = assemble_run_report(
        run_id=str(run_id),
        run_status=run_status,
        config_sha256=config_sha256,
        model_digest=model.resolved_version if model else "",
        dataset_sha256=dataset.content_sha256 if dataset else "",
        model=_model_context(model),
        dataset=_dataset_context(dataset, case_count=len(cases)),
        prompt_template=_prompt_context(template),
        decode_params=decode_params,
        planned_generations=planned,
        generations=generations,
        scores=scores,
        aggregates=aggregates,
        cases=cases,
        coverage_floor=coverage_floor,
        primary_metric=primary_metric,
    )
    logger.info(
        "report_build_finished",
        run_id=str(run_id),
        publishable=report.publishable,
        coverage=report.coverage,
        primary_metric=report.primary_metric,
        pass_rate=report.pass_rate,
        pass_rate_n=report.pass_rate_n,
        metric_aggregates=len(report.metric_aggregates),
        case_examples=len(report.case_examples),
        duration_ms=timer.elapsed_ms,
    )
    return report


async def write_report(
    run_id: uuid.UUID,
    output_dir: Path,
    coverage_floor: float = 0.98,
    progress: ProgressCallback | None = None,
    primary_metric: str = PRIMARY_METRIC,
    run_store: RunStoreFactory | None = None,
) -> RunReport:
    timer = StageTimer()
    emit_progress(
        progress,
        ProgressEvent(PipelineStage.REPORTING, 0, 3, "Building report artifacts"),
    )
    with log_context(run_id=str(run_id)):
        report = await build_report(
            run_id,
            coverage_floor,
            primary_metric,
            run_store=run_store,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / str(run_id)
        artifacts = (
            ("json", stem.with_suffix(".json"), json.dumps(report_to_json(report), indent=2)),
            ("html", stem.with_suffix(".html"), report_to_html(report)),
            ("junit", stem.with_suffix(".xml"), report_to_junit(report)),
        )
        for index, (format_name, path, content) in enumerate(artifacts, start=1):
            path.write_text(content, encoding="utf-8")
            logger.info(
                "report_artifact_written",
                format=format_name,
                path=str(path),
                bytes=len(content.encode("utf-8")),
            )
            emit_progress(
                progress,
                ProgressEvent(
                    PipelineStage.REPORTING,
                    index,
                    len(artifacts),
                    f"Wrote {format_name}",
                    {"path": str(path)},
                ),
            )
        logger.info(
            "reporting_finished",
            artifacts=len(artifacts),
            output_dir=str(output_dir),
            duration_ms=timer.elapsed_ms,
        )
        return report
