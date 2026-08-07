"""Batch scoring and zero-generation rescoring."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable

from evalharness.core.constants import OVERALL_SLICE as OVERALL_SLICE
from evalharness.core.enums import Requirement
from evalharness.core.models import Case, Generation, ScoreValue, ScoringContext
from evalharness.core.ports import RunStoreFactory
from evalharness.core.protocols import Metric
from evalharness.observability import (
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    StageTimer,
    emit_progress,
    get_logger,
    get_tracer,
    log_context,
    payload_summary,
)
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig
from evalharness.scoring.registry import MetricRegistry
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository

logger = get_logger(__name__)


def slice_key(dimension: str, value: str) -> str:
    return f"{dimension}={value}"


class ScoringEngine:
    def __init__(
        self,
        registry: MetricRegistry | None = None,
        batch_size: int = 500,
        max_slice_cardinality: int = 50,
        *,
        run_store: RunStoreFactory | None = None,
    ) -> None:
        self.registry = registry or MetricRegistry.discover()
        self.batch_size = batch_size
        self.max_slice_cardinality = max_slice_cardinality
        self.normalizer = Normalizer(NormalizerConfig())
        self.run_store: RunStoreFactory = run_store or RunRepository

    def rollup_dimensions(self, cases: Iterable[Case]) -> set[str]:
        """Slice dimensions worth rolling up.

        A dimension whose values are near-unique (a case id smuggled into ``slices``)
        would emit one aggregate row per case, so anything above
        ``max_slice_cardinality`` distinct values is skipped rather than written.
        """
        values: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            for dimension, value in case.slices.items():
                values[dimension].add(value)
        return {
            dimension
            for dimension, seen in values.items()
            if len(seen) <= self.max_slice_cardinality
        }

    def validate(self, metric: Metric, case: Case) -> None:
        if case.task_type not in metric.task_types:
            raise ValueError(f"{metric.name} does not support task type {case.task_type}")
        if Requirement.REFERENCE in metric.requires and not (
            case.reference_answer is not None or case.references
        ):
            raise ValueError(f"{metric.name} requires a reference")
        if Requirement.QRELS in metric.requires and case.qrels is None:
            raise ValueError(f"{metric.name} requires qrels")

    def score_one(
        self,
        generation: Generation,
        case: Case,
        metric_names: list[str],
    ) -> list[ScoreValue]:
        context = ScoringContext(normalizer_id=self.normalizer.config_id)
        scores: list[ScoreValue] = []
        for name in metric_names:
            metric = self.registry.get(name)
            self.validate(metric, case)
            scores.extend(metric.score(generation, case, context))
        return scores

    async def rescore_run(
        self,
        run_id: uuid.UUID,
        metric_names: list[str],
        progress: ProgressCallback | None = None,
    ) -> int:
        """Score stored generations only. No provider is accepted or invoked."""
        timer = StageTimer()
        saved = 0
        grouped: dict[tuple[str, str], list[ScoreValue]] = defaultdict(list)
        with log_context(run_id=str(run_id), metrics=metric_names):
            async with session_scope() as session:
                repo = self.run_store(session)
                run = await repo.get_run(run_id)
                if run is None:
                    raise ValueError(f"Run not found: {run_id}")
                cases = dict(await repo.get_cases_for_dataset(run.dataset_id))
                dimensions = self.rollup_dimensions(cases.values())
                generations = await repo.get_generations_for_run(run_id)
                total = len(generations)
                logger.info(
                    "scoring_started",
                    generations=total,
                    batch_size=self.batch_size,
                    slice_dimensions=sorted(dimensions),
                )
                emit_progress(
                    progress,
                    ProgressEvent(PipelineStage.SCORING, 0, total, "Scoring generations"),
                )
                with get_tracer().start_as_current_span("run.score") as span:
                    span.set_attribute("eval.run_id", str(run_id))
                    span.set_attribute("eval.generation_count", total)
                    for start in range(0, total, self.batch_size):
                        batch = generations[start : start + self.batch_size]
                        for row in batch:
                            case = cases[row.case_id]
                            keys = [OVERALL_SLICE] + [
                                slice_key(dimension, value)
                                for dimension, value in sorted(case.slices.items())
                                if dimension in dimensions
                            ]
                            generation = await repo.generation_to_domain(row, case.external_id)
                            scores = self.score_one(generation, case, metric_names)
                            logger.debug(
                                "generation_scored",
                                generation_id=row.id,
                                case_external_id=case.external_id,
                                generation_output=payload_summary(generation.output),
                                scores=[
                                    {
                                        "metric": score.metric_name,
                                        "value": score.value,
                                        "passed": score.passed,
                                    }
                                    for score in scores
                                ],
                                slices=keys,
                            )
                            for score in scores:
                                await repo.save_score(
                                    generation_id=row.id,
                                    metric_name=score.metric_name,
                                    metric_version=score.metric_version,
                                    metric_config_sha256=score.metric_config_sha256,
                                    value=score.value,
                                    passed=score.passed,
                                    detail=score.detail,
                                )
                                for key in keys:
                                    grouped[(score.metric_name, key)].append(score)
                                saved += 1
                        completed = min(start + len(batch), total)
                        logger.info(
                            "scoring_batch_finished",
                            completed=completed,
                            total=total,
                            scores_processed=saved,
                        )
                        emit_progress(
                            progress,
                            ProgressEvent(
                                PipelineStage.SCORING,
                                completed,
                                total,
                                "Scoring generations",
                                {"scores": saved},
                            ),
                        )

                aggregate_total = len(grouped)
                emit_progress(
                    progress,
                    ProgressEvent(
                        PipelineStage.AGGREGATING,
                        0,
                        aggregate_total,
                        "Aggregating metrics and slices",
                    ),
                )
                for index, ((name, key), values) in enumerate(sorted(grouped.items()), start=1):
                    metric = self.registry.get(name)
                    aggregate = metric.aggregate(values)
                    await repo.save_metric_aggregate(
                        run_id=run_id,
                        metric_name=aggregate.metric_name,
                        metric_version=aggregate.metric_version,
                        metric_config_sha256=values[0].metric_config_sha256,
                        slice_key=key,
                        n=aggregate.n,
                        value=aggregate.value,
                        ci_low=aggregate.ci_low,
                        ci_high=aggregate.ci_high,
                        stddev=aggregate.stddev,
                        method=aggregate.method,
                    )
                    logger.debug(
                        "aggregate_written",
                        metric=name,
                        slice=key,
                        n=aggregate.n,
                        value=aggregate.value,
                        ci_low=aggregate.ci_low,
                        ci_high=aggregate.ci_high,
                        method=aggregate.method,
                    )
                    emit_progress(
                        progress,
                        ProgressEvent(
                            PipelineStage.AGGREGATING,
                            index,
                            aggregate_total,
                            key,
                            {"metric": name},
                        ),
                    )
            logger.info(
                "scoring_finished",
                generations=len(generations),
                scores_processed=saved,
                aggregates=len(grouped),
                duration_ms=timer.elapsed_ms,
            )
        return saved


type ScoringEngineFactory = Callable[[], ScoringEngine]
"""Produces an engine already bound to its store, so callers never wire one themselves."""
