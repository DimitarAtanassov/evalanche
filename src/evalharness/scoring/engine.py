"""Batch scoring and zero-generation rescoring."""

from __future__ import annotations

import uuid
from collections import defaultdict

from evalharness.core.enums import Requirement
from evalharness.core.models import Case, Generation, ScoreValue, ScoringContext
from evalharness.core.protocols import Metric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig
from evalharness.scoring.registry import MetricRegistry
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository


class ScoringEngine:
    def __init__(self, registry: MetricRegistry | None = None, batch_size: int = 500) -> None:
        self.registry = registry or MetricRegistry.defaults()
        self.batch_size = batch_size
        self.normalizer = Normalizer(NormalizerConfig())

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

    async def rescore_run(self, run_id: uuid.UUID, metric_names: list[str]) -> int:
        """Score stored generations only. No provider is accepted or invoked."""
        saved = 0
        by_metric: dict[str, list[ScoreValue]] = defaultdict(list)
        async with session_scope() as session:
            repo = RunRepository(session)
            run = await repo.get_run(run_id)
            if run is None:
                raise ValueError(f"Run not found: {run_id}")
            cases = dict(await repo.get_cases_for_dataset(run.dataset_id))
            generations = await repo.get_generations_for_run(run_id)
            for start in range(0, len(generations), self.batch_size):
                for row in generations[start : start + self.batch_size]:
                    case = cases[row.case_id]
                    generation = await repo.generation_to_domain(row, case.external_id)
                    for score in self.score_one(generation, case, metric_names):
                        await repo.save_score(
                            generation_id=row.id,
                            metric_name=score.metric_name,
                            metric_version=score.metric_version,
                            metric_config_sha256=score.metric_config_sha256,
                            value=score.value,
                            passed=score.passed,
                            detail=score.detail,
                        )
                        by_metric[score.metric_name].append(score)
                        saved += 1
            for name, values in by_metric.items():
                metric = self.registry.get(name)
                aggregate = metric.aggregate(values)
                config_sha = values[0].metric_config_sha256
                await repo.save_metric_aggregate(
                    run_id=run_id,
                    metric_name=aggregate.metric_name,
                    metric_version=aggregate.metric_version,
                    metric_config_sha256=config_sha,
                    slice_key=aggregate.slice_key,
                    n=aggregate.n,
                    value=aggregate.value,
                    ci_low=aggregate.ci_low,
                    ci_high=aggregate.ci_high,
                    stddev=aggregate.stddev,
                    method=aggregate.method,
                )
        return saved
