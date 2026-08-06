"""Exact match metric."""

from __future__ import annotations

from evalharness.core.constants import OVERALL_SLICE
from evalharness.core.enums import Requirement, TaskType
from evalharness.core.models import AggregateValue, Case, Generation, ScoreValue, ScoringContext
from evalharness.scoring.normalizer import Normalizer
from evalharness.statistics import wilson_interval


class ExactMatchMetric:
    name = "exact_match"
    version = "1.0.0"
    task_types = frozenset(
        {
            TaskType.GENERATION,
            TaskType.QA_SHORT,
            TaskType.SUMMARIZATION,
            TaskType.RAG,
        }
    )
    requires = frozenset({Requirement.REFERENCE})

    def __init__(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]:
        reference = case.reference_answer
        if reference is None and case.references:
            reference = case.references[0]
        if reference is None or gen.output is None:
            return [
                ScoreValue(
                    metric_name=self.name,
                    metric_version=self.version,
                    metric_config_sha256=self.normalizer.config_id,
                    value=None,
                    passed=None,
                    detail={"reason": "missing_reference_or_output"},
                )
            ]
        pred = self.normalizer.normalize(gen.output)
        gold = self.normalizer.normalize(reference)
        passed = pred == gold
        return [
            ScoreValue(
                metric_name=self.name,
                metric_version=self.version,
                metric_config_sha256=self.normalizer.config_id,
                value=1.0 if passed else 0.0,
                passed=passed,
                detail={"normalized_prediction": pred, "normalized_reference": gold},
            )
        ]

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [v for v in values if v.value is not None]
        n = len(valid)
        if n == 0:
            return AggregateValue(
                metric_name=self.name,
                metric_version=self.version,
                slice_key=OVERALL_SLICE,
                n=0,
                value=0.0,
                ci_low=None,
                ci_high=None,
                stddev=None,
                method="wilson",
            )
        successes = sum(1 for v in valid if v.passed)
        rate = successes / n
        ci_low, ci_high = wilson_interval(successes, n)
        return AggregateValue(
            metric_name=self.name,
            metric_version=self.version,
            slice_key=OVERALL_SLICE,
            n=n,
            value=rate,
            ci_low=ci_low,
            ci_high=ci_high,
            stddev=None,
            method="wilson",
        )
