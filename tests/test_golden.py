"""Golden tests for metric stability."""

import pytest

from evalharness.core.enums import FailureOutcome, FinishReason, TaskType
from evalharness.core.models import Case, Generation, ScoringContext
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig


def test_golden_exact_match_and_wilson() -> None:
    metric = ExactMatchMetric(Normalizer(NormalizerConfig()))
    case = Case(
        external_id="golden-1",
        task_type=TaskType.QA_SHORT,
        inputs={"question": "capital of France"},
        reference_answer="Paris",
    )
    gen = Generation(
        id=1,
        run_id="golden",
        case_external_id="golden-1",
        repeat_idx=0,
        output="Paris",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        outcome=FailureOutcome.PASSED,
        prompt_tokens=10,
        completion_tokens=2,
        cost_usd=0.0,
        ttft_ms=5.0,
        total_ms=15.0,
        queue_wait_ms=0.0,
        attempts=1,
        attempt_log=[],
        cached=False,
        raw_response=None,
        trace_id=None,
    )
    score = metric.score(gen, case, ScoringContext(normalizer_id=metric.normalizer.config_id))[0]
    assert score.value == 1.0
    assert score.passed is True
    assert score.detail["normalized_prediction"] == "paris"
    assert score.detail["normalized_reference"] == "paris"

    agg = metric.aggregate([score])
    assert agg.value == 1.0
    # Pinned literals, not a re-run of wilson_interval: the published lower bound moved
    # by ~1e-6 when the z constant became norm.ppf(0.975), and nothing else catches that.
    assert agg.ci_low == pytest.approx(0.20654931437723745, abs=1e-12)
    assert agg.ci_high == pytest.approx(1.0, abs=1e-12)
