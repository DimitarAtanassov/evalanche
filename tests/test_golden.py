"""Golden tests for metric stability."""

from evalharness.core.enums import FailureOutcome, FinishReason, TaskType
from evalharness.core.models import Case, Generation, ScoringContext
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig
from evalharness.scoring.stats import wilson_interval


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
    low, high = wilson_interval(1, 1)
    assert agg.ci_low == low
    assert agg.ci_high == high
