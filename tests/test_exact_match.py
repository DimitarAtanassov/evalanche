from evalharness.core.enums import FailureOutcome, FinishReason, TaskType
from evalharness.core.models import Case, Generation, ScoringContext
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig


def _gen(output: str) -> Generation:
    return Generation(
        id=1,
        run_id="r",
        case_external_id="c1",
        repeat_idx=0,
        output=output,
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        outcome=FailureOutcome.PASSED,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        ttft_ms=1.0,
        total_ms=2.0,
        queue_wait_ms=0.0,
        attempts=1,
        attempt_log=[],
        cached=False,
        raw_response=None,
        trace_id=None,
    )


def test_exact_match_pass() -> None:
    metric = ExactMatchMetric(Normalizer(NormalizerConfig()))
    case = Case(
        external_id="c1",
        task_type=TaskType.QA_SHORT,
        inputs={"q": "x"},
        reference_answer="Paris",
    )
    scores = metric.score(_gen("Paris"), case, ScoringContext(normalizer_id="x"))
    assert scores[0].passed is True
    assert scores[0].value == 1.0


def test_exact_match_aggregate_wilson() -> None:
    metric = ExactMatchMetric(Normalizer(NormalizerConfig()))
    case = Case(external_id="c1", task_type=TaskType.QA_SHORT, inputs={}, reference_answer="a")
    values = metric.score(_gen("a"), case, ScoringContext(normalizer_id="x"))
    agg = metric.aggregate(values)
    assert agg.method == "wilson"
    assert agg.value == 1.0
