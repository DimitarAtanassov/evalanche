from evalharness.core.enums import FailureOutcome, FinishReason
from evalharness.core.models import Case, TaskType
from evalharness.execution.executor import classify_outcome, render_prompt


def test_render_prompt() -> None:
    case = Case(external_id="c", task_type=TaskType.QA_SHORT, inputs={"question": "2+2?"})
    rendered = render_prompt("Q: {{question}}", case)
    assert rendered == "Q: 2+2?"


def test_classify_outcome_harness_timeout() -> None:
    assert (
        classify_outcome(
            output="x",
            finish_reason=FinishReason.STOP,
            harness_error=False,
            harness_timeout=True,
        )
        == FailureOutcome.HARNESS_TIMEOUT
    )


def test_classify_outcome_empty() -> None:
    assert (
        classify_outcome(
            output="  ",
            finish_reason=FinishReason.STOP,
            harness_error=False,
            harness_timeout=False,
        )
        == FailureOutcome.EMPTY_OUTPUT
    )
