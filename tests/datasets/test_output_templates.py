"""Constrained output templates for task shapes."""

from pathlib import Path

import pytest

from evalharness.domain.dataset import Case
from evalharness.domain.enums import TaskType
from evalharness.execution.executor import render_prompt


@pytest.mark.parametrize(
    ("template_name", "inputs", "required_instruction"),
    [
        ("qa_short.jinja", {"question": "What is two plus two?"}, "concise short answer"),
        (
            "classification.jinja",
            {"text": "A fictional match ended.", "labels": ["sports", "world"]},
            "exactly one label",
        ),
        (
            "extraction.jinja",
            {
                "text": "Two blue cups.",
                "json_schema": {"type": "object"},
            },
            "valid JSON object only",
        ),
        (
            "summarization.jinja",
            {"document": "A fictional library changed its hours."},
            "concise factual summary",
        ),
        (
            "numeric.jinja",
            {"question": "What is two plus two?"},
            "only the final numeric answer",
        ),
        (
            "retrieval.jinja",
            {"query": "fern care", "candidates": [{"id": "doc-a"}]},
            "only a JSON array",
        ),
    ],
)
def test_template_renders_a_constrained_output_contract(
    template_name: str,
    inputs: dict[str, object],
    required_instruction: str,
) -> None:
    template = (Path("fixtures/templates") / template_name).read_text(encoding="utf-8")
    case = Case(external_id="template", task_type=TaskType.GENERATION, inputs=inputs)

    rendered = render_prompt(template, case)

    assert required_instruction in rendered
    assert "{{" not in rendered
    assert "{%" not in rendered
