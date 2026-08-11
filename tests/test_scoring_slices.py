"""Slice rollup selection in the scoring engine."""

from __future__ import annotations

from evalharness.domain.dataset import Case
from evalharness.domain.enums import TaskType
from evalharness.scoring.engine import ScoringEngine, slice_key


def _case(external_id: str, slices: dict[str, str]) -> Case:
    return Case(
        external_id=external_id,
        task_type=TaskType.QA_SHORT,
        inputs={"question": "q"},
        reference_answer="a",
        slices=slices,
    )


def test_slice_key_format() -> None:
    assert slice_key("difficulty", "hard") == "difficulty=hard"


def test_rollup_dimensions_keeps_low_cardinality_dimensions() -> None:
    cases = [
        _case("a", {"difficulty": "easy", "lang": "en"}),
        _case("b", {"difficulty": "hard", "lang": "en"}),
    ]
    assert ScoringEngine().rollup_dimensions(cases) == {"difficulty", "lang"}


def test_rollup_dimensions_skips_near_unique_dimensions() -> None:
    """A case id smuggled into slices would emit one aggregate row per case."""
    cases = [_case(f"case-{index}", {"uid": f"u{index}"}) for index in range(10)]
    engine = ScoringEngine(max_slice_cardinality=5)
    assert engine.rollup_dimensions(cases) == set()


def test_rollup_dimensions_handles_cases_without_slices() -> None:
    assert ScoringEngine().rollup_dimensions([_case("a", {})]) == set()
