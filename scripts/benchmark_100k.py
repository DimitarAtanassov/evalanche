"""Bounded-memory synthetic 100k-case scoring benchmark."""

from __future__ import annotations

import json
import time
import tracemalloc

from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome, TaskType
from evalharness.domain.generation import Generation
from evalharness.scoring.engine import ScoringEngine


def main() -> None:
    engine = ScoringEngine()
    started = time.perf_counter()
    tracemalloc.start()
    passed = 0
    for index in range(100_000):
        expected = str(index % 100)
        case = Case(str(index), TaskType.QA_SHORT, {}, reference_answer=expected)
        generation = Generation(
            None,
            "benchmark",
            case.external_id,
            0,
            expected,
            [],
            None,
            FailureOutcome.PASSED,
            None,
            None,
            0.0,
            None,
            None,
            None,
            0,
            [],
            False,
            None,
            None,
        )
        score = engine.score_one(generation, case, ["exact_match"])[0]
        passed += int(bool(score.passed))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "cases": 100_000,
        "passed": passed,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_mib": round(peak / 1024 / 1024, 3),
    }
    print(json.dumps(result, indent=2))
    if peak > 256 * 1024 * 1024:
        raise SystemExit("memory gate exceeded")


if __name__ == "__main__":
    main()
