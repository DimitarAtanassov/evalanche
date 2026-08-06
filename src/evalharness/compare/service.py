"""Paired comparison of two stored runs on a single metric."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from evalharness.core.constants import COMPARE_SCHEMA_VERSION
from evalharness.statistics import apply_multiplicity, compare_binary
from evalharness.store.db import session_scope
from evalharness.store.models import CaseRow, GenerationRow, ScoreRow
from evalharness.store.repository import RunRepository


async def compare_runs(
    baseline_id: uuid.UUID,
    candidate_id: uuid.UUID,
    metric: str,
    allow_compatible: bool,
) -> dict[str, Any]:
    """Compare aligned case/repeat outcomes of two runs.

    Both runs must be identical in dataset, prompt template, and configuration hash.
    ``allow_compatible`` relaxes that to a deliberate same-dataset, same-repeats
    comparison. Cases whose repeats disagree in either arm are flaky and are excluded
    from the paired test rather than counted as evidence.
    """
    async with session_scope() as session:
        repo = RunRepository(session)
        baseline_run = await repo.get_run(baseline_id)
        candidate_run = await repo.get_run(candidate_id)
        if baseline_run is None or candidate_run is None:
            raise ValueError("Both runs must exist")
        identity = (
            baseline_run.dataset_id == candidate_run.dataset_id
            and baseline_run.prompt_template_id == candidate_run.prompt_template_id
            and baseline_run.config_sha256 == candidate_run.config_sha256
        )
        compatible = (
            baseline_run.dataset_id == candidate_run.dataset_id
            and baseline_run.repeats == candidate_run.repeats
        )
        if not identity and not (allow_compatible and compatible):
            raise ValueError(
                "Dataset/template/config mismatch; pass --allow-compatible only for a "
                "deliberate same-dataset comparison"
            )

        async def outcomes(run_id: uuid.UUID) -> dict[tuple[str, int], bool]:
            statement = (
                select(CaseRow.external_id, GenerationRow.repeat_idx, ScoreRow.passed)
                .join(GenerationRow, ScoreRow.generation_id == GenerationRow.id)
                .join(CaseRow, GenerationRow.case_id == CaseRow.id)
                .where(
                    GenerationRow.run_id == run_id,
                    ScoreRow.metric_name == metric,
                    ScoreRow.passed.is_not(None),
                )
            )
            return {
                (case_id, repeat): bool(passed)
                for case_id, repeat, passed in (await session.execute(statement)).all()
            }

        baseline = await outcomes(baseline_id)
        candidate = await outcomes(candidate_id)
        keys = sorted(set(baseline) & set(candidate))
        baseline_by_case: dict[str, set[bool]] = {}
        candidate_by_case: dict[str, set[bool]] = {}
        for case_id, repeat in keys:
            baseline_by_case.setdefault(case_id, set()).add(baseline[(case_id, repeat)])
            candidate_by_case.setdefault(case_id, set()).add(candidate[(case_id, repeat)])
        flaky = {
            case_id
            for case_id in baseline_by_case
            if len(baseline_by_case[case_id]) > 1 or len(candidate_by_case[case_id]) > 1
        }
        stable_keys = [key for key in keys if key[0] not in flaky]
        result = apply_multiplicity(
            [
                compare_binary(
                    metric,
                    [baseline[key] for key in stable_keys],
                    [candidate[key] for key in stable_keys],
                )
            ]
        )[0]
        return {
            "schema_version": COMPARE_SCHEMA_VERSION,
            "baseline_run_id": str(baseline_id),
            "candidate_run_id": str(candidate_id),
            "excluded_flaky_cases": sorted(flaky),
            "result": result.to_dict(),
        }
