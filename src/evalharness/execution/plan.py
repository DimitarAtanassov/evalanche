"""Work out which (case, repeat) generations a run still owes."""

from __future__ import annotations

import uuid

from evalharness.app.settings import Settings
from evalharness.domain.ports import RunStoreFactory
from evalharness.execution.models import RunConfig, RunPlanItem
from evalharness.db.session import session_scope


async def build_run_plan(
    run_id: uuid.UUID,
    *,
    run_store: RunStoreFactory,
    settings: Settings,
) -> tuple[RunConfig, list[RunPlanItem]]:
    """Return the run's execution budget and the generations not yet persisted.

    Replanning an in-flight run is how resume works: completed keys are read fresh each
    time, so the same call answers "what is left" both before and after execution.
    """
    async with session_scope() as session:
        repo = run_store(session)
        run = await repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        completed = await repo.get_completed_keys(run_id)
        cases = await repo.get_cases_for_dataset(run.dataset_id)
        items = [
            RunPlanItem(case_db_id=case_db_id, case=case, repeat_idx=repeat_idx)
            for case_db_id, case in cases
            for repeat_idx in range(run.repeats)
            if (case_db_id, repeat_idx) not in completed
        ]
        config = RunConfig(
            dataset_id=run.dataset_id,
            prompt_template_id=run.prompt_template_id,
            model_version_id=run.model_version_id,
            config_sha256=run.config_sha256,
            decode_params=run.decode_params,
            repeats=run.repeats,
            concurrency=settings.default_concurrency,
            case_timeout_s=settings.default_case_timeout_s,
            request_timeout_s=settings.default_request_timeout_s,
            run_timeout_s=settings.default_run_timeout_s,
            drain_timeout_s=settings.default_shutdown_drain_timeout_s,
            max_retries=settings.default_max_retries,
        )
        return config, items
