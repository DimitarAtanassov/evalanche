"""Façade smoke: stable public imports survive the execution module extract."""

from evalharness.execution import (
    DecodeParamsError,
    ExecutionResult,
    Executor,
    GracefulShutdown,
    ResumeError,
    RunConfig,
    RunPlanItem,
    classify_outcome,
    render_prompt,
    response_cache_key,
    validate_decode_params,
)
from evalharness.execution import executor as executor_mod
from evalharness.execution import helpers as helpers_mod


def test_execution_package_reexports_match_executor_facade() -> None:
    assert Executor is executor_mod.Executor
    assert ExecutionResult is executor_mod.ExecutionResult
    assert ResumeError is executor_mod.ResumeError
    assert classify_outcome is executor_mod.classify_outcome
    assert render_prompt is executor_mod.render_prompt
    assert response_cache_key is executor_mod.response_cache_key
    assert RunConfig is executor_mod.RunConfig
    assert RunPlanItem is executor_mod.RunPlanItem
    assert GracefulShutdown is executor_mod.GracefulShutdown
    assert validate_decode_params is helpers_mod.validate_decode_params
    assert DecodeParamsError.__name__ == "DecodeParamsError"
