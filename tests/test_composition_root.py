"""Composition-root contracts."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

from evalharness.app import AppContainer, build_container
from evalharness.app.settings import Settings
from evalharness.providers.factory import build_managed_provider
from evalharness.repositories import RunStoreUow
from evalharness.scoring.engine import ScoringEngine
from evalharness.services import CompareService, EvaluationService


def test_build_container_returns_frozen_container_with_service_fields() -> None:
    container = build_container()

    assert isinstance(container, AppContainer)
    assert {field.name for field in fields(container)} == {
        "settings",
        "build_provider",
        "metric_registry",
        "scoring_engine",
        "run_store",
        "evaluation",
        "compare",
        "scoring",
        "judge",
        "rag",
        "dataset",
        "suite",
        "matrix",
        "gates",
    }
    with pytest.raises(FrozenInstanceError):
        container.run_store = RunStoreUow  # type: ignore[misc]


def test_build_container_binds_production_collaborators() -> None:
    container = build_container()

    assert isinstance(container.settings, Settings)
    assert container.build_provider is build_managed_provider
    assert container.run_store is RunStoreUow
    assert isinstance(container.evaluation, EvaluationService)
    assert isinstance(container.compare, CompareService)


def test_injected_collaborators_override_the_production_defaults() -> None:
    """The CLI cannot inject, so a test drives the container through this seam instead."""

    def sentinel_store(_session: object) -> object:
        raise AssertionError("sentinel factory must not be invoked")

    def sentinel_provider(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sentinel builder must not be invoked")

    container = build_container(
        build_provider=sentinel_provider,  # type: ignore[arg-type]
        run_store=sentinel_store,  # type: ignore[arg-type]
    )

    assert container.build_provider is sentinel_provider
    assert container.run_store is sentinel_store
    assert container.scoring_engine().run_store is sentinel_store
    assert container.evaluation._run_store is sentinel_store
    assert container.compare._run_store is sentinel_store


def test_scoring_engine_factory_shares_run_store_with_container() -> None:
    container = build_container()

    assert container.scoring_engine().run_store is container.run_store


def test_scoring_engine_keeps_injected_run_store_factory() -> None:
    def sentinel(_session: object) -> object:
        raise AssertionError("sentinel factory must not be invoked")

    engine = ScoringEngine(run_store=sentinel)

    assert engine.run_store is sentinel


@pytest.mark.parametrize(
    ("service_type", "collaborator"),
    [
        (EvaluationService, "settings"),
        (EvaluationService, "build_provider"),
        (EvaluationService, "scoring_engine"),
        (EvaluationService, "run_store"),
        (CompareService, "run_store"),
    ],
)
def test_service_collaborator_has_no_production_default(
    service_type: type[object],
    collaborator: str,
) -> None:
    """A defaulted collaborator would let a service bypass the store the caller injected."""
    parameter = inspect.signature(service_type.__init__).parameters[collaborator]

    assert parameter.default is inspect.Parameter.empty
