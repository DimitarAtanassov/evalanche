"""Composition-root wiring contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from evalharness.config import Settings
from evalharness.pipeline.run import resolve_run_store_and_scoring_engine
from evalharness.providers.factory import build_managed_provider
from evalharness.scoring.engine import ScoringEngine
from evalharness.store.repository import RunRepository
from evalharness.wiring import AppContext, build_app_context


def test_build_app_context_returns_frozen_context_with_four_fields() -> None:
    context = build_app_context()

    assert isinstance(context, AppContext)
    assert {field.name for field in fields(context)} == {
        "settings",
        "build_provider",
        "scoring_engine",
        "run_store",
    }
    with pytest.raises(FrozenInstanceError):
        context.run_store = RunRepository  # type: ignore[misc]


def test_build_app_context_binds_production_collaborators() -> None:
    context = build_app_context()

    assert isinstance(context.settings, Settings)
    assert context.build_provider is build_managed_provider
    assert context.run_store is RunRepository


def test_scoring_engine_factory_shares_run_store_with_context() -> None:
    context = build_app_context()

    assert context.scoring_engine().run_store is context.run_store


def test_scoring_engine_keeps_injected_run_store_factory() -> None:
    def sentinel(_session: object) -> object:
        raise AssertionError("sentinel factory must not be invoked")

    engine = ScoringEngine(run_store=sentinel)

    assert engine.run_store is sentinel


def test_resolve_defaults_couple_scoring_engine_to_injected_run_store() -> None:
    def sentinel(_session: object) -> object:
        raise AssertionError("sentinel factory must not be invoked")

    store, make_engine = resolve_run_store_and_scoring_engine(sentinel, None)

    assert store is sentinel
    assert make_engine().run_store is sentinel
