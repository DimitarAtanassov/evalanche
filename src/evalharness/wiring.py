"""Composition root: the one module that chooses concrete collaborators.

Entry points (the CLI, scripts) build an ``AppContext`` here and hand its pieces to the
services. Every service still declares its own dependencies as optional parameters that
default to the production choice, so a direct caller keeps working and nothing below
this module imports it. There is no container and no framework: wiring is a function.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalharness.config import Settings, get_settings
from evalharness.core.ports import RunStoreFactory
from evalharness.providers.factory import ProviderBuilder, build_managed_provider
from evalharness.scoring.engine import ScoringEngine, ScoringEngineFactory
from evalharness.store.repository import RunRepository


@dataclass(frozen=True, slots=True)
class AppContext:
    """The collaborators an entry point needs, resolved once per invocation."""

    settings: Settings
    build_provider: ProviderBuilder
    scoring_engine: ScoringEngineFactory
    run_store: RunStoreFactory


def build_app_context() -> AppContext:
    """Assemble the production wiring.

    Cheap enough to call per command: settings are cached and the other fields are
    factories, so no connection, client, or engine is created here.
    """
    run_store: RunStoreFactory = RunRepository
    return AppContext(
        settings=get_settings(),
        build_provider=build_managed_provider,
        scoring_engine=lambda: ScoringEngine(run_store=run_store),
        run_store=run_store,
    )
