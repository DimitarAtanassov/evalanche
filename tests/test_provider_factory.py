"""Provider construction and the rate limits that actually reach the built provider.

``build_managed_provider`` is the single construction point for the run, judge, and
RAG paths, and the limits it resolves are what bound production traffic. Every
assertion here reads the token buckets on the returned ``ManagedProvider`` rather
than trusting that a provider came back at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from evalharness.config import get_settings
from evalharness.providers.factory import build_managed_provider
from evalharness.providers.mock import MockProvider
from evalharness.providers.ollama import OllamaProvider
from evalharness.providers.openai_compatible import OpenAICompatibleProvider
from evalharness.providers.runtime import ManagedProvider

OPENAI_COMPATIBLE_ENV = {
    "OPENAI_COMPATIBLE_BASE_URL": "https://openai.invalid/v1",
    "OPENAI_COMPATIBLE_API_KEY": "test-key",
    "OPENAI_COMPATIBLE_MODEL_REVISION": "rev-2024-11-01",
}


@pytest.fixture(autouse=True)
def _isolate_settings_cache() -> Iterator[None]:
    """Never hand a settings object built from this module's env to another test."""
    yield
    get_settings.cache_clear()


def _apply_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str | None]) -> None:
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _managed(provider: object) -> ManagedProvider:
    assert isinstance(provider, ManagedProvider)
    return provider


async def test_ollama_provider_is_built_from_settings_and_wrapped_for_rate_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_env(monkeypatch, {"OLLAMA_BASE_URL": "http://ollama.invalid:11434"})

    managed = _managed(build_managed_provider("ollama", concurrency=3))

    assert managed.name == "ollama"
    assert isinstance(managed.provider, OllamaProvider)
    assert managed.provider.base_url == "http://ollama.invalid:11434"
    assert managed.semaphore._value == 3
    await managed.aclose()


async def test_openai_compatible_provider_carries_base_url_key_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_env(monkeypatch, dict(OPENAI_COMPATIBLE_ENV))

    managed = _managed(build_managed_provider("openai_compatible", concurrency=4))

    assert managed.name == "openai_compatible"
    assert isinstance(managed.provider, OpenAICompatibleProvider)
    assert managed.provider.model_revision == "rev-2024-11-01"
    # httpx normalizes a base URL with a trailing slash; the host and path are the point.
    assert str(managed.provider._client.base_url) == "https://openai.invalid/v1/"
    assert managed.provider._client.headers["authorization"] == "Bearer test-key"
    await managed.aclose()


@pytest.mark.parametrize(
    "missing",
    ["OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_MODEL_REVISION"],
)
def test_openai_compatible_refuses_to_build_without_base_url_or_revision(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    env: dict[str, str | None] = dict(OPENAI_COMPATIBLE_ENV)
    env[missing] = None
    _apply_env(monkeypatch, env)

    with pytest.raises(ValueError) as exc:
        build_managed_provider("openai_compatible", concurrency=2)

    assert str(exc.value) == (
        "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL_REVISION are required"
    )


async def test_unregistered_kind_resolves_through_the_entry_point_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_env(monkeypatch, {})

    managed = _managed(build_managed_provider("mock", concurrency=5))

    assert managed.name == "mock"
    assert isinstance(managed.provider, MockProvider)
    assert managed.semaphore._value == 5
    await managed.aclose()


def test_unknown_provider_name_is_rejected_by_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_env(monkeypatch, {})

    with pytest.raises(ValueError, match="Unknown provider 'not-a-provider'"):
        build_managed_provider("not-a-provider", concurrency=1)


async def test_unset_limits_keep_each_provider_kinds_own_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two built-in kinds do not share a default, and neither is the caller's.

    A single shared fallback here would silently halve Ollama's 120 rpm or double
    the OpenAI-compatible 60 rpm, and no downstream artifact would show it.
    """
    _apply_env(monkeypatch, dict(OPENAI_COMPATIBLE_ENV))

    ollama = _managed(build_managed_provider("ollama", concurrency=2, rpm=None, tpm=None))
    openai = _managed(
        build_managed_provider("openai_compatible", concurrency=2, rpm=None, tpm=None)
    )
    entry_point = _managed(build_managed_provider("mock", concurrency=2, rpm=None, tpm=None))

    assert (ollama.requests.capacity, ollama.requests.refill_per_second) == (120.0, 2.0)
    assert (ollama.tokens.capacity, ollama.tokens.refill_per_second) == (120_000.0, 2_000.0)
    assert (openai.requests.capacity, openai.requests.refill_per_second) == (60.0, 1.0)
    assert (openai.tokens.capacity, openai.tokens.refill_per_second) == (60_000.0, 1_000.0)
    assert (entry_point.requests.capacity, entry_point.tokens.capacity) == (
        1_000_000.0,
        1_000_000_000.0,
    )
    for managed in (ollama, openai, entry_point):
        await managed.aclose()


@pytest.mark.parametrize("name", ["ollama", "openai_compatible", "mock"])
async def test_explicit_limits_override_every_kinds_default(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _apply_env(monkeypatch, dict(OPENAI_COMPATIBLE_ENV))

    managed = _managed(build_managed_provider(name, concurrency=2, rpm=30, tpm=9_000))

    assert (managed.requests.capacity, managed.requests.refill_per_second) == (30.0, 0.5)
    assert (managed.tokens.capacity, managed.tokens.refill_per_second) == (9_000.0, 150.0)
    await managed.aclose()
