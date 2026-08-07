"""Secret handling on Settings: API key SecretStr and DB URL redaction."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from evalharness.config import Settings, get_settings, redacted_database_url
from evalharness.providers.factory import build_managed_provider
from evalharness.providers.openai_compatible import OpenAICompatibleProvider
from evalharness.providers.runtime import ManagedProvider


@pytest.fixture(autouse=True)
def _isolate_settings_cache() -> Iterator[None]:
    yield
    get_settings.cache_clear()


def test_openai_compatible_api_key_is_secret_str_and_hidden_in_repr() -> None:
    settings = Settings(openai_compatible_api_key=SecretStr("super-secret-key"))

    assert isinstance(settings.openai_compatible_api_key, SecretStr)
    assert settings.openai_compatible_api_key.get_secret_value() == "super-secret-key"
    rendered = repr(settings)
    assert "super-secret-key" not in rendered
    assert "**********" in rendered


async def test_factory_passes_plaintext_api_key_via_get_secret_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://openai.invalid/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "plaintext-from-env")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL_REVISION", "rev-1")
    get_settings.cache_clear()

    settings = get_settings()
    assert isinstance(settings.openai_compatible_api_key, SecretStr)
    assert settings.openai_compatible_api_key.get_secret_value() == "plaintext-from-env"

    captured: dict[str, object] = {}
    real_init = OpenAICompatibleProvider.__init__

    def spy_init(
        self: OpenAICompatibleProvider,
        base_url: str,
        model_revision: str,
        api_key: str | None = None,
        organization: str | None = None,
    ) -> None:
        captured["api_key"] = api_key
        real_init(
            self,
            base_url,
            model_revision,
            api_key=api_key,
            organization=organization,
        )

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", spy_init)

    managed = build_managed_provider("openai_compatible", concurrency=1)
    assert isinstance(managed, ManagedProvider)
    assert isinstance(managed.provider, OpenAICompatibleProvider)
    assert captured["api_key"] == "plaintext-from-env"
    assert isinstance(captured["api_key"], str)
    assert not isinstance(captured["api_key"], SecretStr)
    await managed.aclose()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql+asyncpg://evalharness:s3cret@localhost:5432/evalharness",
            "postgresql+asyncpg://localhost:5432/evalharness",
        ),
        (
            "postgresql://user:p%40ss@db.example.com/app",
            "postgresql://db.example.com/app",
        ),
        (
            "postgresql://localhost:5432/evalharness",
            "postgresql://localhost:5432/evalharness",
        ),
        (
            "postgresql://user@[::1]:5432/db",
            "postgresql://[::1]:5432/db",
        ),
    ],
)
def test_redacted_database_url_strips_userinfo(url: str, expected: str) -> None:
    assert redacted_database_url(url) == expected


def test_settings_database_url_omitted_from_repr_and_property_is_redacted() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://evalharness:hunter2@localhost:5432/evalharness"
    )

    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings)
    assert settings.database_url_for_logs == ("postgresql+asyncpg://localhost:5432/evalharness")
    # Engine path still sees the real URL.
    assert "hunter2" in settings.database_url
