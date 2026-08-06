"""Resolve a provider name to a rate-limited provider for CLI entry points."""

from __future__ import annotations

from evalharness.config import get_settings
from evalharness.core.protocols import Provider
from evalharness.providers.config import OllamaConfig, OpenAICompatibleConfig, ProviderConfig
from evalharness.providers.registry import create_provider, load_provider
from evalharness.providers.runtime import ManagedProvider

# Entry-point adapters have no config dataclass to carry a limit, and the run path never
# rate limited them, so an unspecified limit stays effectively unbounded instead of
# inventing a cap that would throttle an offline provider.
_UNBOUNDED_RPM = 1_000_000
_UNBOUNDED_TPM = 1_000_000_000


def build_managed_provider(
    name: str,
    *,
    concurrency: int,
    rpm: int | None = None,
    tpm: int | None = None,
) -> Provider:
    """Build a rate-limited, circuit-broken provider for ``name``.

    ``rpm``/``tpm`` of ``None`` keeps each provider kind's own default limit; the built-in
    kinds do not share one, so a single caller-supplied pair cannot stand in for them.
    Names outside the two configured kinds resolve through the entry-point registry.
    """
    settings = get_settings()
    config: ProviderConfig
    if name == "ollama":
        config = OllamaConfig(base_url=settings.ollama_base_url, concurrency=concurrency)
    elif name == "openai_compatible":
        if (
            settings.openai_compatible_base_url is None
            or settings.openai_compatible_model_revision is None
        ):
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL_REVISION are required"
            )
        config = OpenAICompatibleConfig(
            base_url=settings.openai_compatible_base_url,
            api_key=settings.openai_compatible_api_key,
            model_revision=settings.openai_compatible_model_revision,
            concurrency=concurrency,
        )
    else:
        return ManagedProvider(
            load_provider(name),
            rpm=rpm if rpm is not None else _UNBOUNDED_RPM,
            tpm=tpm if tpm is not None else _UNBOUNDED_TPM,
            concurrency=concurrency,
        )

    overrides: dict[str, int] = {}
    if rpm is not None:
        overrides["rpm"] = rpm
    if tpm is not None:
        overrides["tpm"] = tpm
    return create_provider(config.model_copy(update=overrides))
